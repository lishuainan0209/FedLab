# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import torch
from torch.multiprocessing import Queue

from .handler import ServerHandler
from ..network import DistNetwork
from ..network_manager import NetworkManager
from ..coordinator import Coordinator
from ...utils import Logger
from ..communicator.package import MessageCode

DEFAULT_SERVER_RANK = 0


class ServerManager(NetworkManager):
    """Base class of ServerManager.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ServerHandler): Performe global model update procedure.
    """

    def __init__(self, network: DistNetwork, handler: ServerHandler):
        super().__init__(network)
        self._handler = handler
        self.coordinator = None  # initialized in setup stage.

    def setup(self):
        """Initialization Stage.

        - Server accept local client num report from client manager.
        - Init a coordinator for client_id -> rank mapping.
        """
        super().setup()
        rank_client_num_map = {}
        # 记录rank与rank对应进程中的虚拟客户端数量
        # 如果你在循环中先执行 recv(src=1)，而此时 rank=1 还未发送消息，当前进程会卡在这个 recv 调用上，不会继续执行后续的 recv(src=2)。
        # 即使此时 rank=2 已经发送了消息，由于当前进程正在等待 rank=1 的消息，rank=2 的消息会被暂时缓存（存在通信缓冲区中），不会丢失。
        # dict遍历时, 是按照加入顺序遍历的, Coordinator的map_id方法是使用字典遍历分类global_id的
        # note 上面两点保证了rank较小的进程客户端分配小id
        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_num_map[rank] = content[0].item()  # 记录rank与其下虚拟客户端对应的全局id
        self.coordinator = Coordinator(rank_client_num_map)
        if self._handler is not None:
            self._handler.num_clients = self.coordinator.total


class SynchronousServerManager(ServerManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`main_loop`.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ServerHandler): Backend calculation handler for parameter server.
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(self, network: DistNetwork, handler: ServerHandler, logger: Logger = None):
        super(SynchronousServerManager, self).__init__(network, handler)
        self._LOGGER = Logger() if logger is None else logger

    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server handler.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ServerHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        # 如果不停, 就开启 activate_clients 这个线程,将数据送给各个客户端, 开启下一轮迭代工作
        while self._handler.if_stop is not True:
            activator = threading.Thread(target=self.activate_clients)
            activator.start()

            while True:
                # 每接收一个 MessageCode.ParameterUpdate 就尝试聚合,但是未达到指定轮次,aggregation_algorithm会返回false,继续while
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler.aggregation_algorithm(payload):
                        break
                else:
                    raise Exception("Unexpected message code {}".format(message_code))

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The id of clients are obtained from :meth:`handler.sample_clients`. And their communication ranks are are obtained via coordinator.
        """
        self._LOGGER.info("Client activation procedure,begin next round")
        # 选取参与下一轮循环的客户端编号, 并将其送到各个客户端
        clients_global_id_list_this_round = (
            self._handler.sample_clients()
        )  # clients_global_id_list_this_round是参与聚合的客户端的全局id
        rank_dict = self.coordinator.map_id_list(clients_global_id_list_this_round)

        self._LOGGER.info("activate_clients rank_dict: {}".format(rank_dict))

        # 将数据发送到对应客户端
        # 数据主要是 虚拟客户端id_list(全局id)与downlink_package(即数据模型)
        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package, message_code=MessageCode.ParameterUpdate, dst=rank)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.num_clients)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package, message_code=MessageCode.Exit, dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size - 1)
        assert message_code == MessageCode.Exit


class AsynchronousServerManager(ServerManager):
    """Asynchronous communication network manager for server

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Asynchronously communicate with clients following agreements defined in :meth:`mail_loop`.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ServerHandler): Backend computation handler for parameter server.
        logger (Logger, optional): Object of :class:`Logger`.
    """

    def __init__(self, network: DistNetwork, handler: ServerHandler, logger: Logger = None):
        super(AsynchronousServerManager, self).__init__(network, handler)
        self._LOGGER = Logger() if logger is None else logger

        self.message_queue = Queue()

    def main_loop(self):
        """Communication agreements of asynchronous FL.

        - Server receive ParameterRequest from client. Send model parameter to client.
        - Server receive ParameterUpdate from client. Transmit parameters to queue waiting for aggregation.

        Raises:
            ValueError: invalid message code.
        """
        updater = threading.Thread(target=self.aggregate_thread, daemon=True)
        updater.start()

        while self._handler.if_stop is not True:
            sender, message_code, payload = self._network.recv()

            if message_code == MessageCode.ParameterRequest:
                self._network.send(content=self._handler.downlink_package, message_code=MessageCode.ParameterUpdate, dst=sender)

            elif message_code == MessageCode.ParameterUpdate:
                self.message_queue.put((sender, message_code, payload))

            else:
                raise ValueError("Unexpected message code {}.".format(message_code))

    def shutdown(self):
        self.shutdown_clients()
        super().shutdown()

    def aggregate_thread(self):
        """Asynchronous communication maintain a message queue. A new thread will be started to keep monitoring message queue."""
        while self._handler.if_stop is not True:
            _, message_code, payload = self.message_queue.get()
            self._handler.aggregation_algorithm(payload)

            assert message_code == MessageCode.ParameterUpdate

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to clients with ``MessageCode.Exit``.
        """
        for rank in range(1, self._network.world_size):
            _, message_code, _ = self._network.recv(src=rank)  # client request
            if message_code == MessageCode.ParameterUpdate:
                self._network.recv(src=rank)  # the next package is model request, which is ignored in shutdown stage.
            self._network.send(message_code=MessageCode.Exit, dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size - 1)
        assert message_code == MessageCode.Exit
