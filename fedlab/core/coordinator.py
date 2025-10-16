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


class Coordinator(object):
    """Deal with the mapping relation between client id and process rank in FL system.

    Note
        Server Manager creates a Coordinator following:
        1. init network connection.
        2. client send local group info (the number of client simulating in local) to server.
        4. server receive all info and init a server Coordinator.

    Args:
        setup_dict (dict): A dict like {rank:client_num ...}, representing the map relation between process rank and client id.
    """
    """
    setup_dict是一个字典，格式为{rank: 客户端数量}，表示：
    键rank：分布式训练中进程的唯一标识（如 0,1,2...）；
    值客户端数量：该进程上模拟的本地虚拟客户端总数。
    
     转换过程（map_id方法）
    对于输入的全局客户端 ID（如id=3），map_id通过以下步骤计算本地虚拟序号：
    用m_id跟踪当前 ID 在 “剩余客户端数量” 中的偏移；
    遍历setup_dict中的每个进程（rank）及其客户端数量（num）：
        如果m_id >= num：说明该 ID 不在当前进程，减去当前进程的客户端数量（m_id -= num），继续检查下一个进程；
        如果m_id < num：说明该 ID 在当前进程内，此时m_id就是该进程内的本地虚拟序号。
    """
    def __init__(self, setup_dict: dict) -> None:
        self.rank_client_num_map = setup_dict
    def map_id(self, global_id):
        """a map function from client id to (rank,global_id)
        
        Args:
            id (int): client id

        Returns:
            rank, id : rank in distributed group and local id.
        """
        m_id = global_id
        for rank, num in self.rank_client_num_map.items():
            if m_id >= num:
                m_id -= num
            else:
                local_id = m_id
                ret_id = global_id
                return rank, ret_id

    def map_id_list(self, global_id_list: list):
        """a map function from id_list to dict{rank:local id}

            This can be very useful in Scale modules.

        Args:
            id_list (list(int)): a list of client id.

        Returns:
            map_dict (dict): contains process rank and its relative local client ids.
        """
        map_dict = {}
        for global_id in global_id_list:
            rank, id = self.map_id(global_id)
            if rank in map_dict.keys():
                map_dict[rank].append(id)
            else:
                map_dict[rank] = [id]
        return map_dict



    @property
    def total(self):
        return int(sum(self.rank_client_num_map.values()))

    def __str__(self) -> str:
        return "Coordinator map information: {} \nMap \nTotal: {}".format(
            self.rank_client_num_map,  self.total)

    def __call__(self, info):
        if isinstance(info, int):
            return self.map_id(info)
        if isinstance(info, list):
            return self.map_id_list(info)