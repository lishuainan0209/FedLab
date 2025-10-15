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

import warnings
from copy import deepcopy
from enum import Enum
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from . import DEFAULT_SLICE_SIZE, DEFAULT_MESSAGE_CODE_VALUE
from . import HEADER_DATA_TYPE_IDX, HEADER_SIZE, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX, dtype_flab2torch, dtype_torch2flab
from . import HEADER_SENDER_RANK_IDX, HEADER_MESSAGE_CODE_IDX


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    # Server and Client communication agreements
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3
    Exit = 4
    SetUp = 5
    Activation = 6


supported_torch_dtypes = [
    torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
    torch.float32, torch.float64
]


class Package(object):
    """A basic network package data structure used in FedLab. Everything is Tensor in  FedLab.

    Note:
        ``slice_size_i = tensor_i.shape[0]``, that is, every element in slices indicates the size
        of a sub-Tensor in content.

    :class:`Package` maintains 3 variables:
        - :attr:`header` : ``torch.Tensor([sender_rank, recv_rank, content_size, message_code, data_type])``
        - :attr:`slices` : ``list[slice_size_1, slice_size_2]``
        - :attr:`content` : ``torch.Tensor([tensor_1, tensor_2, ...])``

    Args:
        message_code (MessageCode): Message code
        content (torch.Tensor, optional): Tensors contained in this package.
    """

    def __init__(self,
                 message_code: MessageCode = None,
                 content: List[torch.Tensor] = None):

        if message_code is None:
            message_code = DEFAULT_MESSAGE_CODE_VALUE
        else:
            # change-001    可以自定义包的码,但是码必须是enum类型的
            if isinstance(message_code, Enum):
                message_code = message_code.value
        assert isinstance(
            message_code, int
        ), "message_code can only be MessageCode or integer, not {}".format(
            type(message_code))

        # initialize header. The dtype of header is set as torch.int32 as default.
        self.header = torch.zeros(size=(HEADER_SIZE,), dtype=torch.int32)

        if dist.is_initialized():
            self.header[HEADER_SENDER_RANK_IDX] = dist.get_rank()
        else:
            self.header[HEADER_SENDER_RANK_IDX] = -1
        self.header[HEADER_RECEIVER_RANK_IDX] = -1  # assigned by processor
        self.header[HEADER_MESSAGE_CODE_IDX] = message_code
        self.header[HEADER_SLICE_SIZE_IDX] = DEFAULT_SLICE_SIZE
        self.header[HEADER_DATA_TYPE_IDX] = -1  # assigned by processor

        # initialize content and slices
        self._slices = []
        self.content = None
        self.dtype = None

        if isinstance(content, torch.Tensor):
            self.append_tensor(content)
        if isinstance(content, List):
            self.append_tensor_list(content)

    def append_tensor(self, tensor: torch.Tensor):
        """Append new tensor to :attr:`Package.content`

        Args:
            tensor (torch.Tensor): Tensor to append in content.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "Invalid content type, expecting torch.Tensor but get {}".
                format(type(tensor)))

        shape = list(tensor.shape)
        slice = [tensor.numel(), len(shape)] + shape

        tensor = tensor.view(-1)
        if self.content is None:
            self.content = deepcopy(tensor)
            self.dtype = tensor.dtype
        else:
            if tensor.dtype is not self.dtype:
                warnings.warn(
                    "The dtype of current tensor is {}. But package dtype is {}. The current data type will be casted to {} and fedlab do not guarantee lossless conversion."
                    .format(tensor.dtype, self.dtype, self.dtype))
            tensor = tensor.to(self.dtype)
            self.content = torch.cat((self.content, tensor))

        self._slices += slice
        self.header[HEADER_SLICE_SIZE_IDX] = len(self._slices)

    def append_tensor_list(self, tensor_list: List[torch.Tensor]):
        """Append a list of tensors to :attr:`Package.content`.

        Args:
            tensor_list (list[torch.Tensor]): A list of tensors to append to :attr:`Package.content`.
        """
        for tensor in tensor_list:
            self.append_tensor(tensor)

    def to(self, dtype):
        if dtype in supported_torch_dtypes:
            self.dtype = dtype
            self.content = self.content.to(self.dtype)
        else:
            warnings.warn(
                "FedLab only supports following data types: torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64."
            )

    def _send_header(self, dst):
        self.header[HEADER_RECEIVER_RANK_IDX] = dst
        dist.send(self.header, dst=dst)

    def _send_slices(self, dst):
        np_slices = np.array(self._slices, dtype=np.int32)
        tensor_slices = torch.from_numpy(np_slices)
        dist.send(tensor_slices, dst=dst)

    def _send_content(self, dst):
        dist.send(self.content, dst=dst)

    def send_package(self, dst):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """

        # body
        if self.dtype is not None:
            self.header[HEADER_DATA_TYPE_IDX] = dtype_torch2flab(self.dtype)

        # sender header firstly
        self._send_header(dst=dst)

        # if package got content, then send remain parts
        if self.header[HEADER_SLICE_SIZE_IDX] > 0:
            self._send_slices(dst=dst)
            self._send_content(dst=dst)

    def _recv_header(self, src):
        buffer = torch.zeros(size=(HEADER_SIZE,), dtype=torch.int32)
        dist.recv(buffer, src=src)
        self.header[HEADER_SENDER_RANK_IDX] = int(buffer[HEADER_SENDER_RANK_IDX])
        self.header[HEADER_RECEIVER_RANK_IDX] = int(buffer[HEADER_RECEIVER_RANK_IDX])
        self.header[HEADER_SLICE_SIZE_IDX] = int(buffer[HEADER_SLICE_SIZE_IDX])
        self.header[HEADER_MESSAGE_CODE_IDX] = MessageCode(int(buffer[HEADER_MESSAGE_CODE_IDX])).value
        self.header[HEADER_DATA_TYPE_IDX] = int(buffer[HEADER_DATA_TYPE_IDX])

    def _recv_slices(self, src):
        buffer_slices = torch.zeros(size=(self.header[HEADER_SLICE_SIZE_IDX],), dtype=torch.int32)
        dist.recv(buffer_slices, src=src)
        slices = [slc.item() for slc in buffer_slices]
        return slices

    def _recv_content(self, src):
        slices = self._recv_slices(src=src)
        #         content_size = sum(slices)  # warn 原来fedlab源码中, 错误地将 slices 中的所有元素（包括形状、维度数等无关信息）求和，而没有区分 “元素数” 和 “辅助元信息”。
        content_size = 0
        i = 0
        while i < len(slices):
            numel = slices[i]  # 每个张量元信息的第一个元素是 numel
            content_size += numel
            # 跳过当前张量的其他元信息（len(shape) + 1 个元素：1个len(shape) + len(shape)个shape值）
            len_shape = slices[i + 1]
            i += 2 + len_shape  # 移动到下一个张量的元信息起始位置
        dtype = dtype_flab2torch(self.header[HEADER_DATA_TYPE_IDX])
        buffer = torch.zeros(size=(content_size,), dtype=dtype)
        dist.recv(buffer, src=src)

        index = 0  # parse variable for content, 代表每个tensor在content开始的下标
        iter = 0  # parse variable for slices
        self.content = []
        # slices中, 是 [张量数据量, 张量维度, 张量每个维度的大小...,张量数据量, 张量维度, 张量每个维度的大小...] 这样不断重复的
        # iter代表的是每个张量的 张量数据量 的下标
        while iter < len(slices):
            offset = slices[iter]  # offset of content
            shape_len = slices[iter + 1]  # offset of shape tuple
            shape = tuple(slices[iter + 2:iter + 2 + shape_len])  # obtain shape tuple
            # 参数准备
            # 获取内容
            seg_tensor = buffer[index:index + offset]  # tensor具体内容
            reshape_tensor = seg_tensor.view(size=shape)  # reshape
            self.content.append(reshape_tensor)

            # 为下一个循环进行准备
            index += offset
            iter += shape_len + 2  # 张量数据量,张量维度是两个量,所以加2

    def recv_package(self, src=None):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """
        self.content = None
        self.header = torch.zeros(size=(HEADER_SIZE,), dtype=torch.int32)
        self.dtype = None
        # body
        self._recv_header(src=src)
        if self.header[HEADER_SLICE_SIZE_IDX] > 0:
            self._recv_content(src=src)
        return int(self.header[HEADER_SENDER_RANK_IDX]), MessageCode(int(self.header[HEADER_MESSAGE_CODE_IDX])), self.content
