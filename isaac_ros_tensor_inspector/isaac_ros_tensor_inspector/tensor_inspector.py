# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import struct

from isaac_ros_tensor_msgs.msg import TensorList
import numpy as np
import rclpy
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor


class TensorInspectorNode(Node):
    MSG_TO_PYTHON_MAP = {
        (0, 8, 1):  ('b', 1),
        (1, 8, 1):  ('B', 1),
        (0, 16, 1): ('h', 2),
        (1, 16, 1): ('H', 2),
        (0, 32, 1): ('i', 4),
        (1, 32, 1): ('I', 4),
        (0, 64, 1): ('q', 8),
        (1, 64, 1): ('Q', 8),
        (2, 32, 1): ('f', 4),
        (2, 64, 1): ('d', 8),
    }

    PYTHON_TO_MSG_MAP = {
        np.dtype('int8'):    (0, 8, 1),
        np.dtype('uint8'):   (1, 8, 1),
        np.dtype('int16'):   (0, 16, 1),
        np.dtype('uint16'):  (1, 16, 1),
        np.dtype('int32'):   (0, 32, 1),
        np.dtype('uint32'):  (1, 32, 1),
        np.dtype('int64'):   (0, 64, 1),
        np.dtype('uint64'):  (1, 64, 1),
        np.dtype('float32'): (2, 32, 1),
        np.dtype('float64'): (2, 64, 1),
    }

    def __init__(self):
        super().__init__('tensor_inspector')

        # Input side: whether or not to save the original tensor received
        self.subscription = self.create_subscription(
            TensorList,
            'original_tensor',
            self.listener_callback,
            10)

        self.declare_parameter('original_tensor_npz_path', '')

        self.original_tensor_npz_path = self.get_parameter(
            'original_tensor_npz_path').get_parameter_value().string_value

        self.should_save_original = len(self.original_tensor_npz_path) > 0

        if self.should_save_original:
            self.get_logger().info(
                f'Original tensor received will be saved to {self.original_tensor_npz_path}')

        # Output side: whether or not to replace the original tensor with an edited one
        self.publisher = self.create_publisher(
            TensorList,
            'edited_tensor',
            10)

        self.declare_parameter('edited_tensor_npz_path', '')

        edited_tensor_npz_path = self.get_parameter(
            'edited_tensor_npz_path').get_parameter_value().string_value

        self.should_edit = len(edited_tensor_npz_path) > 0

        if self.should_edit:
            self.edited_tensor_data = np.load(edited_tensor_npz_path)
            self.get_logger().info(
                f'Edited tensor will be loaded from {edited_tensor_npz_path}')

    def listener_callback(self, msg):
        if len(msg.names) != len(msg.tensors):
            self.get_logger().error(
                f'Invalid tensor list: len(names)={len(msg.names)} '
                f'!= len(tensors)={len(msg.tensors)}')
            return

        if self.should_save_original:
            tensor_dict = {}
            for tensor_name, tensor in zip(msg.names, msg.tensors):
                dims = tensor.shape
                N = int(np.prod(dims))
                self.get_logger().debug(
                    f'Tensor {tensor_name} with dims {dims} has N={N} elements')

                dtype = (tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes)
                conversion = self.MSG_TO_PYTHON_MAP.get(dtype)
                if conversion is None:
                    self.get_logger().error(
                        f'Original tensor {tensor_name} has unknown data type {dtype}')
                    continue

                element_format, element_num_bytes = conversion
                required_num_bytes = tensor.byte_offset + N * element_num_bytes
                if required_num_bytes > len(tensor.data):
                    self.get_logger().error(
                        f'Original tensor {tensor_name} has invalid data size: '
                        f'need {required_num_bytes}, got {len(tensor.data)}')
                    continue

                elements = []
                for i in range(N):
                    # Interpret tensor fields as particular numeric type from bytes
                    data_offset = tensor.byte_offset + element_num_bytes * i
                    element = struct.unpack(f'<{element_format}', tensor.data[
                        data_offset:data_offset + element_num_bytes
                    ])[0]  # struct.unpack returns a tuple with one element

                    elements.append(element)

                # Add element as numpy array to dictionary
                tensor_dict[tensor_name] = np.resize(elements, dims)

            np.savez(self.original_tensor_npz_path, **tensor_dict)
            self.get_logger().debug('Saved original tensor')

        if self.should_edit:
            names = []
            tensors = []
            for tensor_name, tensor_npz in self.edited_tensor_data.items():
                # Create new tensor from data
                tensor = ExperimentalTensor()
                tensor.shape = [int(dim) for dim in tensor_npz.shape]

                element_data_type = self.PYTHON_TO_MSG_MAP.get(tensor_npz.dtype)
                if element_data_type is None:
                    self.get_logger().error(
                        f'Edited tensor {tensor_name} has unknown data type {tensor_npz.dtype}')
                    continue

                tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes = element_data_type
                tensor.strides = []
                tensor.byte_offset = 0
                tensor.data = tensor_npz.tobytes()

                names.append(tensor_name)
                tensors.append(tensor)

            # Overwrite original tensors with new tensors
            msg.names = names
            msg.tensors = tensors
            self.get_logger().debug('Edited tensor')

        self.publisher.publish(msg)
        self.get_logger().debug('Published tensor message')


def main(args=None):
    rclpy.init(args=args)
    node = TensorInspectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
