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

from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList
import numpy as np
import rclpy
from rclpy.node import Node


class TensorInspectorNode(Node):
    # Conversion map from Tensor.msg's data types to Python types
    MSG_TO_PYTHON_MAP = {
        # data_type_number : ('struct_specifier', num_bytes)
        1:  ('b', 1),  # int8
        2:  ('B', 1),  # uint8
        3:  ('h', 2),  # int16
        4:  ('H', 2),  # uint16
        5:  ('i', 4),  # int32
        6:  ('I', 4),  # uint32
        7:  ('q', 8),  # int64
        8:  ('Q', 8),  # uint64
        9:  ('f', 4),  # float32
        10: ('d', 8)   # float64
    }

    # Conversion map from Numpy types to Tensor.msg's data types
    PYTHON_TO_MSG_MAP = {
        # dtype : data_type_number
        np.dtype('int8'):    1,
        np.dtype('uint8'):   2,
        np.dtype('int16'):   3,
        np.dtype('uint16'):  4,
        np.dtype('int32'):   5,
        np.dtype('uint32'):  6,
        np.dtype('int64'):   7,
        np.dtype('uint64'):  8,
        np.dtype('float32'): 9,
        np.dtype('float64'): 10
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

        if self.should_save_original:
            tensor_dict = {}
            for tensor in msg.tensors:
                dims = tensor.shape.dims
                N = np.prod(dims)
                self.get_logger().debug(
                    f'Tensor {tensor.name} with dims {dims} has N={N} elements')

                conversion = self.MSG_TO_PYTHON_MAP.get(tensor.data_type)
                if conversion is None:
                    self.get_logger().error(
                        f'Original tensor {tensor.name} has unknown data type {tensor.data_type}')
                    continue

                element_format, element_num_bytes = conversion

                elements = []
                for i in range(N):
                    # Interpret tensor fields as particular numeric type from bytes
                    element = struct.unpack(f'<{element_format}', tensor.data[
                        element_num_bytes * i:
                        element_num_bytes * (i + 1)
                    ])[0]  # struct.unpack returns a tuple with one element

                    elements.append(element)

                # Add element as numpy array to dictionary
                tensor_dict[tensor.name] = np.resize(elements, dims)

            np.savez(self.original_tensor_npz_path, **tensor_dict)
            self.get_logger().debug('Saved original tensor')

        if self.should_edit:
            tensors = []
            for tensor_name, tensor_npz in self.edited_tensor_data.items():
                # Create new tensor from data
                tensor = Tensor()

                tensor.name = tensor_name

                tensor.shape.rank = len(tensor_npz.shape)
                tensor.shape.dims = tensor_npz.shape

                element_data_type = self.PYTHON_TO_MSG_MAP.get(tensor_npz.dtype)
                if element_data_type is None:
                    self.get_logger().error(
                        f'Edited tensor {tensor.name} has unknown data type {tensor_npz.dtype}')
                    continue

                tensor.data_type = element_data_type
                tensor.strides = tensor_npz.strides

                tensor.data = tensor_npz.tobytes()

                tensors.append(tensor)

            # Overwrite original tensors with new tensors
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
