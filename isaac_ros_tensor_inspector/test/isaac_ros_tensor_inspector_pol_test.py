# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import pathlib
import time

from isaac_ros_tensor_msgs.msg import TensorList
from isaac_ros_test import IsaacROSBaseTest, JSONConversion
from launch_ros.actions import Node

import pytest
import rclpy
from tensor_msgs.msg import ExperimentalTensor


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with all ROS 2 nodes for testing."""
    tensor_inspector_node = Node(
        package='isaac_ros_tensor_inspector',
        executable='tensor_inspector',
        name='tensor_inspector',
        namespace=IsaacROSTensorInspectorPipelineTest.generate_namespace(),
        output='screen'
    )
    return IsaacROSTensorInspectorPipelineTest.generate_test_description([
        tensor_inspector_node
    ])


def load_tensor_list_from_json(json_filepath: pathlib.Path) -> TensorList:
    """
    Load a TensorList message from a JSON filepath.

    Parameters
    ----------
    json_filepath : Path
        The path to a JSON file containing the TensorList fields

    Returns
    -------
    TensorList
        Generated TensorList message

    """
    tensor_list_json = JSONConversion.load_from_json(json_filepath)

    tensor_list = TensorList()
    tensor_list.header.frame_id = tensor_list_json['header']['frame_id']

    tensor_list.names = tensor_list_json['names']
    for tensor_json in tensor_list_json['tensors']:
        tensor = ExperimentalTensor()
        tensor.dtype_code = tensor_json['dtype_code']
        tensor.dtype_bits = tensor_json['dtype_bits']
        tensor.dtype_lanes = tensor_json['dtype_lanes']
        tensor.shape = tensor_json['shape']
        tensor.strides = tensor_json['strides']
        tensor.byte_offset = tensor_json['byte_offset']
        tensor.data = bytes(tensor_json['data'])

        tensor_list.tensors.append(tensor)

    return tensor_list


class IsaacROSTensorInspectorPipelineTest(IsaacROSBaseTest):
    """Test for Isaac ROS Tensor Inspector Pipeline."""

    filepath = pathlib.Path(os.path.dirname(__file__))

    def test_tensor_inspector_pipeline(self) -> None:
        """Expect the pipeline to pass a tensor list through the inspector node."""
        self.generate_namespace_lookup(
            ['original_tensor', 'edited_tensor'])

        tensor_list_pub = self.node.create_publisher(
            TensorList, self.namespaces['original_tensor'], self.DEFAULT_QOS)

        received_messages = {}
        tensor_list_sub, = self.create_logging_subscribers(
            [('edited_tensor', TensorList)], received_messages)

        try:
            tensor_list = load_tensor_list_from_json(
                self.filepath / 'data' / 'tensor_list.json')

            # Wait at most TIMEOUT seconds for subscriber to respond
            TIMEOUT = 10
            end_time = time.time() + TIMEOUT

            done = False
            while time.time() < end_time:
                # Publish test case multiple times in case messages are dropped
                tensor_list_pub.publish(tensor_list)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                # If we have received exactly one message on the output topic, break
                if 'edited_tensor' in received_messages:
                    done = True
                    break

            self.assertTrue(
                done, "Didn't receive output on edited_tensor topic!")

            # Collect received tensors
            tensor_list_actual = received_messages['edited_tensor']

            # Make sure that at least one tensor was found
            self.assertGreaterEqual(len(tensor_list_actual.tensors), 1,
                                    "Didn't find at least 1 tensor in tensor list!")
            self.assertEqual(tensor_list_actual.names, ['test_tensor'])

            for tensor in tensor_list_actual.tensors:
                self.assertEqual(tensor.dtype_code, 2)
                self.assertEqual(tensor.dtype_bits, 32)
                self.assertEqual(tensor.dtype_lanes, 1)
                self.assertEqual(list(tensor.shape), [3])
                self.assertEqual(list(tensor.strides), [])
                self.assertEqual(tensor.byte_offset, 0)

                # Allow for 2 bytes of error in data
                self.assertAlmostEqual(
                    tensor.data[0], 0.0, delta=2, msg='Tensor data is not accurate')

        finally:
            self.assertTrue(self.node.destroy_subscription(tensor_list_sub))
