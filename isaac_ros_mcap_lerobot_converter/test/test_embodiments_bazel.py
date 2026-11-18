# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Bazel smoke test for packaged MCAP-to-LeRobot embodiment resources."""

import unittest

from isaac_ros_mcap_lerobot_converter.embodiments import (
    available_embodiments,
    load_embodiment,
)


class EmbodimentsTest(unittest.TestCase):
    """Exercise YAML resource discovery in Bazel runfiles."""

    def test_required_embodiments_load(self):
        required = {'unitree_g1', 'unitree_h2_sharpa_wave'}
        self.assertTrue(required <= set(available_embodiments()))

        h2 = load_embodiment('unitree_h2_sharpa_wave')
        self.assertEqual(58, len(h2.action_joints))


if __name__ == '__main__':
    unittest.main()
