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
"""Unit tests for the config-driven LeRobot feature schema."""

from pathlib import Path

from isaac_ros_mcap_lerobot_converter.embodiments import (
    _parse_modality_section,
    available_embodiments,
    load_embodiment,
)
from isaac_ros_mcap_lerobot_converter.mcap_to_lerobot import build_features
import pytest


def test_both_embodiments_available():
    names = available_embodiments()
    assert 'unitree_g1' in names and 'unitree_h2_sharpa_wave' in names


def test_g1_features_from_config():
    emb = load_embodiment('unitree_g1')
    feats = build_features(emb)  # pure function of the config
    assert feats['observation.state']['shape'] == (len(emb.state_joints),)
    assert feats['action']['shape'] == (len(emb.action_joints),)
    # every declared camera becomes a video feature
    for cam in emb.cameras:
        assert f'observation.images.{cam.name}' in feats


def test_h2_features_from_config():
    emb = load_embodiment('unitree_h2_sharpa_wave')
    feats = build_features(emb)
    assert feats['observation.state']['shape'] == (len(emb.state_joints),)
    assert feats['action']['shape'] == (len(emb.action_joints),)
    for cam in emb.cameras:
        assert f'observation.images.{cam.name}' in feats


def test_g1_effort_columns_split_by_action_modality_group():
    emb = load_embodiment('unitree_g1')
    feats = build_features(emb)
    for group, (start, end) in emb.action_modalities.items():
        key = f'action.effort_{group}'
        assert key in feats
        assert feats[key]['shape'] == (end - start,)


def test_g1_teleop_features_declared_from_config():
    emb = load_embodiment('unitree_g1')
    feats = build_features(emb)
    assert feats['action.navigate_command']['shape'] == (3,)
    assert feats['action.base_height_command']['shape'] == (1,)
    base_height = next(
        modality for modality in emb.actions
        if modality.name == 'base_height_command')
    assert base_height.default == (0.75,)


def test_rejects_vector_default_with_wrong_width():
    raw_modalities = {
        'base_height_command': {
            'type': 'vector',
            'element_names': ['z', 'yaw'],
            'default': [0.75],
        },
    }
    with pytest.raises(
        ValueError,
        match='base_height_command.default must have 2 values',
    ):
        _parse_modality_section(raw_modalities, Path('test.yaml'))
