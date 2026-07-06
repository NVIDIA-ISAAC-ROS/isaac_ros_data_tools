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
"""Data structures for MCAP-to-LeRobot conversion."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Timestamp:
    seconds: int
    nanoseconds: int


@dataclass
class JointState:
    position: list[float]
    velocity: list[float]
    effort: list[float]
    names: list[str]
    timestamp: Timestamp


@dataclass
class Observation:
    data: list[float]
    joint_names: list[str]
    joint_efforts: list[float]
    joint_velocities: list[float]
    joint_positions: list[float]
    timestamp: Timestamp


@dataclass
class TopicStamp:
    topic_name: str
    stamp: Timestamp
    delay_ms: float


@dataclass
class RecordData:
    """Synchronization record with generic per-topic timestamps.

    Each entry in ``topic_stamps`` records which message (by stamp) was
    captured for a given topic at recording time.
    """

    topic_stamps: list[TopicStamp]


@dataclass
class TeleopCommand:
    """Teleop commands for whole-body control."""
    navigate_command: list[float]  # [vx, vy, vyaw]
    base_height_command: list[float]  # [z]
    timestamp: Timestamp


@dataclass
class TrainingTrajectoryItem:
    observation: Observation
    action: Observation
    image_frames: dict[str, np.ndarray]
    teleop_command: TeleopCommand | None = None
