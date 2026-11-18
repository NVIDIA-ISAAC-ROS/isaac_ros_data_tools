#!/usr/bin/env python3
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
"""Convert one or more MCAP rosbags directly into a LeRobot dataset.

No ROS installation is required.

Usage::

    mcap-to-lerobot \\
        --rosbags-dir /path/to/rosbags \\
        --output-dir /path/to/lerobot_dataset \\
        --task displayport_insertion

Each immediate subdirectory of --rosbags-dir that contains an MCAP bag
(identified by a metadata.yaml file) is treated as one episode.
"""

import argparse
import bisect
import concurrent.futures
import contextlib
import dataclasses
import glob
import importlib.metadata
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import av
from isaac_ros_mcap_lerobot_converter.codec import (
    probe_hw_codecs,
    resolve_codec_opts,
    software_codec_opts,
)
from isaac_ros_mcap_lerobot_converter.embodiments import (
    available_embodiments,
    EmbodimentConfig,
    load_embodiment,
    ModalityConfig,
)
from isaac_ros_mcap_lerobot_converter.rosbag_reader import RosbagReader
import isaac_ros_mcap_lerobot_converter.structures as structures

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Canonical GR00T UNITREE_G1 joint orders (43 dims each). MUST stay
# aligned with the policy YAML's semantic.observations.*.element_names
# (state) and semantic.actions.*.element_names (action), and with
# embodiments/unitree_g1.yaml's state/action element ordering (state uses
# thumb/middle/index for the left hand and thumb/index/middle for the
# right — asymmetric per the physical robot's element naming — while
# action uses index/middle/thumb for both hands, what
# JointCommandConverter writes back). The converter reorders purely from
# embodiment.state_joints/action_joints (config-driven, no bag-content
# inspection); these constants are kept only as the independent
# expected-value fixtures the test suite checks the YAML against.
_LEGS_WAIST_ARMS_29: list[str] = [
    # left_leg (6)
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    # right_leg (6)
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    # waist (3)
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    # left_arm (7)
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    # right_arm (7)
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

G1_CANONICAL_STATE_JOINT_ORDER: list[str] = [
    *_LEGS_WAIST_ARMS_29,
    # left_hand (7) — state order: thumb, middle, index
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint',
    'left_hand_thumb_2_joint', 'left_hand_middle_0_joint',
    'left_hand_middle_1_joint', 'left_hand_index_0_joint',
    'left_hand_index_1_joint',
    # right_hand (7) — state order: thumb, index, middle (asymmetric vs left!)
    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint',
    'right_hand_thumb_2_joint', 'right_hand_index_0_joint',
    'right_hand_index_1_joint', 'right_hand_middle_0_joint',
    'right_hand_middle_1_joint',
]

G1_CANONICAL_ACTION_JOINT_ORDER: list[str] = [
    *_LEGS_WAIST_ARMS_29,
    # left_hand (7) — action order: index, middle, thumb
    'left_hand_index_0_joint', 'left_hand_index_1_joint',
    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint',
    'left_hand_thumb_2_joint',
    # right_hand (7) — action order: index, middle, thumb (same as left here)
    'right_hand_index_0_joint', 'right_hand_index_1_joint',
    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint',
    'right_hand_thumb_2_joint',
]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ts_key(ts: structures.Timestamp) -> tuple[int, int]:
    return (ts.seconds, ts.nanoseconds)


def _build_joint_state_index(
    joint_states: list[structures.JointState],
) -> dict[tuple[int, int], structures.JointState]:
    return {_ts_key(js.timestamp): js for js in joint_states}


def _build_image_index(
    image_timestamps: list[structures.Timestamp],
) -> dict[tuple[int, int], int]:
    return {_ts_key(ts): idx for idx, ts in enumerate(image_timestamps)}


def _build_teleop_index(
    commands: list[tuple[structures.Timestamp, list[float]]],
) -> dict[tuple[int, int], list[float]]:
    return {_ts_key(ts): cmd for ts, cmd in commands}


def _joint_state_to_observation(
    joint_state: structures.JointState,
    order: list[str],
) -> structures.Observation:
    """Reorder joint state fields to match *order* and wrap as Observation."""
    indices = []
    for name in order:
        if name not in joint_state.names:
            raise ValueError(
                f"Joint '{name}' not in joint_state.names: {joint_state.names}"
            )
        indices.append(joint_state.names.index(name))

    ordered_pos = [joint_state.position[i] for i in indices]
    ordered_vel = (
        [joint_state.velocity[i] for i in indices] if joint_state.velocity else []
    )
    ordered_eff = (
        [joint_state.effort[i] for i in indices] if joint_state.effort else []
    )

    return structures.Observation(
        data=ordered_pos,
        joint_names=list(order),
        joint_efforts=ordered_eff,
        joint_velocities=ordered_vel,
        joint_positions=ordered_pos,
        timestamp=joint_state.timestamp,
    )


def _timestamp_ns(timestamp: structures.Timestamp) -> int:
    return timestamp.seconds * 10 ** 9 + timestamp.nanoseconds


def _build_zoh_joint_state_index(
    joint_states: list[structures.JointState],
) -> tuple[list[int], list[structures.JointState]]:
    """Build the sorted timestamp index used by causal ZOH lookups."""
    ordered = sorted(joint_states, key=lambda state: _timestamp_ns(state.timestamp))
    return ([_timestamp_ns(state.timestamp) for state in ordered], ordered)


def _zoh_joint_state(
    zoh_index: tuple[list[int], list[structures.JointState]],
    timestamp: structures.Timestamp,
    max_staleness_s: float | None = None,
) -> structures.JointState | None:
    """Return the newest joint state no later than timestamp."""
    timestamps, ordered = zoh_index
    index = bisect.bisect_right(timestamps, _timestamp_ns(timestamp))
    if not index:
        return None
    selected = ordered[index - 1]
    if (
        max_staleness_s is not None
        and _timestamp_ns(timestamp) - _timestamp_ns(selected.timestamp)
        > int(max_staleness_s * 1e9)
    ):
        return None
    return selected


def _resolve_action_sources(
    command_js: structures.JointState,
    embodiment: EmbodimentConfig,
    zoh_indices: dict[
        str, tuple[list[int], list[structures.JointState]]
    ],
    max_staleness_s: float | None,
) -> list[structures.JointState] | None:
    """Resolve one JointState per embodiment.action_sources entry.

    Sources with no sync reuse the caller's time-synced command message
    directly. Sources with sync: causal_zoh (e.g. finger commands on a
    separate, slower topic) are resolved via zero-order-hold lookup against
    the command's timestamp. Returns None when a ZOH source has no sample
    within staleness tolerance, signalling the caller to drop the frame.
    """
    resolved = []
    for source in embodiment.action_sources:
        if source.sync is None:
            resolved.append(command_js)
            continue
        if source.sync != 'causal_zoh':
            raise ValueError(f'Unsupported action source sync {source.sync!r}')
        joint_state = _zoh_joint_state(
            zoh_indices[source.topic],
            command_js.timestamp,
            max_staleness_s,
        )
        if joint_state is None:
            return None
        resolved.append(joint_state)
    return resolved


def _merge_action(
    resolved_sources: list[structures.JointState],
    embodiment: EmbodimentConfig,
) -> structures.Observation:
    """Join resolved action sources into the YAML-declared action order."""
    zero_effort_joints = {
        joint
        for modality in embodiment.actions
        if modality.type == 'joint_effort' and modality.value == 'zero'
        for joint in modality.element_names
    }
    positions = []
    efforts = []
    velocities = []
    # Track whether any joint carried a real (source-reported) effort, as
    # opposed to a config-forced zero or a default padding zero. A bag that
    # reports no efforts at all must yield empty joint_efforts so downstream
    # can tell "position-only recording" from "all torques happened to be 0".
    any_real_effort = False
    for name in embodiment.action_joints:
        for source in resolved_sources:
            if name not in source.names:
                continue
            index = source.names.index(name)
            positions.append(source.position[index])
            velocities.append(
                source.velocity[index] if index < len(source.velocity) else 0.0)
            if name in zero_effort_joints:
                efforts.append(0.0)
            elif index < len(source.effort):
                efforts.append(source.effort[index])
                any_real_effort = True
            else:
                efforts.append(0.0)
            break
        else:
            raise ValueError(f'Action source is missing joint {name!r}')
    timestamp = resolved_sources[0].timestamp if resolved_sources else None
    return structures.Observation(
        data=positions,
        joint_names=list(embodiment.action_joints),
        joint_efforts=efforts if any_real_effort else [],
        joint_velocities=velocities,
        joint_positions=positions,
        timestamp=timestamp,
    )


def _vector_modality_default(modality: ModalityConfig) -> list[float]:
    """Return the configured default, or zeros, for a vector modality."""
    if modality.default is not None:
        return list(modality.default)
    return [0.0] * len(modality.element_names)


# ---------------------------------------------------------------------------
# Time-synchronization: build trajectory from in-memory reader data
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Strip leading '/' so topic names compare consistently."""
    return name.lstrip('/')


def _warn_undeclared_cameras(
    reader: RosbagReader,
    camera_topics: list[str],
    embodiment: EmbodimentConfig,
) -> None:
    """Warn about compressed camera topics present in the bag but not
    declared by the embodiment.

    Camera selection is config-driven: only topics declared in the
    embodiment YAML are exported. A bag camera that no declared camera
    references is dropped from the dataset, so surface it explicitly rather
    than losing the stream silently.
    """
    undeclared = (
        set(reader.compressed_camera_image_topics) - set(camera_topics))
    if undeclared:
        logger.warning(
            '%s bag carries camera topic(s) not declared in the embodiment '
            '— these streams are excluded from the dataset: %s',
            embodiment.name, sorted(undeclared))


def _match_camera_topic(
    record_topic: str,
    bag_camera_topics: list[str],
) -> str | None:
    """Match a RecordData topic name to the actual compressed image topic.

    The data flywheel records the *base* image topic (e.g.
    /camera_1/color/image_raw) while the bag stores the transport
    variant (e.g. /camera_1/color/image_raw_compressed or
    /camera_1/color/image_raw/compressed).
    """
    norm = _normalize(record_topic)
    for bag_topic in bag_camera_topics:
        bag_norm = _normalize(bag_topic)
        if bag_norm == norm:
            return bag_topic
        # Accept only path-boundary ('/') or transport-suffix ('_') matches
        # to avoid false positives like '/camera_10' matching '/camera_1'.
        if bag_norm.startswith(norm) and bag_norm[len(norm)] in ('/', '_'):
            return bag_topic
    return None


def _get_stamps_for_record(
    record: structures.RecordData,
    camera_topics: list[str],
) -> tuple[
    structures.Timestamp | None,
    structures.Timestamp | None,
    dict[str, structures.Timestamp],
    structures.Timestamp | None,
    structures.Timestamp | None,
]:
    joint_stamp = None
    command_stamp = None
    image_stamps: dict[str, structures.Timestamp] = {}
    twist_stamp = None
    pose_stamp = None

    for ts in record.topic_stamps:
        norm = _normalize(ts.topic_name)
        if norm == 'joint_states':
            joint_stamp = ts.stamp
        elif norm == 'applied_joint_commands':
            command_stamp = ts.stamp
        elif norm.endswith('root_twist'):
            twist_stamp = ts.stamp
        elif norm.endswith('root_pose'):
            pose_stamp = ts.stamp
        else:
            matched = _match_camera_topic(ts.topic_name, camera_topics)
            if matched is not None:
                image_stamps[matched] = ts.stamp

    return joint_stamp, command_stamp, image_stamps, twist_stamp, pose_stamp


def build_synchronized_trajectory(
    reader: RosbagReader,
    decoded_frames: dict[str, list[np.ndarray]],
    embodiment: EmbodimentConfig,
) -> list[structures.TrainingTrajectoryItem]:
    """Build a time-synchronized trajectory directly from reader data.

    Joint ordering, action-source resolution and teleop defaults are all
    driven by embodiment — no bag-content inspection or heuristics.
    """
    camera_topics = [camera.topic for camera in embodiment.cameras]
    vector_modalities = [m for m in embodiment.actions if m.type == 'vector']
    nav_modality = next(
        (m for m in vector_modalities if m.name == 'navigate_command'), None)
    height_modality = next(
        (m for m in vector_modalities if m.name == 'base_height_command'), None)
    trajectory: list[structures.TrainingTrajectoryItem] = []

    # Build O(1) lookup indices.
    js_index = _build_joint_state_index(reader.robot_joint_states)
    cmd_index = _build_joint_state_index(reader.joint_commands)
    img_indices = {
        topic: _build_image_index(
            reader.compressed_camera_image_timestamps.get(topic, []))
        for topic in camera_topics
    }
    nav_index = _build_teleop_index(reader.navigate_commands)
    height_index = _build_teleop_index(reader.base_height_commands)
    zoh_topics = {
        source.topic for source in embodiment.action_sources
        if source.sync is not None
    }
    zoh_indices = {
        topic: _build_zoh_joint_state_index(
            reader.joint_messages_for_topic(topic))
        for topic in zoh_topics
    }

    for record in reader.record_data:
        js_stamp, cmd_stamp, img_stamps, twist_stamp, pose_stamp = (
            _get_stamps_for_record(record, camera_topics)
        )

        if js_stamp is None or cmd_stamp is None:
            continue

        joint_state = js_index.get(_ts_key(js_stamp))
        if joint_state is None:
            continue

        command_js = cmd_index.get(_ts_key(cmd_stamp))
        if command_js is None:
            continue

        frame_dict: dict[str, np.ndarray] = {}
        skip = False
        for cam_topic in camera_topics:
            cam_stamp = img_stamps.get(cam_topic)
            if cam_stamp is None:
                skip = True
                break
            img_idx = img_indices[cam_topic].get(_ts_key(cam_stamp), -1)
            cam_frames = decoded_frames.get(cam_topic, [])
            if img_idx < 0 or img_idx >= len(cam_frames):
                skip = True
                break
            frame_dict[cam_topic] = cam_frames[img_idx]

        if skip:
            continue

        resolved_sources = _resolve_action_sources(
            command_js,
            embodiment,
            zoh_indices,
            getattr(reader, 'max_staleness_s', None),
        )
        if resolved_sources is None:
            continue

        observation = _joint_state_to_observation(
            joint_state, order=embodiment.state_joints)
        action = _merge_action(resolved_sources, embodiment)

        teleop_cmd = None
        if vector_modalities:
            nav = nav_index.get(_ts_key(twist_stamp)) if twist_stamp is not None else None
            height = (
                height_index.get(_ts_key(pose_stamp)) if pose_stamp is not None else None)
            if nav is not None or height is not None:
                nav_default = (
                    _vector_modality_default(nav_modality) if nav_modality else [0.0] * 3)
                height_default = (
                    _vector_modality_default(height_modality) if height_modality else [0.0])
                teleop_cmd = structures.TeleopCommand(
                    navigate_command=nav if nav is not None else nav_default,
                    base_height_command=height if height is not None else height_default,
                    timestamp=joint_state.timestamp,
                )

        trajectory.append(structures.TrainingTrajectoryItem(
            observation=observation,
            action=action,
            image_frames=frame_dict,
            teleop_command=teleop_cmd,
        ))

    return trajectory


# ---------------------------------------------------------------------------
# LeRobot dataset creation
# ---------------------------------------------------------------------------

def build_features(embodiment: EmbodimentConfig) -> dict:
    """Build the LeRobot features dict purely from an embodiment config.

    Video feature shapes are declared with a placeholder (0, 0, 3) — the
    real height/width are only known once camera frames are decoded, so the
    caller patches those in before passing the dict to LeRobotDataset.create.
    """
    features: dict = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (len(embodiment.state_joints),),
            'names': ['observation_to_policy'],
        },
        'action': {
            'dtype': 'float32',
            'shape': (len(embodiment.action_joints),),
            'names': ['action_from_policy'],
        },
    }

    # Per-body-part effort columns, one per action_modalities group. Efforts
    # are always recorded in action_joints order, so the same group
    # boundaries used to slice action slice action.effort too. Lets the
    # trainer reference effort body parts as standalone modality keys
    # (effort_left_arm, etc.) without sharing a single column across
    # multiple modality entries.
    for group, (start, end) in embodiment.action_modalities.items():
        features[f'action.effort_{group}'] = {
            'dtype': 'float32',
            'shape': (end - start,),
            'names': ['action_from_policy'],
        }

    for modality in embodiment.actions:
        if modality.type != 'vector':
            continue
        features[f'action.{modality.name}'] = {
            'dtype': 'float32',
            'shape': (len(modality.element_names),),
            'names': ['action_from_policy'],
        }

    for camera in embodiment.cameras:
        features[f'observation.images.{camera.name}'] = {
            'dtype': 'video',
            'shape': (0, 0, 3),
            'names': ['height', 'width', 'channels'],
        }

    return features


_LEROBOT_SHAPE1_BUG_VERSIONS = frozenset({'0.3.2'})


@contextlib.contextmanager
def _lerobot_shape1_compat():
    """Scoped workaround for the LeRobot 0.3.2 shape-(1,) feature bug.

    LeRobot 0.3.2 maps shape == (1,) to a scalar datasets.Value in
    get_hf_features_from_features but add_frame validates the same
    feature as a 1-element np.ndarray. save_episode then fails
    because Value.encode_example calls float(ndarray).

    We replace the mapping with Sequence(length=1) only for user
    features (those carrying a names list). Default LeRobot scalar
    features like frame_index or timestamp store Python scalars
    and need to stay as Value.

    The override is installed inside this context manager and restored on
    exit so other code in the same process sees unpatched LeRobot. On any
    LeRobot version outside _LEROBOT_SHAPE1_BUG_VERSIONS this is a
    no-op — the workaround would likely misfire if upstream semantics
    change, so we refuse to apply it blindly.
    """
    try:
        lerobot_version = importlib.metadata.version('lerobot')
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = ''

    if lerobot_version not in _LEROBOT_SHAPE1_BUG_VERSIONS:
        yield
        return

    import datasets as _datasets
    from lerobot.datasets import utils as _lerobot_utils
    from lerobot.datasets import lerobot_dataset as _lerobot_dataset

    original = _lerobot_utils.get_hf_features_from_features

    def patched(features: dict) -> _datasets.Features:
        overrides: dict = {}
        for key, ft in features.items():
            if (
                tuple(ft['shape']) == (1,)
                and ft['dtype'] not in ('image', 'video')
                and ft.get('names')
            ):
                overrides[key] = _datasets.Sequence(
                    length=1, feature=_datasets.Value(dtype=ft['dtype']))
        if not overrides:
            return original(features)
        hf = dict(original(features))
        hf.update(overrides)
        return _datasets.Features(hf)

    # Patch BOTH modules: lerobot_dataset.py does
    # from .utils import get_hf_features_from_features so the symbol is
    # rebound at import time and patching only utils would miss it.
    _lerobot_utils.get_hf_features_from_features = patched
    _lerobot_dataset.get_hf_features_from_features = patched
    try:
        yield
    finally:
        _lerobot_utils.get_hf_features_from_features = original
        _lerobot_dataset.get_hf_features_from_features = original


def _encode_video_frames_hw(
    imgs_dir,
    video_path,
    fps: int,
    pix_fmt: str = 'yuv420p',
    g: int | None = 2,
    crf: int | None = 30,
    log_level: int | None = None,
    overwrite: bool = False,
) -> None:
    """Encode imgs_dir's PNG frames into video_path via NVIDIA nvenc.

    Mirrors lerobot.datasets.video_utils.encode_video_frames's I/O contract
    but selects the h264_nvenc hardware encoder, which LeRobot's own wrapper
    refuses to accept (its vcodec whitelist is h264/hevc/libsvtav1 only).
    Called via _nvenc_encode_compat's monkeypatch, never directly.

    crf (an x264/libsvtav1 concept) is translated to nvenc's constant-
    quality knobs (rc=vbr, cq). A small g (GOP size) is paired with bf=0:
    nvenc's default non-zero B-frame count fails to open the codec with a
    GOP as small as lerobot's default g=2.
    """
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    template = 'frame_' + ('[0-9]' * 6) + '.png'
    input_list = sorted(
        glob.glob(str(imgs_dir / template)),
        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not input_list:
        raise FileNotFoundError(f'No images found in {imgs_dir}.')
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    video_options: dict = {}
    if g is not None:
        video_options['g'] = str(g)
        video_options['bf'] = '0'
    if crf is not None:
        video_options['rc'] = 'vbr'
        video_options['cq'] = str(crf)

    if log_level is not None:
        logging.getLogger('libav').setLevel(log_level)

    with av.open(str(video_path), 'w') as output:
        output_stream = output.add_stream('h264_nvenc', fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        for input_data in input_list:
            input_image = Image.open(input_data).convert('RGB')
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        packet = output_stream.encode()  # flush
        if packet:
            output.mux(packet)

    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f'Video encoding did not work. File not found: {video_path}.')


@contextlib.contextmanager
def _nvenc_encode_compat(codec_opts):
    """Scoped monkeypatch routing LeRobot's video encoder through nvenc.

    A no-op unless codec_opts.use_nvenc is set. Otherwise patches
    lerobot.datasets.lerobot_dataset.encode_video_frames (imported by name
    into that module, so patching only video_utils would miss it — same
    caveat _lerobot_shape1_compat documents) to try _encode_video_frames_hw
    first. There is no public codec hook to select an encoder, and
    LeRobot's own wrapper hard-rejects h264_nvenc, so this follows the same
    monkeypatch-a-scoped-symbol pattern as _lerobot_shape1_compat.

    On a hw-encode failure: falls back to LeRobot's default software
    encoder and logs once when codec_opts.nvenc_required is False (--nvenc
    auto); re-raises when it's True (--nvenc true, a hard requirement). The
    override is restored on exit so other code in the same process sees
    unpatched LeRobot.
    """
    if codec_opts is None or not codec_opts.use_nvenc:
        yield
        return

    from lerobot.datasets import lerobot_dataset as _lerobot_dataset
    from lerobot.datasets.video_utils import (
        encode_video_frames as _original_encode_video_frames,
    )

    warned = False

    def patched(imgs_dir, video_path, fps, **kwargs):
        nonlocal warned
        try:
            _encode_video_frames_hw(imgs_dir, video_path, fps, **kwargs)
            return
        except Exception:
            if codec_opts.nvenc_required:
                raise
            if not warned:
                logger.warning(
                    'nvenc encode failed; falling back to software encode',
                    exc_info=True)
                warned = True
            _original_encode_video_frames(imgs_dir, video_path, fps, **kwargs)

    _lerobot_dataset.encode_video_frames = patched
    try:
        yield
    finally:
        _lerobot_dataset.encode_video_frames = _original_encode_video_frames


_FPS_MATCH_TOLERANCE = 0.05  # ±5% — well under the worst-case recorder jitter.


def _detect_rate_from_record_data(reader: RosbagReader) -> float | None:
    """Return the median rate of /record_data (Hz), or None if there
    are fewer than two records to estimate from.
    """
    rd = reader.record_data
    if len(rd) < 2:
        return None
    # Each record's first topic_stamp.stamp is a fine reference; using the
    # first stamp keeps the function independent of which topics happen to be
    # present in the record (some bags only reference a subset).
    times_ns = []
    for rec in rd:
        if not rec.topic_stamps:
            continue
        ts = rec.topic_stamps[0].stamp
        times_ns.append(ts.seconds * 10 ** 9 + ts.nanoseconds)
    if len(times_ns) < 2:
        return None
    times_ns.sort()
    deltas = [times_ns[i + 1] - times_ns[i] for i in range(len(times_ns) - 1)]
    deltas.sort()
    median_ns = deltas[len(deltas) // 2]
    if median_ns <= 0:
        return None
    return 1e9 / median_ns


# ---------------------------------------------------------------------------
# Standalone per-bag conversion (the MAP unit for the process-pool driver)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BagResult:
    """Outcome of converting one bag into its own single-episode shard."""

    shard_dir: str
    num_frames: int
    ok: bool
    error: str | None = None
    any_real_efforts: bool = False


def convert_one_bag(
    bag_dir: str,
    embodiment: EmbodimentConfig,
    task: str,
    fps: float,
    shard_dir: str,
    codec_opts,
    resample_to_rate: float | None = None,
) -> BagResult:
    """Convert a single MCAP bag into a standalone single-episode LeRobot shard.

    This is the MAP unit for the parallel converter: it opens one bag,
    decodes its frames, builds the synchronized trajectory and writes a
    fresh LeRobotDataset (one episode) rooted at shard_dir. No shared or
    module-global state, so distinct calls with distinct shard_dir values
    are safe to run concurrently across a process pool; merge_shards
    later reduces the shards into the final dataset.

    fps is the already-resolved target rate for the whole conversion, used
    to label the shard. resample_to_rate, when not None, resyncs the bag to
    that rate via causal ZOH before building frames; when None the bag's
    own /record_data is used as-is. The resample decision is made once for
    the whole batch by _resolve_resample_to_rate, not per-worker, so every
    shard agrees on whether resampling happened.

    Never raises — any failure is caught and reported via
    BagResult(ok=False, error=...) so a batch driver can continue with the
    remaining bags.
    """
    try:
        max_staleness_s = 2.0 / fps
        reader = RosbagReader(bag_dir, max_staleness_s=max_staleness_s)
        reader.construct_dataset()
        if resample_to_rate is not None:
            reader.max_staleness_s = 2.0 / resample_to_rate
            reader.resync_at_rate(resample_to_rate)

        camera_topics = [camera.topic for camera in embodiment.cameras]
        missing_camera_topics = (
            set(camera_topics) - set(reader.compressed_camera_image_topics))
        if missing_camera_topics:
            raise ValueError(
                f'{embodiment.name} conversion is missing camera topic(s): '
                f'{sorted(missing_camera_topics)}')
        _warn_undeclared_cameras(reader, camera_topics, embodiment)
        for source in embodiment.action_sources:
            if source.sync is not None and not reader.joint_messages_for_topic(
                    source.topic):
                raise ValueError(
                    f'{embodiment.name} conversion requires {source.topic} '
                    f'messages')

        frames = {
            t: reader.decode_video_frames(t, codec_opts=codec_opts)
            for t in camera_topics
        }
        trajectory = build_synchronized_trajectory(reader, frames, embodiment)
        if not trajectory:
            raise ValueError(f'{bag_dir} yielded zero synchronized frames')

        sample_frames = trajectory[0].image_frames
        if not sample_frames:
            raise ValueError(f'{bag_dir} has no camera data')
        sample_cam = next(iter(sample_frames.values()))
        img_h, img_w = sample_cam.shape[:2]

        camera_key_map = {
            camera.topic: camera.name for camera in embodiment.cameras}
        features = build_features(embodiment)
        for camera in embodiment.cameras:
            features[f'observation.images.{camera.name}']['shape'] = (
                img_h, img_w, 3)

        with _lerobot_shape1_compat(), _nvenc_encode_compat(codec_opts):
            dataset = LeRobotDataset.create(
                root=shard_dir,
                repo_id='isaac_ros_mcap_lerobot',
                fps=int(round(fps)),
                robot_type=embodiment.name,
                features=features,
            )
            any_real_efforts = _add_trajectory_to_dataset(
                dataset, trajectory, camera_topics, camera_key_map, task,
                embodiment)
            if hasattr(dataset, 'consolidate'):
                dataset.consolidate()

        return BagResult(
            shard_dir=shard_dir, num_frames=len(trajectory), ok=True,
            any_real_efforts=any_real_efforts)
    except Exception as exc:
        logger.exception('Failed to convert bag %s', bag_dir)
        return BagResult(
            shard_dir=shard_dir, num_frames=0, ok=False, error=str(exc))


def create_lerobot_dataset(
    rosbags_dirs: list[str],
    output_dir: str,
    task: str,
    fps: int | None = None,
    robot_type: str = 'unitree_g1',
    embodiment_name: str = 'unitree_g1',
    max_frames_per_episode: int | None = None,
    strict_fps: bool = False,
):
    """Create a LeRobot dataset from one or more directories of MCAP rosbags.

    Each sub-directory containing a metadata.yaml is treated as a
    separate episode. Pass multiple rosbags_dirs to merge several
    recording sessions into one dataset; episodes are numbered sequentially
    across the supplied directories in the order given.

    When sessions were recorded at different rates, pass fps explicitly
    so the converter resamples each bag to the same target rate. Otherwise
    the rate detected from the first bag is applied to all and any
    mis-matched sessions get rejected by the ±5% tolerance check.

    Args:
        fps: Target dataset rate (Hz).  Resolution rules:

            * None (default) — use the bag's existing /record_data as
              the row source; the dataset's labeled fps is the rate detected
              from those records.
            * Set, matches detected rate within ±5% — same as None, just
              with the supplied label.
            * Set, differs from detected rate — discard the bag's
              /record_data and regenerate it via causal ZOH at the
              requested rate.  The dataset is then both physically sampled
              and labeled at fps.

        strict_fps: When True, error out instead of resampling when the
            requested fps does not match the bag's detected rate. Useful
            for CI pipelines that want to catch mismatches between
            recording- and conversion-time rate assumptions.
        robot_type: Free-form label written to info.json. Independent of
            embodiment_name — it does not select the schema.
        embodiment_name: Name of the packaged embodiment YAML (see
            embodiments/) that defines the joint layout, action sources,
            teleop declarations and cameras used to build the dataset
            schema.
        max_frames_per_episode: If set, only the first N frames of each
            episode are written.  Useful for fast smoke-tests.
    """
    embodiment = load_embodiment(embodiment_name)
    bag_dirs: list[str] = []
    for d in rosbags_dirs:
        bag_dirs.extend(_discover_bag_dirs(d))
    if not bag_dirs:
        logger.error('No MCAP bags found under %s', rosbags_dirs)
        return
    if len(rosbags_dirs) > 1:
        logger.info('Merging %d sessions into one dataset: %s',
                    len(rosbags_dirs), rosbags_dirs)

    # max_staleness_s gates how stale a ZOH'd sample is allowed to be when
    # building synthetic record_data. 2/fps keeps one missed publish
    # tolerable without leaking stale actions.  We need a value before
    # opening the first bag, so use the requested fps if provided, else a
    # safe default — the real rate is resolved after the first bag is read.
    max_staleness_s = 2.0 / (fps if fps is not None else 15)

    first_reader = RosbagReader(bag_dirs[0], max_staleness_s=max_staleness_s)
    first_reader.construct_dataset()

    # Resolve the dataset's effective rate.  Three outcomes:
    #   resolved_fps      — what we'll write to info.json.fps
    #   resample_to_rate  — if not None, regenerate /record_data at this
    #                        rate before building the trajectory
    detected_fps = _detect_rate_from_record_data(first_reader)
    if fps is None:
        if detected_fps is None:
            raise ValueError(
                'Could not detect rate from bag /record_data and no --fps '
                'was provided. Either pass --fps explicitly, or convert a '
                'bag with /record_data populated by the recorder.')
        resolved_fps = int(round(detected_fps))
        resample_to_rate = None
        logger.info(
            'fps unspecified — using detected rate %.2f Hz (rounded to %d)',
            detected_fps, resolved_fps)
    else:
        resolved_fps = fps
        if detected_fps is None:
            # Fewer than two /record_data entries — too few to detect a rate
            # and resample meaningfully.  Honour the requested label and let
            # the synthetic-sync fallback (camera-locked) produce the rows.
            resample_to_rate = None
        elif abs(detected_fps - fps) / max(detected_fps, 1.0) <= _FPS_MATCH_TOLERANCE:
            resample_to_rate = None
            logger.info(
                'fps %d ≈ detected rate %.2f Hz; using bag /record_data as-is',
                fps, detected_fps)
        else:
            if strict_fps:
                raise ValueError(
                    f'--fps {fps} does not match bag rate '
                    f'{detected_fps:.2f} Hz (±{_FPS_MATCH_TOLERANCE:.0%}) and '
                    f'--strict-fps is set. Either pass --fps {int(round(detected_fps))} '
                    f'or drop --strict-fps to resample.')
            resample_to_rate = float(fps)
            logger.warning(
                'fps %d differs from detected rate %.2f Hz — '
                'resampling all bags via causal ZOH at %d Hz',
                fps, detected_fps, fps)

    # If we're resampling, recompute max_staleness_s from the resolved rate
    # and regenerate the first bag's /record_data at the requested rate.
    if resample_to_rate is not None:
        first_reader.max_staleness_s = 2.0 / resample_to_rate
        first_reader.resync_at_rate(resample_to_rate)

    camera_topics = [camera.topic for camera in embodiment.cameras]
    missing_camera_topics = (
        set(camera_topics) - set(first_reader.compressed_camera_image_topics))
    if missing_camera_topics:
        raise ValueError(
            f'{embodiment.name} conversion is missing camera topic(s): '
            f'{sorted(missing_camera_topics)}')
    _warn_undeclared_cameras(first_reader, camera_topics, embodiment)
    for source in embodiment.action_sources:
        if source.sync is not None and not first_reader.joint_messages_for_topic(
                source.topic):
            raise ValueError(
                f'{embodiment.name} conversion requires {source.topic} messages')

    first_frames = {
        t: first_reader.decode_video_frames(t) for t in camera_topics
    }
    first_traj = build_synchronized_trajectory(
        first_reader, first_frames, embodiment)

    if max_frames_per_episode is not None:
        first_traj = first_traj[:max_frames_per_episode]

    if not first_traj:
        logger.error('First bag yielded zero synchronized frames')
        return

    sample_frames = first_traj[0].image_frames
    if not sample_frames:
        logger.error('First bag has no camera data')
        return
    sample_cam = next(iter(sample_frames.values()))
    img_h, img_w = sample_cam.shape[:2]

    camera_key_map = {camera.topic: camera.name for camera in embodiment.cameras}

    features = build_features(embodiment)
    for camera in embodiment.cameras:
        features[f'observation.images.{camera.name}']['shape'] = (img_h, img_w, 3)

    with _lerobot_shape1_compat():
        dataset = LeRobotDataset.create(
            root=output_dir,
            repo_id='isaac_ros_mcap_lerobot',
            fps=resolved_fps,
            robot_type=robot_type,
            features=features,
        )

        any_real_efforts = _add_trajectory_to_dataset(
            dataset, first_traj, camera_topics, camera_key_map, task, embodiment)
        logger.info('Added episode 0 (%d frames) from %s',
                    len(first_traj), bag_dirs[0])

        per_bag_staleness = (
            2.0 / resample_to_rate if resample_to_rate is not None
            else max_staleness_s
        )
        for ep_idx, bag_dir in enumerate(bag_dirs[1:], start=1):
            try:
                reader = RosbagReader(bag_dir, max_staleness_s=per_bag_staleness)
                reader.construct_dataset()
                if resample_to_rate is not None:
                    reader.resync_at_rate(resample_to_rate)
                frames = {t: reader.decode_video_frames(t)
                          for t in camera_topics}
                traj = build_synchronized_trajectory(reader, frames, embodiment)
                if max_frames_per_episode is not None:
                    traj = traj[:max_frames_per_episode]
                if traj:
                    ep_has_efforts = _add_trajectory_to_dataset(
                        dataset, traj, camera_topics, camera_key_map, task,
                        embodiment)
                    any_real_efforts = any_real_efforts or ep_has_efforts
                    logger.info(
                        'Added episode %d (%d frames) from %s',
                        ep_idx, len(traj), bag_dir,
                    )
                else:
                    logger.warning('Episode %d from %s yielded 0 frames',
                                   ep_idx, bag_dir)
            except Exception:
                logger.exception('Failed to process %s, skipping', bag_dir)

        if hasattr(dataset, 'consolidate'):
            dataset.consolidate()
    _write_modality_json(output_dir, camera_key_map, embodiment)
    _write_modality_config_defaults(output_dir, embodiment, any_real_efforts)
    logger.info('LeRobot dataset saved to %s', output_dir)


def _add_trajectory_to_dataset(
    dataset: LeRobotDataset,
    trajectory: list[structures.TrainingTrajectoryItem],
    camera_topics: list[str],
    camera_key_map: dict[str, str],
    task: str,
    embodiment: EmbodimentConfig,
) -> bool:
    """Append a trajectory to the dataset; returns whether real efforts were
    observed in any frame of the trajectory.

    action.effort is split into per-body-part columns (action.effort_<group>)
    using embodiment.action_modalities — the same group boundaries used to
    slice action itself. When joint_efforts is empty on a frame the effort
    columns are zero-padded, but those frames don't count as "real" effort
    observations — the returned flag tracks whether any frame had non-empty
    joint_efforts so the autogen modality config can decide whether to
    register effort modality keys.
    """
    vector_modalities = [m for m in embodiment.actions if m.type == 'vector']
    vector_defaults = {
        m.name: _vector_modality_default(m) for m in vector_modalities
    }
    any_real_efforts = False
    for item in trajectory:
        has_efforts = bool(item.action.joint_efforts)
        any_real_efforts = any_real_efforts or has_efforts
        num_action = len(item.action.data)
        action_effort = (
            item.action.joint_efforts
            if has_efforts
            else [0.0] * num_action
        )
        frame: dict = {
            'observation.state': np.array(item.observation.data, dtype=np.float32),
            'action': np.array(item.action.data, dtype=np.float32),
        }
        effort_arr = np.array(action_effort, dtype=np.float32)
        for group, (start, end) in embodiment.action_modalities.items():
            frame[f'action.effort_{group}'] = effort_arr[start:end]
        for modality in vector_modalities:
            value = (
                getattr(item.teleop_command, modality.name, None)
                if item.teleop_command is not None else None
            )
            frame[f'action.{modality.name}'] = np.array(
                value if value is not None else vector_defaults[modality.name],
                dtype=np.float32,
            )
        for cam_topic in camera_topics:
            key = f'observation.images.{camera_key_map[cam_topic]}'
            frame[key] = item.image_frames[cam_topic]
        dataset.add_frame(frame, task=task)
    dataset.save_episode()
    return any_real_efforts


def _discover_bag_dirs(rosbags_dir: str) -> list[str]:
    """Return sorted list of sub-directories that contain a metadata.yaml.

    Supports two layouts:
    - Flat: rosbags_dir/<episode>/metadata.yaml
    - Nested: rosbags_dir/<episode>/bag/metadata.yaml
    """
    dirs = []
    for entry in sorted(os.listdir(rosbags_dir)):
        full = os.path.join(rosbags_dir, entry)
        if not os.path.isdir(full):
            continue
        if os.path.exists(os.path.join(full, 'metadata.yaml')):
            dirs.append(full)
        elif os.path.exists(os.path.join(full, 'bag', 'metadata.yaml')):
            dirs.append(os.path.join(full, 'bag'))
    if not dirs and os.path.exists(os.path.join(rosbags_dir, 'metadata.yaml')):
        dirs.append(rosbags_dir)
    return dirs


def _write_modality_json(
    output_dir: str,
    camera_key_map: dict[str, str],
    embodiment: EmbodimentConfig,
):
    """Write modality.json's body-part mapping purely from embodiment.

    state/action slices come directly from
    embodiment.state_modalities/action_modalities. Effort and vector
    (teleop) modalities are declared as standalone modality keys pointing
    at the physical columns _add_trajectory_to_dataset writes
    (action.effort_<group>, action.<vector_name>).
    """
    state_modality = {
        group: {'start': start, 'end': end}
        for group, (start, end) in embodiment.state_modalities.items()
    }

    action_modality: dict = {
        group: {'start': start, 'end': end}
        for group, (start, end) in embodiment.action_modalities.items()
    }
    for group, (start, end) in embodiment.action_modalities.items():
        action_modality[f'effort_{group}'] = {
            'original_key': f'action.effort_{group}',
            'start': 0,
            'end': end - start,
        }
    for modality in embodiment.actions:
        if modality.type != 'vector':
            continue
        action_modality[modality.name] = {
            'original_key': f'action.{modality.name}',
            'start': 0,
            'end': len(modality.element_names),
        }

    modality: dict = {
        'state': state_modality,
        'action': action_modality,
        'video': {},
        'annotation': {
            'human.task_description': {'original_key': 'task_index'},
        },
    }
    for cam_key in camera_key_map.values():
        modality['video'][cam_key] = {
            'original_key': f'observation.images.{cam_key}',
        }

    meta_dir = os.path.join(output_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, 'modality.json'), 'w', encoding='utf-8') as f:
        json.dump(modality, f, indent=4)


# ---------------------------------------------------------------------------
# Auto-generated modality config (Python file consumed by launch_finetune)
# ---------------------------------------------------------------------------

_MODALITY_CONFIG_HEADER = '''\
# Auto-generated by mcap-to-lerobot. Re-running the converter regenerates
# this file; copy it to new_embodiment_config.py before editing.
'''


def _abs_action_config_literal() -> str:
    return ('ActionConfig(rep=ActionRepresentation.ABSOLUTE, '
            'type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)')


def _rel_action_config_literal() -> str:
    return ('ActionConfig(rep=ActionRepresentation.RELATIVE, '
            'type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)')


def _write_modality_config_defaults(
    output_dir: str,
    embodiment: EmbodimentConfig,
    has_real_efforts: bool,
):
    """Write new_embodiment_config_defaults.py to the dataset root.

    Always regenerated — overwrites any prior version. Users who want
    custom training choices copy this to new_embodiment_config.py and
    edit the copy; the converter never touches that copy.
    """
    num_state = len(embodiment.state_joints)
    num_action = len(embodiment.action_joints)
    if num_state != 43 or num_action != 43:
        # This GR00T-specific autogen assumes the G1 43-DOF layout — skip
        # for other embodiments; users author their own config against
        # whatever layout they have.
        logger.info(
            'Skipping new_embodiment_config_defaults.py: dataset is not G1 '
            '(num_state=%d, num_action=%d)', num_state, num_action)
        return

    state_keys = list(embodiment.state_modalities.keys())
    has_teleop = any(m.type == 'vector' for m in embodiment.actions)

    action_position_keys = ['left_arm', 'right_arm',
                            'left_hand', 'right_hand', 'waist']
    action_locomotion_keys = (
        ['navigate_command', 'base_height_command'] if has_teleop else [])
    action_effort_keys = (
        [f'effort_{g}' for g in action_position_keys] if has_real_efforts
        else [])
    action_keys = (action_position_keys + action_locomotion_keys
                   + action_effort_keys)

    # Per-key representation: arms get RELATIVE, everything else ABSOLUTE.
    # Order matches action_keys above.
    def _rep_for(key: str) -> str:
        if key in ('left_arm', 'right_arm'):
            return _rel_action_config_literal()
        return _abs_action_config_literal()

    state_lines = ',\n            '.join(f'"{k}"' for k in state_keys)
    action_lines = ',\n            '.join(f'"{k}"' for k in action_keys)
    # , goes inside the f-string so the trailing comment doesn't
    # swallow the list separator; the join uses just a newline + indent.
    action_config_lines = '\n            '.join(
        f'{_rep_for(k)},  # {k}' for k in action_keys
    )

    body = f'''\
from gr00t.configs.data.embodiment_configs import (
    MODALITY_CONFIGS,
    register_modality_config,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


config = {{
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            {state_lines},
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            {action_lines},
        ],
        action_configs=[
            {action_config_lines}
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}}

# register_modality_config asserts the tag isn't already in
# MODALITY_CONFIGS, so a second import of this file inside the same
# Python process (notebook, stats regen after train, pytest re-runs)
# would crash. Re-binding is safe — the new config wins.
MODALITY_CONFIGS.pop(EmbodimentTag.NEW_EMBODIMENT.value, None)
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''

    path = os.path.join(output_dir, 'new_embodiment_config_defaults.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(_MODALITY_CONFIG_HEADER)
        f.write('\n')
        f.write(body)
    logger.info('Wrote modality config defaults to %s', path)
    if has_real_efforts:
        logger.info('  (includes effort_<group> keys — bag had non-zero torques)')
    else:
        logger.info('  (position-only — bag had no observed torques)')


# ---------------------------------------------------------------------------
# Parallel driver: ProcessPoolExecutor MAP (convert_one_bag) / REDUCE
# (merge_shards) across every mcap under rosbags_dirs.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConversionSummary:
    """Outcome of a full run_conversion call across every discovered bag."""

    converted: int
    skipped: int
    skipped_paths: list[str]


_RESAMPLE_UNSET = object()


def _detect_first_bag_rate(rosbags_dirs: list[str]) -> float | None:
    """Read the first discovered bag once and return its /record_data rate.

    Shared by the fps-resolution entry points so a batch decodes the first
    bag exactly once to detect its rate.
    """
    bag_dirs: list[str] = []
    for d in rosbags_dirs:
        bag_dirs.extend(_discover_bag_dirs(d))
    if not bag_dirs:
        raise ValueError(f'No MCAP bags found under {rosbags_dirs}')
    reader = RosbagReader(bag_dirs[0])
    reader.construct_dataset()
    return _detect_rate_from_record_data(reader)


def _resolve_rate_decision(
    detected_fps: float | None,
    fps: int | None,
    strict_fps: bool = False,
) -> tuple[int, float | None]:
    """Resolve (labeled fps, resample rate) from an already-detected rate.

    Pure — no bag I/O. resample_to_rate is None when each bag's
    /record_data is used as-is, or float(fps) when every bag is resynced to
    fps via causal ZOH.

    When fps is None (auto-detect) the label is the detected rate rounded to
    an int and resampling is never requested — matching the sequential
    create_lerobot_dataset(fps=None) path. Rounding a non-integer native
    rate could otherwise push it past _FPS_MATCH_TOLERANCE and silently
    resample bags the sequential path leaves untouched.
    """
    if fps is None:
        if detected_fps is None:
            raise ValueError(
                'Could not detect rate from bag /record_data and no --fps '
                'was provided. Either pass --fps explicitly, or convert a '
                'bag with /record_data populated by the recorder.')
        resolved = int(round(detected_fps))
        logger.info(
            'fps unspecified — using detected rate %.2f Hz (rounded to %d)',
            detected_fps, resolved)
        return resolved, None

    if detected_fps is None:
        return fps, None
    if abs(detected_fps - fps) / max(detected_fps, 1.0) <= _FPS_MATCH_TOLERANCE:
        logger.info(
            'fps %d ≈ detected rate %.2f Hz; using bag /record_data as-is',
            fps, detected_fps)
        return fps, None
    if strict_fps:
        raise ValueError(
            f'--fps {fps} does not match bag rate {detected_fps:.2f} Hz '
            f'(±{_FPS_MATCH_TOLERANCE:.0%}) and --strict-fps is set. Either '
            f'pass --fps {int(round(detected_fps))} or drop --strict-fps to '
            'resample.')
    logger.warning(
        'fps %d differs from detected rate %.2f Hz — '
        'resampling all bags via causal ZOH at %d Hz',
        fps, detected_fps, fps)
    return fps, float(fps)


def _resolve_rates_for_batch(
    rosbags_dirs: list[str],
    fps: int | None,
    strict_fps: bool = False,
) -> tuple[int, float | None]:
    """Detect the first bag's rate once and resolve (labeled fps,
    resample rate) for the whole batch. See _resolve_rate_decision.
    """
    return _resolve_rate_decision(
        _detect_first_bag_rate(rosbags_dirs), fps, strict_fps)


def _resolve_resample_to_rate(
    bag_dirs: list[str],
    fps: int,
    strict_fps: bool = False,
) -> float | None:
    """Decide, once for the whole batch, whether every bag should be
    resampled to fps via causal ZOH before conversion.

    Reads the first bag's /record_data rate and defers the decision to
    _resolve_rate_decision. Only the first bag's rate is checked, so the
    decision is shared across the whole batch and every shard stays
    mergeable. Used as run_conversion's fallback when the caller does not
    pass an explicit resample_to_rate.
    """
    reader = RosbagReader(bag_dirs[0])
    reader.construct_dataset()
    detected_fps = _detect_rate_from_record_data(reader)
    return _resolve_rate_decision(detected_fps, fps, strict_fps)[1]


def run_conversion(
    rosbags_dirs: list[str],
    output_dir: str,
    embodiment: EmbodimentConfig,
    task: str,
    fps: int,
    num_workers: int | None = None,
    codec_opts=None,
    strict_fps: bool = False,
    resample_to_rate=_RESAMPLE_UNSET,
) -> ConversionSummary:
    """Convert every MCAP bag under rosbags_dirs into one LeRobot dataset.

    MAP: convert_one_bag runs once per bag, in parallel, across a
    ProcessPoolExecutor(max_workers=num_workers or os.cpu_count()) — each
    call writes a standalone single-episode shard to its own temp
    directory, with no shared state between workers. REDUCE: merge_shards
    concatenates the successful shards, in sorted bag order, into
    output_dir, then the modality config is (re)written from embodiment
    and the OR-reduction of every shard's any_real_efforts flag.

    A bag that fails to convert is logged and counted in the returned
    ConversionSummary rather than aborting the whole batch; only a bag
    list that yields zero successful shards is fatal.

    fps must already be resolved (see _resolve_rates_for_batch) — every
    shard has to agree on the same rate, so it's resolved once by the caller
    rather than per-worker. Whether bags are resampled to that rate is a
    separate decision passed identically to every convert_one_bag call: pass
    resample_to_rate (None = use each bag's /record_data as-is, float = ZOH
    resync every bag to that rate) to control it explicitly, or leave it
    unset to have it resolved here from the first bag via
    _resolve_resample_to_rate. main() resolves both together from a single
    first-bag decode and passes resample_to_rate in; leaving it unset (as
    tests do with an explicit fps) resolves it here instead. strict_fps is
    honored identically to the sequential path: it turns a detected-rate
    mismatch into an error instead of a resample.
    """
    # Deferred import: merge.py imports build_features from this module, so
    # importing merge_shards at module scope here would be circular.
    from isaac_ros_mcap_lerobot_converter.merge import merge_shards

    if codec_opts is None:
        codec_opts = software_codec_opts()

    bag_dirs: list[str] = []
    for d in rosbags_dirs:
        bag_dirs.extend(_discover_bag_dirs(d))
    bag_dirs.sort()

    if not bag_dirs:
        logger.error('No MCAP bags found under %s', rosbags_dirs)
        return ConversionSummary(converted=0, skipped=0, skipped_paths=[])

    if resample_to_rate is _RESAMPLE_UNSET:
        resample_to_rate = _resolve_resample_to_rate(
            bag_dirs, fps, strict_fps=strict_fps)

    resolved_workers = num_workers or os.cpu_count()

    tmp_root = tempfile.mkdtemp(prefix='mcap_to_lerobot_shards_')
    shard_dirs = [
        os.path.join(tmp_root, f'shard_{i:06d}')
        for i in range(len(bag_dirs))
    ]

    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=resolved_workers) as pool:
            # ProcessPoolExecutor.map returns results in input order
            # regardless of completion order, so bag_dirs/shard_dirs/results
            # stay aligned without any extra bookkeeping.
            results = list(pool.map(
                convert_one_bag,
                bag_dirs,
                itertools.repeat(embodiment),
                itertools.repeat(task),
                itertools.repeat(fps),
                shard_dirs,
                itertools.repeat(codec_opts),
                itertools.repeat(resample_to_rate),
            ))

        successful_shards = []
        skipped_paths = []
        any_real_efforts = False
        for bag_dir, result in zip(bag_dirs, results):
            if result.ok:
                successful_shards.append(result.shard_dir)
                any_real_efforts = any_real_efforts or result.any_real_efforts
            else:
                logger.error('Failed to convert %s: %s', bag_dir, result.error)
                skipped_paths.append(bag_dir)

        if skipped_paths:
            logger.warning(
                '%d/%d bag(s) failed to convert: %s',
                len(skipped_paths), len(bag_dirs), skipped_paths)

        if not successful_shards:
            raise ValueError(f'All {len(bag_dirs)} bag(s) failed to convert')

        merge_shards(successful_shards, output_dir, embodiment, task)

        camera_key_map = {
            camera.topic: camera.name for camera in embodiment.cameras}
        _write_modality_json(output_dir, camera_key_map, embodiment)
        _write_modality_config_defaults(output_dir, embodiment, any_real_efforts)
        logger.info('LeRobot dataset saved to %s', output_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return ConversionSummary(
        converted=len(successful_shards),
        skipped=len(skipped_paths),
        skipped_paths=skipped_paths,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MCAP rosbags directly into a LeRobot dataset.',
    )
    parser.add_argument(
        '--rosbags-dir', type=str, action='append', required=True,
        help='Directory containing one or more MCAP rosbag sub-folders. '
             'Repeat the flag to merge multiple recording sessions into one '
             'dataset (e.g. --rosbags-dir session_A --rosbags-dir session_B).',
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for the LeRobot dataset.',
    )
    parser.add_argument('--task', type=str, required=True, help='Task label for the dataset.')
    parser.add_argument(
        '--fps', type=int, default=None,
        help='Target dataset rate (Hz). When omitted, the rate is detected '
             'from the bag\'s /record_data. When set and within ±5%% of the '
             'detected rate, the bag\'s /record_data is used as-is. When set '
             'and differs, /record_data is regenerated via causal '
             'zero-order-hold at the requested rate (resampling).',
    )
    parser.add_argument(
        '--strict-fps', action='store_true',
        help='Error out instead of resampling when --fps does not match the '
             'bag\'s detected rate.',
    )
    parser.add_argument(
        '--embodiment', type=str, default='unitree_g1',
        help='Embodiment to convert to. Selects both the dataset schema '
             '(joint layout, action sources, teleop declarations and '
             'cameras) and the robot_type label written to info.json. Must '
             'be one of the packaged embodiment names, e.g. unitree_g1 or '
             'unitree_h2_sharpa_wave (see '
             'isaac_ros_mcap_lerobot_converter/embodiments/). Defaults to '
             'unitree_g1.',
    )
    parser.add_argument(
        '--num-workers', type=int, default=os.cpu_count(),
        help='Number of worker processes used to convert bags in parallel '
             '(one process per in-flight bag). Defaults to the number of '
             'CPUs on this machine.',
    )
    parser.add_argument(
        '--nvdec', type=str, choices=('auto', 'true', 'false'), default='auto',
        help='Use the NVIDIA h264_cuvid hardware decoder for camera video. '
             '"auto" (default) uses it if available, else falls back to '
             'software. "true" requires it and errors out if unavailable. '
             '"false" always uses software decode.',
    )
    parser.add_argument(
        '--nvenc', type=str, choices=('auto', 'true', 'false'), default='auto',
        help='Use the NVIDIA h264_nvenc hardware encoder for episode '
             'videos. "auto" (default) uses it if available, else falls '
             'back to software (LeRobot\'s default libsvtav1 encoder). '
             '"true" requires it and errors out if unavailable. "false" '
             'always uses software encode.',
    )
    args = parser.parse_args()

    valid_embodiments = available_embodiments()
    if args.embodiment not in valid_embodiments:
        parser.error(
            f'--embodiment {args.embodiment!r} is not a known embodiment. '
            f'Valid values: {", ".join(valid_embodiments)}'
        )

    return args


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    embodiment = load_embodiment(args.embodiment)
    # Resolved once here (not per-worker), and before any bag is opened, so
    # a "true" request that can't be satisfied fails immediately -- every
    # worker agrees on the same codec_opts.
    codec_opts = resolve_codec_opts(
        probe_hw_codecs(), nvenc=args.nvenc, nvdec=args.nvdec)
    # Decode the first bag once to resolve both the labeled fps and the
    # resample decision, then thread both into run_conversion so it does not
    # re-decode the first bag.
    fps, resample_to_rate = _resolve_rates_for_batch(
        args.rosbags_dir, args.fps, strict_fps=args.strict_fps)
    summary = run_conversion(
        rosbags_dirs=args.rosbags_dir,
        output_dir=args.output_dir,
        embodiment=embodiment,
        task=args.task,
        fps=fps,
        num_workers=args.num_workers,
        codec_opts=codec_opts,
        strict_fps=args.strict_fps,
        resample_to_rate=resample_to_rate,
    )
    if summary.skipped:
        logger.warning(
            'Converted %d bag(s), skipped %d: %s',
            summary.converted, summary.skipped, summary.skipped_paths)
    else:
        logger.info('Converted %d bag(s)', summary.converted)


if __name__ == '__main__':
    main()
