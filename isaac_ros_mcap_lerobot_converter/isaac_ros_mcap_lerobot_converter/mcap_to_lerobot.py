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
        --task gear_insertion

Each immediate subdirectory of ``--rosbags-dir`` that contains an MCAP bag
(identified by a ``metadata.yaml`` file) is treated as one episode.
"""

import argparse
import contextlib
import importlib.metadata
import json
import logging
import os

from isaac_ros_mcap_lerobot_converter.rosbag_reader import RosbagReader
import isaac_ros_mcap_lerobot_converter.structures as structures

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

import numpy as np

logger = logging.getLogger(__name__)


# Canonical GR00T UNITREE_G1 joint orders (43 dims each). MUST stay
# aligned with the policy YAML's ``semantic.observations.*.element_names``
# (state) and ``semantic.actions.*.element_names`` (action). State and
# action differ in the hand slices: state uses thumb/middle/index for the
# left hand and thumb/index/middle for the right (asymmetric per the
# physical robot's element naming), while action uses index/middle/thumb
# for both hands (what JointCommandConverter writes back). MCAP publishers
# typically emit joints in an interleaved order — both observation.state
# and action are reordered into the appropriate canonical layout so
# modality.json slices and downstream replay both index the right joints.
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


# Body-part slice ranges in G1's 43-element canonical layout. Used to split
# ``observation.state`` and ``action`` into per-body-part modality slices,
# and to split ``action.effort`` into per-body-part columns so each can be
# referenced as a standalone modality key.
G1_JOINT_GROUPS: dict[str, tuple[int, int]] = {
    'left_leg':   (0, 6),
    'right_leg':  (6, 12),
    'waist':      (12, 15),
    'left_arm':   (15, 22),
    'right_arm':  (22, 29),
    'left_hand':  (29, 36),
    'right_hand': (36, 43),
}


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


# ---------------------------------------------------------------------------
# Time-synchronization: build trajectory from in-memory reader data
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Strip leading '/' so topic names compare consistently."""
    return name.lstrip('/')


def _match_camera_topic(
    record_topic: str,
    bag_camera_topics: list[str],
) -> str | None:
    """Match a RecordData topic name to the actual compressed image topic.

    The data flywheel records the *base* image topic (e.g.
    ``/camera_1/color/image_raw``) while the bag stores the transport
    variant (e.g. ``/camera_1/color/image_raw_compressed`` or
    ``/camera_1/color/image_raw/compressed``).
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
) -> list[structures.TrainingTrajectoryItem]:
    """Build a time-synchronized trajectory directly from reader data."""
    camera_topics = reader.compressed_camera_image_topics
    has_teleop = bool(reader.navigate_commands or reader.base_height_commands)
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

        # Reorder to the GR00T G1 canonical layout when the bag's 43 joints
        # match (auto-detected); otherwise preserve the publisher's order so
        # non-G1 bags still work.  State and action use DIFFERENT canonical
        # orderings in the hand slices — see the G1_CANONICAL_*_JOINT_ORDER
        # comments.  Writing either column in publisher order silently
        # breaks every downstream consumer that trusts modality.json.
        if set(command_js.names) == set(G1_CANONICAL_STATE_JOINT_ORDER):
            state_order = G1_CANONICAL_STATE_JOINT_ORDER
            action_order = G1_CANONICAL_ACTION_JOINT_ORDER
        else:
            state_order = command_js.names
            action_order = command_js.names
        observation = _joint_state_to_observation(joint_state, order=state_order)
        action = _joint_state_to_observation(command_js, order=action_order)

        teleop_cmd = None
        if has_teleop:
            nav = None
            height = None
            if twist_stamp is not None:
                nav = nav_index.get(_ts_key(twist_stamp))
            if pose_stamp is not None:
                height = height_index.get(_ts_key(pose_stamp))
            if nav is not None or height is not None:
                teleop_cmd = structures.TeleopCommand(
                    navigate_command=nav if nav is not None else [0.0, 0.0, 0.0],
                    base_height_command=height if height is not None else [0.0],
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

def _sanitize_camera_key(topic: str) -> str:
    """Turn a ROS topic name into a safe LeRobot feature key component."""
    return topic.strip('/').replace('/', '_')


def _build_camera_key_map(
    camera_topics: list[str],
    video_key: str | None,
) -> dict[str, str]:
    """Map each camera topic to its LeRobot feature key suffix."""
    if video_key and len(camera_topics) == 1:
        return {camera_topics[0]: video_key}
    return {t: _sanitize_camera_key(t) for t in camera_topics}


_LEROBOT_SHAPE1_BUG_VERSIONS = frozenset({'0.3.2'})


@contextlib.contextmanager
def _lerobot_shape1_compat():
    """Scoped workaround for the LeRobot 0.3.2 shape-(1,) feature bug.

    LeRobot 0.3.2 maps ``shape == (1,)`` to a scalar ``datasets.Value`` in
    ``get_hf_features_from_features`` but ``add_frame`` validates the same
    feature as a 1-element ``np.ndarray``. ``save_episode`` then fails
    because ``Value.encode_example`` calls ``float(ndarray)``.

    We replace the mapping with ``Sequence(length=1)`` only for user
    features (those carrying a ``names`` list). Default LeRobot scalar
    features like ``frame_index`` or ``timestamp`` store Python scalars
    and need to stay as ``Value``.

    The override is installed inside this context manager and restored on
    exit so other code in the same process sees unpatched LeRobot. On any
    LeRobot version outside ``_LEROBOT_SHAPE1_BUG_VERSIONS`` this is a
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
    # ``from .utils import get_hf_features_from_features`` so the symbol is
    # rebound at import time and patching only ``utils`` would miss it.
    _lerobot_utils.get_hf_features_from_features = patched
    _lerobot_dataset.get_hf_features_from_features = patched
    try:
        yield
    finally:
        _lerobot_utils.get_hf_features_from_features = original
        _lerobot_dataset.get_hf_features_from_features = original


_FPS_MATCH_TOLERANCE = 0.05  # ±5% — well under the worst-case recorder jitter.


def _detect_rate_from_record_data(reader: RosbagReader) -> float | None:
    """Return the median rate of ``/record_data`` (Hz), or ``None`` if there
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


def create_lerobot_dataset(
    rosbags_dirs: list[str],
    output_dir: str,
    task: str,
    fps: int | None = None,
    robot_type: str = 'unitree_g1',
    video_key: str | None = 'ego_view',
    max_frames_per_episode: int | None = None,
    strict_fps: bool = False,
):
    """Create a LeRobot dataset from one or more directories of MCAP rosbags.

    Each sub-directory containing a ``metadata.yaml`` is treated as a
    separate episode. Pass multiple ``rosbags_dirs`` to merge several
    recording sessions into one dataset; episodes are numbered sequentially
    across the supplied directories in the order given.

    When sessions were recorded at different rates, pass ``fps`` explicitly
    so the converter resamples each bag to the same target rate. Otherwise
    the rate detected from the first bag is applied to all and any
    mis-matched sessions get rejected by the ±5% tolerance check.

    Args:
        fps: Target dataset rate (Hz).  Resolution rules:

            * ``None`` (default) — use the bag's existing ``/record_data`` as
              the row source; the dataset's labeled fps is the rate detected
              from those records.
            * Set, matches detected rate within ±5% — same as ``None``, just
              with the supplied label.
            * Set, differs from detected rate — discard the bag's
              ``/record_data`` and regenerate it via causal ZOH at the
              requested rate.  The dataset is then both physically sampled
              and labeled at ``fps``.

        strict_fps: When ``True``, error out instead of resampling when the
            requested ``fps`` does not match the bag's detected rate. Useful
            for CI pipelines that want to catch mismatches between
            recording- and conversion-time rate assumptions.
        video_key: If set and there is exactly one camera, use this as the
            image feature key instead of the sanitized topic name.
        max_frames_per_episode: If set, only the first N frames of each
            episode are written.  Useful for fast smoke-tests.
    """
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

    camera_topics = first_reader.compressed_camera_image_topics
    first_frames = {
        t: first_reader.decode_video_frames(t) for t in camera_topics
    }
    first_traj = build_synchronized_trajectory(first_reader, first_frames)

    if max_frames_per_episode is not None:
        first_traj = first_traj[:max_frames_per_episode]

    if not first_traj:
        logger.error('First bag yielded zero synchronized frames')
        return

    num_state = len(first_traj[0].observation.data)
    num_action = len(first_traj[0].action.data)
    # Teleop stamps may not be present on the very first synchronized frame
    # (e.g. the VR headset starts publishing slightly after recording
    # begins). Scan all frames so teleop features are always declared when
    # the episode contains any teleop at all.
    teleop_sample = next(
        (item.teleop_command for item in first_traj
         if item.teleop_command is not None),
        None,
    )
    has_teleop = teleop_sample is not None
    sample_frames = first_traj[0].image_frames
    if not sample_frames:
        logger.error('First bag has no camera data')
        return
    sample_cam = next(iter(sample_frames.values()))
    img_h, img_w = sample_cam.shape[:2]

    camera_key_map = _build_camera_key_map(camera_topics, video_key)

    features: dict = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (num_state,),
            'names': ['observation_to_policy'],
        },
        'action': {
            'dtype': 'float32',
            'shape': (num_action,),
            'names': ['action_from_policy'],
        },
    }
    # Per-body-part effort columns when the bag is full G1 (43-DOF). Lets
    # the trainer reference effort body parts as standalone modality keys
    # (``effort_left_arm``, etc.) without sharing a single ``action.effort``
    # column across multiple modality entries — sidesteps any dataloader
    # ambiguity about per-key column resolution. For non-G1 bags fall back
    # to a single flat ``action.effort`` column.
    if num_action == 43:
        for group, (s, e) in G1_JOINT_GROUPS.items():
            features[f'action.effort_{group}'] = {
                'dtype': 'float32',
                'shape': (e - s,),
                'names': ['action_from_policy'],
            }
    else:
        features['action.effort'] = {
            'dtype': 'float32',
            'shape': (num_action,),
            'names': ['action_from_policy'],
        }
    if has_teleop:
        features['action.navigate_command'] = {
            'dtype': 'float32',
            'shape': (len(teleop_sample.navigate_command),),
            'names': ['action_from_policy'],
        }
        features['action.base_height_command'] = {
            'dtype': 'float32',
            'shape': (len(teleop_sample.base_height_command),),
            'names': ['action_from_policy'],
        }

    for cam_topic in camera_topics:
        key = f'observation.images.{camera_key_map[cam_topic]}'
        features[key] = {
            'dtype': 'video',
            'shape': (img_h, img_w, 3),
            'names': ['height', 'width', 'channels'],
        }

    teleop_dims = (
        (len(teleop_sample.navigate_command),
         len(teleop_sample.base_height_command))
        if has_teleop else None
    )

    with _lerobot_shape1_compat():
        dataset = LeRobotDataset.create(
            root=output_dir,
            repo_id='isaac_ros_mcap_lerobot',
            fps=resolved_fps,
            robot_type=robot_type,
            features=features,
        )

        any_real_efforts = _add_trajectory_to_dataset(
            dataset, first_traj, camera_topics, camera_key_map, task,
            has_teleop, teleop_dims)
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
                traj = build_synchronized_trajectory(reader, frames)
                if max_frames_per_episode is not None:
                    traj = traj[:max_frames_per_episode]
                if traj:
                    ep_has_efforts = _add_trajectory_to_dataset(
                        dataset, traj, camera_topics, camera_key_map, task,
                        has_teleop, teleop_dims)
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
    _write_modality_json(
        output_dir, camera_key_map, num_state, num_action, teleop_dims)
    _write_modality_config_defaults(
        output_dir, num_state, num_action, has_teleop, any_real_efforts)
    logger.info('LeRobot dataset saved to %s', output_dir)


def _add_trajectory_to_dataset(
    dataset: LeRobotDataset,
    trajectory: list[structures.TrainingTrajectoryItem],
    camera_topics: list[str],
    camera_key_map: dict[str, str],
    task: str,
    has_teleop: bool = False,
    teleop_dims: tuple[int, int] | None = None,
) -> bool:
    """Append a trajectory to the dataset; returns whether real efforts were
    observed in any frame of the trajectory.

    For 43-DOF G1 bags, ``action.effort`` is split into per-body-part columns
    (``action.effort_left_leg`` etc.) using ``G1_JOINT_GROUPS``. Non-G1 bags
    keep a single flat ``action.effort`` column. When ``joint_efforts`` is
    empty on a frame the effort columns are zero-padded, but those frames
    don't count as "real" effort observations — the returned flag tracks
    whether any frame had non-empty ``joint_efforts`` so the autogen
    modality config can decide whether to register effort modality keys.
    """
    # Fallback shapes must match the feature declarations written at dataset
    # creation (derived from teleop_sample) — hard-coding (3, 1) here would
    # silently mismatch if the recorded teleop dims ever change.
    nav_dim, height_dim = teleop_dims if teleop_dims is not None else (3, 1)
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
        if num_action == 43:
            effort_arr = np.array(action_effort, dtype=np.float32)
            for group, (s, e) in G1_JOINT_GROUPS.items():
                frame[f'action.effort_{group}'] = effort_arr[s:e]
        else:
            frame['action.effort'] = np.array(action_effort, dtype=np.float32)
        if has_teleop:
            if item.teleop_command is not None:
                frame['action.navigate_command'] = np.array(
                    item.teleop_command.navigate_command, dtype=np.float32)
                frame['action.base_height_command'] = np.array(
                    item.teleop_command.base_height_command, dtype=np.float32)
            else:
                frame['action.navigate_command'] = np.zeros(nav_dim, dtype=np.float32)
                frame['action.base_height_command'] = np.zeros(height_dim, dtype=np.float32)
        for cam_topic in camera_topics:
            key = f'observation.images.{camera_key_map[cam_topic]}'
            frame[key] = item.image_frames[cam_topic]
        dataset.add_frame(frame, task=task)
    dataset.save_episode()
    return any_real_efforts


def _discover_bag_dirs(rosbags_dir: str) -> list[str]:
    """Return sorted list of sub-directories that contain a metadata.yaml.

    Supports two layouts:
    - Flat: ``rosbags_dir/<episode>/metadata.yaml``
    - Nested: ``rosbags_dir/<episode>/bag/metadata.yaml``
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
    num_state: int,
    num_action: int,
    teleop_dims: tuple[int, int] | None,
):
    """Write modality.json with GR00T UNITREE_G1 body part mapping.

    When ``num_state`` / ``num_action`` is 43, the slices below are
    authoritative — the trajectory build has already reordered
    ``observation.state`` into ``G1_CANONICAL_STATE_JOINT_ORDER`` and
    ``action`` into ``G1_CANONICAL_ACTION_JOINT_ORDER`` (they share
    body-part slice boundaries but differ within the hand slices).
    Any other joint count falls back to a flat layout so downstream
    consumers don't silently read mislabeled slices.

    TODO: the G1 layout is hardcoded below. Once a second robot type is
    supported, extract these groups into a ``RobotSchema``-style profile
    selected by ``robot_type`` rather than by joint count.
    """
    if num_state == 43:
        state_modality: dict = {
            g: {'start': s, 'end': e} for g, (s, e) in G1_JOINT_GROUPS.items()
        }
    else:
        state_modality = {'joint_state_position': {'start': 0, 'end': num_state}}

    # Action shares the same 43-joint canonical layout as state.  Any other
    # size means the upstream bag wasn't a full G1, so fall back to flat.
    if num_action == 43:
        action_modality: dict = {
            g: {'start': s, 'end': e} for g, (s, e) in G1_JOINT_GROUPS.items()
        }
        for group, (s, e) in G1_JOINT_GROUPS.items():
            action_modality[f'effort_{group}'] = {
                'original_key': f'action.effort_{group}',
                'start': 0,
                'end': e - s,
            }
    else:
        action_modality = {'joint_position': {'start': 0, 'end': num_action}}

    if teleop_dims is not None:
        nav_dim, height_dim = teleop_dims
        action_modality['navigate_command'] = {
            'original_key': 'action.navigate_command',
            'start': 0,
            'end': nav_dim,
        }
        action_modality['base_height_command'] = {
            'original_key': 'action.base_height_command',
            'start': 0,
            'end': height_dim,
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
# this file; copy it to ``new_embodiment_config.py`` before editing.
'''


def _abs_action_config_literal() -> str:
    return ('ActionConfig(rep=ActionRepresentation.ABSOLUTE, '
            'type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)')


def _rel_action_config_literal() -> str:
    return ('ActionConfig(rep=ActionRepresentation.RELATIVE, '
            'type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)')


def _write_modality_config_defaults(
    output_dir: str,
    num_state: int,
    num_action: int,
    has_teleop: bool,
    has_real_efforts: bool,
):
    """Write ``new_embodiment_config_defaults.py`` to the dataset root.

    Always regenerated — overwrites any prior version. Users who want
    custom training choices copy this to ``new_embodiment_config.py`` and
    edit the copy; the converter never touches that copy.
    """
    if num_state != 43 or num_action != 43:
        # Non-G1 datasets — skip autogen; user authors their own config
        # against whatever layout they have.
        logger.info(
            'Skipping new_embodiment_config_defaults.py: dataset is not G1 '
            '(num_state=%d, num_action=%d)', num_state, num_action)
        return

    state_keys = list(G1_JOINT_GROUPS.keys())

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
    # ``,`` goes inside the f-string so the trailing comment doesn't
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

# ``register_modality_config`` asserts the tag isn't already in
# ``MODALITY_CONFIGS``, so a second import of this file inside the same
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
        '--robot-type', type=str, default='unitree_g1',
        help='Robot type label (default: unitree_g1).',
    )
    parser.add_argument(
        '--video-key', type=str, default='ego_view',
        help='Feature key for single-camera video (default: ego_view). '
             'Pass empty string to use sanitized topic name.',
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    create_lerobot_dataset(
        rosbags_dirs=args.rosbags_dir,
        output_dir=args.output_dir,
        task=args.task,
        fps=args.fps,
        strict_fps=args.strict_fps,
        robot_type=args.robot_type,
        video_key=args.video_key or None,
    )


if __name__ == '__main__':
    main()
