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
"""End-to-end test for MCAP-to-LeRobot conversion.

Runs the full converter once against the checked-in ``episode_001`` bag and
asserts the shape and content of the resulting LeRobot dataset.
"""

import json
import logging
import math
from pathlib import Path

import av
from isaac_ros_mcap_lerobot_converter.mcap_to_lerobot import (
    create_lerobot_dataset,
    G1_CANONICAL_ACTION_JOINT_ORDER,
    G1_CANONICAL_STATE_JOINT_ORDER,
)
from isaac_ros_mcap_lerobot_converter.rosbag_reader import RosbagReader
import numpy as np
import pyarrow.parquet as pq
import pytest

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

TEST_BAG_DIR = Path(__file__).parent / 'data' / 'episode_001'

EXPECTED_FRAMES = 154
EXPECTED_COMPRESSED_IMAGE_MESSAGES = 157
EXPECTED_STATE_DIM = 43
EXPECTED_ACTION_DIM = 43
EXPECTED_NAVIGATE_DIM = 3
EXPECTED_BASE_HEIGHT_DIM = 1
EXPECTED_CAMERA_TOPIC = '/camera/color/image_compressed'
EXPECTED_CAMERA_KEY = 'ego_view'
EXPECTED_IMAGE_HEIGHT = 480
EXPECTED_IMAGE_WIDTH = 640
EXPECTED_FPS = 30
EXPECTED_ROBOT_TYPE = 'g1'
EXPECTED_TASK = 'grasp with right hand and close left fingers'


@pytest.fixture(scope='session')
def converted_dataset(tmp_path_factory):
    """Convert the test bag once and share the output across all tests."""
    output_dir = tmp_path_factory.mktemp('lerobot') / 'dataset'
    create_lerobot_dataset(
        rosbags_dirs=[str(TEST_BAG_DIR.parent)],
        output_dir=str(output_dir),
        task=EXPECTED_TASK,
        fps=EXPECTED_FPS,
        robot_type=EXPECTED_ROBOT_TYPE,
    )
    return output_dir


@pytest.fixture(scope='session')
def parquet_table(converted_dataset):
    """Load the single episode parquet file once."""
    return pq.read_table(
        converted_dataset / 'data' / 'chunk-000' / 'episode_000000.parquet')


class TestMcapToLerobotConversion:
    """Verify the output of create_lerobot_dataset on a real MCAP bag."""

    def test_frame_count(self, converted_dataset):
        info = json.loads(
            (converted_dataset / 'meta' / 'info.json').read_text())
        assert info['total_frames'] == EXPECTED_FRAMES
        assert info['total_episodes'] == 1

    def test_observation_state_shape(self, parquet_table):
        obs = np.asarray(parquet_table['observation.state'].to_pylist())
        assert obs.shape == (EXPECTED_FRAMES, EXPECTED_STATE_DIM)

    def test_action_shape(self, parquet_table):
        action = np.asarray(parquet_table['action'].to_pylist())
        assert action.shape == (EXPECTED_FRAMES, EXPECTED_ACTION_DIM)

    def test_navigate_command_shape(self, parquet_table):
        nav = np.asarray(parquet_table['action.navigate_command'].to_pylist())
        assert nav.shape == (EXPECTED_FRAMES, EXPECTED_NAVIGATE_DIM)

    def test_base_height_command_shape(self, parquet_table):
        height = np.asarray(
            parquet_table['action.base_height_command'].to_pylist())
        assert height.shape == (EXPECTED_FRAMES, EXPECTED_BASE_HEIGHT_DIM)

    def test_video_exists_and_frame_count(self, converted_dataset):
        video_path = (
            converted_dataset / 'videos' / 'chunk-000'
            / f'observation.images.{EXPECTED_CAMERA_KEY}'
            / 'episode_000000.mp4'
        )
        assert video_path.is_file(), f'Video missing at {video_path}'

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            assert stream.frames == EXPECTED_FRAMES
            assert stream.width == EXPECTED_IMAGE_WIDTH
            assert stream.height == EXPECTED_IMAGE_HEIGHT

    def test_timestamps_monotonic(self, parquet_table):
        timestamps = np.asarray(parquet_table['timestamp'].to_pylist())
        assert len(timestamps) == EXPECTED_FRAMES
        assert np.all(np.diff(timestamps) > 0), \
            'Timestamps are not strictly increasing'

    def test_timestamp_spacing(self, parquet_table):
        timestamps = np.asarray(parquet_table['timestamp'].to_pylist())
        diffs = np.diff(timestamps)
        expected_dt = 1.0 / EXPECTED_FPS
        assert np.allclose(diffs, expected_dt, atol=1e-4), (
            f'Timestamp spacing {diffs[:5]} does not match 1/fps={expected_dt}')

    def test_modality_json(self, converted_dataset):
        modality = json.loads(
            (converted_dataset / 'meta' / 'modality.json').read_text())

        # State is a 43-joint G1 layout split into body-part groups.
        state_end = max(g['end'] for g in modality['state'].values())
        assert state_end == EXPECTED_STATE_DIM

        assert modality['action']['navigate_command']['original_key'] == (
            'action.navigate_command')
        assert modality['action']['base_height_command']['original_key'] == (
            'action.base_height_command')

        assert EXPECTED_CAMERA_KEY in modality['video']
        assert modality['video'][EXPECTED_CAMERA_KEY]['original_key'] == (
            f'observation.images.{EXPECTED_CAMERA_KEY}')

    def test_info_json(self, converted_dataset):
        info = json.loads(
            (converted_dataset / 'meta' / 'info.json').read_text())
        assert info['fps'] == EXPECTED_FPS
        assert info['robot_type'] == EXPECTED_ROBOT_TYPE
        assert info['total_frames'] == EXPECTED_FRAMES
        assert info['total_episodes'] == 1

        features = info['features']
        assert features['observation.state']['shape'] == [EXPECTED_STATE_DIM]
        assert features['action']['shape'] == [EXPECTED_ACTION_DIM]
        assert features['action.navigate_command']['shape'] == [
            EXPECTED_NAVIGATE_DIM]
        assert features['action.base_height_command']['shape'] == [
            EXPECTED_BASE_HEIGHT_DIM]
        video_key = f'observation.images.{EXPECTED_CAMERA_KEY}'
        assert features[video_key]['dtype'] == 'video'
        assert features[video_key]['shape'] == [
            EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH, 3]

    def test_dataset_reloadable(self, converted_dataset):
        dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(converted_dataset))
        assert len(dataset) == EXPECTED_FRAMES

        sample = dataset[0]
        assert 'observation.state' in sample
        assert 'action' in sample
        assert 'action.navigate_command' in sample
        assert 'action.base_height_command' in sample
        assert f'observation.images.{EXPECTED_CAMERA_KEY}' in sample
        assert sample['observation.state'].shape == (EXPECTED_STATE_DIM,)
        assert sample['action'].shape == (EXPECTED_ACTION_DIM,)
        assert sample['action.navigate_command'].shape == (
            EXPECTED_NAVIGATE_DIM,)
        assert sample['action.base_height_command'].shape == (
            EXPECTED_BASE_HEIGHT_DIM,)

    def test_joint_values_reasonable(self, parquet_table):
        obs = np.asarray(parquet_table['observation.state'].to_pylist())
        action = np.asarray(parquet_table['action'].to_pylist())

        for name, arr in (('observation.state', obs), ('action', action)):
            assert np.all(np.isfinite(arr)), f'{name} contains non-finite values'
            assert np.all(np.abs(arr) <= 2 * math.pi + 1e-3), (
                f'{name} has joint values outside [-2π, 2π]')

    def test_action_reordered_into_canonical_joint_layout(self, parquet_table):
        """Every action row must equal one /applied_joint_commands msg
        permuted into G1_CANONICAL_ACTION_JOINT_ORDER.  The action-side
        hand layout is index/middle/thumb for both hands (what the policy's
        target joint element_names dictate, consumed by OutputBuilder).
        Writing in publisher or state-canonical order silently misroutes
        finger commands at replay.
        """
        self._assert_parquet_rows_match_permuted_msgs(
            parquet_table=parquet_table,
            parquet_col='action',
            bag_topic='/applied_joint_commands',
            # JointCommand's field is `names` (plural).
            name_attr='names',
            canonical_order=G1_CANONICAL_ACTION_JOINT_ORDER,
            canonical_label='G1_CANONICAL_ACTION_JOINT_ORDER',
        )

    def test_state_reordered_into_canonical_joint_layout(self, parquet_table):
        """Every state row must equal one /joint_states msg permuted into
        G1_CANONICAL_STATE_JOINT_ORDER.  The state-side hand layout is
        thumb/middle/index for the left hand and thumb/index/middle for
        the right (asymmetric — matches the physical robot's observation
        element_names in the GR00T policy YAML).
        """
        self._assert_parquet_rows_match_permuted_msgs(
            parquet_table=parquet_table,
            parquet_col='observation.state',
            bag_topic='/joint_states',
            # sensor_msgs/JointState's field is `name` (singular).
            name_attr='name',
            canonical_order=G1_CANONICAL_STATE_JOINT_ORDER,
            canonical_label='G1_CANONICAL_STATE_JOINT_ORDER',
        )

    @staticmethod
    def _assert_parquet_rows_match_permuted_msgs(
        parquet_table, parquet_col, bag_topic, name_attr,
        canonical_order, canonical_label,
    ):
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import get_typestore, Stores
        from rosbags.typesys.msg import get_types_from_msg

        typestore = get_typestore(Stores.ROS2_JAZZY)
        with Reader(str(TEST_BAG_DIR)) as reader:
            add_types = {}
            for c in reader.connections:
                if c.msgdef:
                    d = getattr(c.msgdef, 'data', c.msgdef)
                    if isinstance(d, str) and d.strip():
                        try:
                            add_types.update(get_types_from_msg(d, c.msgtype))
                        except Exception:
                            # Non-standard / malformed msgdef; skip gracefully
                            # but keep the trace available for debugging.
                            logger.debug(
                                'Skipping type registration for %s',
                                c.msgtype, exc_info=True,
                            )
            typestore.register(add_types)
            msgs = []
            for conn, _, raw in reader.messages():
                if conn.topic == bag_topic:
                    m = typestore.deserialize_cdr(raw, conn.msgtype)
                    msgs.append((list(getattr(m, name_attr)), list(m.position)))

        def permute_to_canonical(names, positions):
            lookup = dict(zip(names, positions, strict=True))
            return np.array([lookup[n] for n in canonical_order])

        permuted = np.stack([permute_to_canonical(n, p) for n, p in msgs])
        rows = np.asarray(parquet_table[parquet_col].to_pylist())

        for i, row in enumerate(rows):
            diffs = np.abs(permuted - row).max(axis=1)
            assert diffs.min() < 1e-7, (
                f'{parquet_col} row {i} does not match any {bag_topic} msg '
                f'permuted into {canonical_label} — indicates the parquet '
                f'is in MCAP-native order, not canonical.'
            )

    def test_state_and_action_hand_orderings_diverge(self):
        """The whole point of keeping two canonical orders is that state
        and action disagree inside the hand slices. If they ever drift
        back to identical lists, one or both has desynced from the policy
        YAML — fail loudly so the drift is noticed before causing silent
        finger misrouting at replay.
        """
        assert len(G1_CANONICAL_STATE_JOINT_ORDER) == 43
        assert len(G1_CANONICAL_ACTION_JOINT_ORDER) == 43
        # Legs, waist, arms (first 29 joints) are identical.
        assert (G1_CANONICAL_STATE_JOINT_ORDER[:29]
                == G1_CANONICAL_ACTION_JOINT_ORDER[:29])
        # Hand slices must differ.
        assert (G1_CANONICAL_STATE_JOINT_ORDER[29:]
                != G1_CANONICAL_ACTION_JOINT_ORDER[29:]), (
            'State and action canonical orders match in the hand slices — '
            'one of them has desynced from the GR00T policy YAML.'
        )

    def test_modality_slices_match_canonical_joint_groups(
        self, converted_dataset,
    ):
        """modality.json slice names must match the canonical joint groups.
        Assumes the checked-in bag has 43 G1 joints.
        """
        modality = json.loads(
            (converted_dataset / 'meta' / 'modality.json').read_text())
        expected_groups = {
            'left_leg': (0, 6),
            'right_leg': (6, 12),
            'waist': (12, 15),
            'left_arm': (15, 22),
            'right_arm': (22, 29),
            'left_hand': (29, 36),
            'right_hand': (36, 43),
        }
        # Joint substrings expected in each canonical body-part slice.
        group_joint_substrings = {
            'left_leg': ('left_hip', 'left_knee', 'left_ankle'),
            'right_leg': ('right_hip', 'right_knee', 'right_ankle'),
            'waist': ('waist',),
            'left_arm': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_arm': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_hand': ('left_hand',),
            'right_hand': ('right_hand',),
        }
        key_to_canonical = {
            'state': G1_CANONICAL_STATE_JOINT_ORDER,
            'action': G1_CANONICAL_ACTION_JOINT_ORDER,
        }
        for key, canonical in key_to_canonical.items():
            for group, (start, end) in expected_groups.items():
                sl = modality[key][group]
                assert sl['start'] == start and sl['end'] == end, (
                    f'modality.{key}.{group} = {sl}, expected '
                    f'start={start} end={end}'
                )
                for jn in canonical[start:end]:
                    assert any(sub in jn for sub in group_joint_substrings[group]), (
                        f'joint {jn!r} at canonical {key} index falls in '
                        f'group {group!r} — ordering is inconsistent with '
                        f'the modality layout.'
                    )


class TestH264Decoding:
    """Verify raw-H264 PyAV decoding recovers every compressed-image message."""

    def test_decoded_frame_count_matches_messages(self):
        reader = RosbagReader(str(TEST_BAG_DIR))
        reader.construct_dataset()
        assert EXPECTED_CAMERA_TOPIC in reader.compressed_camera_image_topics

        frames = reader.decode_video_frames(EXPECTED_CAMERA_TOPIC)
        timestamps = reader.compressed_camera_image_timestamps[
            EXPECTED_CAMERA_TOPIC]

        assert len(timestamps) == EXPECTED_COMPRESSED_IMAGE_MESSAGES
        # Without explicit decoder flush PyAV can silently drop up to a few
        # trailing B-frames from the raw Annex-B stream. Require a full
        # match — any shortfall here means the flush regressed.
        assert len(frames) == len(timestamps), (
            f'Decoded {len(frames)} frames but bag contains '
            f'{len(timestamps)} compressed image messages')


class TestShape1CompatScope:
    """The LeRobot shape-(1,) monkey-patch must not leak out of the context."""

    def test_patch_is_restored_after_conversion(self, converted_dataset):
        # Import after the fixture has run (and thus after the converter has
        # entered/exited the compat context manager). The function reference
        # we see here must be the original LeRobot implementation.
        from lerobot.datasets import utils, lerobot_dataset
        assert (
            utils.get_hf_features_from_features.__module__
            == 'lerobot.datasets.utils'
        )
        assert (
            lerobot_dataset.get_hf_features_from_features.__module__
            == 'lerobot.datasets.utils'
        )
