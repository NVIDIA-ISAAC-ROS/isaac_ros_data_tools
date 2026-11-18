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

Runs the full converter once against the checked-in episode_001 bag and
asserts the shape and content of the resulting LeRobot dataset.
"""

from fractions import Fraction
import inspect
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import av
import isaac_ros_mcap_lerobot_converter.codec as mcap_to_lerobot_codec
from isaac_ros_mcap_lerobot_converter.codec import (
    HwCodecCaps,
    resolve_codec_opts,
    software_codec_opts,
)
from isaac_ros_mcap_lerobot_converter.embodiments import (
    available_embodiments,
    load_embodiment,
)
import isaac_ros_mcap_lerobot_converter.mcap_to_lerobot as mcap_to_lerobot
from isaac_ros_mcap_lerobot_converter.mcap_to_lerobot import (
    build_features,
    convert_one_bag,
    create_lerobot_dataset,
    G1_CANONICAL_ACTION_JOINT_ORDER,
    G1_CANONICAL_STATE_JOINT_ORDER,
    run_conversion,
)
from isaac_ros_mcap_lerobot_converter.rosbag_reader import RosbagReader
import isaac_ros_mcap_lerobot_converter.structures as structures
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import pytest

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _hw_codec_openable(name, mode):
    """Return True only if a hardware codec can actually be opened.

    ``probe_hw_codecs()`` reports whether a codec is *compiled into* PyAV, which
    is True on GPU-less CI runners where ``avcodec_open2()`` still fails with a
    permission error. The hardware tests only make sense when a usable GPU is
    present, so verify the codec opens rather than merely exists.
    """
    try:
        context = av.codec.CodecContext.create(name, mode)
        if mode == 'w':
            context.width = 256
            context.height = 256
            context.pix_fmt = 'yuv420p'
            context.time_base = Fraction(1, 30)
        context.open()
        return True
    except Exception:
        return False


TEST_BAG_DIR = Path(__file__).parent / 'data' / 'episode_001'
TEST_EMBODIMENT = 'unitree_g1'
TEST_FPS = 30

EXPECTED_FRAMES = 157
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

# The bundled bag's own /record_data has ~27.78 Hz native rate (154 records),
# which is >5% off EXPECTED_FPS=30 -- both create_lerobot_dataset and
# convert_one_bag therefore resample it, so converting at EXPECTED_FPS alone
# can never distinguish "resample only on mismatch" from "always resample".
# TestRunConversionDeepContentCorrectness instead resolves fps via
# auto-detection (None), which is by construction within
# _FPS_MATCH_TOLERANCE of the detected rate -- the one bag/fps combination
# in this file where a wrongly-forced resample actually changes the output
# (154 frames as-is vs. 146 if forcibly resynced at ~28 Hz).
EXPECTED_FRAMES_NO_RESAMPLE = 154


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

    def test_g1_parity_matches_build_features(self, converted_dataset):
        """G1 features written to info.json must match build_features exactly.

        The bundled bag is genuinely G1 (43-joint layout on /joint_states +
        /applied_joint_commands, no /xr_teleop/finger_joints), converted
        through the config-only path (embodiment_name defaults to
        unitree_g1). The written info.json features must match exactly
        what build_features derives from the unitree_g1 embodiment config:
        state/action dims, one action.effort_<group> column per
        action-modality group, the declared teleop columns, and one video
        feature per camera.
        """
        embodiment = load_embodiment('unitree_g1')
        expected = build_features(embodiment)
        for cam in embodiment.cameras:
            key = f'observation.images.{cam.name}'
            expected[key]['shape'] = [
                EXPECTED_IMAGE_HEIGHT, EXPECTED_IMAGE_WIDTH, 3]

        info = json.loads(
            (converted_dataset / 'meta' / 'info.json').read_text())
        actual = info['features']

        for key, expected_feature in expected.items():
            assert key in actual, f'missing feature {key!r} in info.json'
            assert list(actual[key]['shape']) == list(expected_feature['shape']), (
                f'{key}: expected shape {expected_feature["shape"]}, '
                f'got {actual[key]["shape"]}')
            assert actual[key]['dtype'] == expected_feature['dtype'], key

    def test_dataset_reloadable(self, converted_dataset):
        dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(converted_dataset))
        assert len(dataset) == EXPECTED_FRAMES

        # Read the tabular sample straight from the reloaded hf_dataset. Indexing
        # LeRobotDataset[i] would decode the episode video through lerobot's
        # torchcodec backend, which needs system FFmpeg that is absent from the
        # hermetic Bazel runfiles; frame decoding is covered separately by the
        # av-based tests (e.g. test_video_exists_and_frame_count).
        sample = dataset.hf_dataset[0]
        features = dataset.meta.features
        assert 'observation.state' in sample
        assert 'action' in sample
        assert 'action.navigate_command' in sample
        assert 'action.base_height_command' in sample
        assert f'observation.images.{EXPECTED_CAMERA_KEY}' in features
        assert np.asarray(sample['observation.state']).shape == (
            EXPECTED_STATE_DIM,)
        assert np.asarray(sample['action']).shape == (EXPECTED_ACTION_DIM,)
        assert np.asarray(sample['action.navigate_command']).shape == (
            EXPECTED_NAVIGATE_DIM,)
        assert np.asarray(sample['action.base_height_command']).shape == (
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
            # JointCommand's field is names (plural).
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
            # sensor_msgs/JointState's field is name (singular).
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


@pytest.fixture(scope='session')
def run_conversion_dataset(tmp_path_factory):
    """Convert the test bag once through run_conversion(num_workers=1) --
    the parallel production entry point main() actually calls -- and share
    the output across the deep-correctness tests below.

    Resolves fps via auto-detection (fps=None) rather than reusing
    EXPECTED_FPS, mirroring exactly what main() does: _resolve_rates_for_batch
    rounds the detected rate for the label and returns resample_to_rate=None
    for the auto-detect case, so no bag is resampled. This exercises the
    "use /record_data as-is" branch -- the branch that silently regresses if
    convert_one_bag/run_conversion ever goes back to unconditionally
    resampling every bag (see EXPECTED_FRAMES_NO_RESAMPLE).
    """
    embodiment = load_embodiment(TEST_EMBODIMENT)
    output_dir = tmp_path_factory.mktemp('lerobot_run_conversion') / 'dataset'
    resolved_fps, resample_to_rate = mcap_to_lerobot._resolve_rates_for_batch(
        [str(TEST_BAG_DIR.parent)], fps=None)
    summary = run_conversion(
        rosbags_dirs=[str(TEST_BAG_DIR.parent)],
        output_dir=str(output_dir),
        embodiment=embodiment,
        task=EXPECTED_TASK,
        fps=resolved_fps,
        num_workers=1,
        codec_opts=software_codec_opts(),
        resample_to_rate=resample_to_rate,
    )
    assert summary.converted == 1
    assert summary.skipped == 0
    return output_dir


@pytest.fixture(scope='session')
def run_conversion_parquet_table(run_conversion_dataset):
    """Load the single episode parquet file once."""
    return pq.read_table(
        run_conversion_dataset / 'data' / 'chunk-000' / 'episode_000000.parquet')


class TestRunConversionDeepContentCorrectness:
    """Deep content assertions ported to the production run_conversion
    entry point.

    main() goes exclusively through run_conversion, so
    TestMcapToLerobotConversion's create_lerobot_dataset assertions above
    don't exercise the code path the CLI actually runs. Every check here
    mirrors one already made there; agreement between the two (plus the
    exact EXPECTED_FRAMES_NO_RESAMPLE count) demonstrates
    convert_one_bag's resample-to-rate decision reproduces
    create_lerobot_dataset's fps-match-tolerance behavior instead of
    unconditionally resampling every bag.
    """

    def test_frame_count(self, run_conversion_dataset):
        info = json.loads(
            (run_conversion_dataset / 'meta' / 'info.json').read_text())
        assert info['total_frames'] == EXPECTED_FRAMES_NO_RESAMPLE
        assert info['total_episodes'] == 1

    def test_video_exists_and_frame_count(self, run_conversion_dataset):
        video_path = (
            run_conversion_dataset / 'videos' / 'chunk-000'
            / f'observation.images.{EXPECTED_CAMERA_KEY}'
            / 'episode_000000.mp4'
        )
        assert video_path.is_file(), f'Video missing at {video_path}'

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            assert stream.frames == EXPECTED_FRAMES_NO_RESAMPLE
            assert stream.width == EXPECTED_IMAGE_WIDTH
            assert stream.height == EXPECTED_IMAGE_HEIGHT

    def test_modality_json(self, run_conversion_dataset):
        modality = json.loads(
            (run_conversion_dataset / 'meta' / 'modality.json').read_text())

        state_end = max(g['end'] for g in modality['state'].values())
        assert state_end == EXPECTED_STATE_DIM

        assert modality['action']['navigate_command']['original_key'] == (
            'action.navigate_command')
        assert modality['action']['base_height_command']['original_key'] == (
            'action.base_height_command')

        assert EXPECTED_CAMERA_KEY in modality['video']
        assert modality['video'][EXPECTED_CAMERA_KEY]['original_key'] == (
            f'observation.images.{EXPECTED_CAMERA_KEY}')

    def test_info_json(self, run_conversion_dataset):
        info = json.loads(
            (run_conversion_dataset / 'meta' / 'info.json').read_text())
        # run_conversion has no independent robot_type override (unlike
        # create_lerobot_dataset's decoupled robot_type/embodiment_name
        # params) -- every shard is labeled with the embodiment name it was
        # built from; see convert_one_bag's LeRobotDataset.create call and
        # merge_shards, which copies that label through verbatim.
        assert info['robot_type'] == TEST_EMBODIMENT
        assert info['total_frames'] == EXPECTED_FRAMES_NO_RESAMPLE
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

    def test_dataset_reloadable(self, run_conversion_dataset):
        dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(run_conversion_dataset))
        assert len(dataset) == EXPECTED_FRAMES_NO_RESAMPLE

        # Tabular sample from the reloaded hf_dataset (see the note in
        # TestMcapToLerobotConversion.test_dataset_reloadable): avoids decoding
        # the episode video, which is not hermetic under Bazel.
        sample = dataset.hf_dataset[0]
        assert 'observation.state' in sample
        assert 'action' in sample
        assert np.asarray(sample['observation.state']).shape == (
            EXPECTED_STATE_DIM,)
        assert np.asarray(sample['action']).shape == (EXPECTED_ACTION_DIM,)

    def test_state_reordered_into_canonical_joint_layout(
        self, run_conversion_parquet_table,
    ):
        """Same permuted-message check as
        TestMcapToLerobotConversion.test_state_reordered_into_canonical_joint_layout,
        run against run_conversion's output.
        """
        TestMcapToLerobotConversion._assert_parquet_rows_match_permuted_msgs(
            parquet_table=run_conversion_parquet_table,
            parquet_col='observation.state',
            bag_topic='/joint_states',
            name_attr='name',
            canonical_order=G1_CANONICAL_STATE_JOINT_ORDER,
            canonical_label='G1_CANONICAL_STATE_JOINT_ORDER',
        )

    def test_action_reordered_into_canonical_joint_layout(
        self, run_conversion_parquet_table,
    ):
        """Same permuted-message check as
        TestMcapToLerobotConversion.test_action_reordered_into_canonical_joint_layout,
        run against run_conversion's output.
        """
        TestMcapToLerobotConversion._assert_parquet_rows_match_permuted_msgs(
            parquet_table=run_conversion_parquet_table,
            parquet_col='action',
            bag_topic='/applied_joint_commands',
            name_attr='names',
            canonical_order=G1_CANONICAL_ACTION_JOINT_ORDER,
            canonical_label='G1_CANONICAL_ACTION_JOINT_ORDER',
        )

    def test_effort_columns_match_bag(self, run_conversion_parquet_table):
        """Every action.effort_<group> row must equal the effort field of
        some real /applied_joint_commands message, permuted into
        G1_CANONICAL_ACTION_JOINT_ORDER -- the same technique as the two
        joint-ordering tests above, applied to effort instead of position.

        G1's single, unsynced action source means effort is read straight
        off /applied_joint_commands with no causal-ZOH merge, so this
        exercises real, non-synthetic, non-zero effort values end-to-end
        through run_conversion. The effort-column assertions in
        TestH2SharpaWaveConversion only cover create_lerobot_dataset, via
        a monkey-patched synthetic reader that can't cross the
        ProcessPoolExecutor boundary run_conversion uses.
        """
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
                            logger.debug(
                                'Skipping type registration for %s',
                                c.msgtype, exc_info=True,
                            )
            typestore.register(add_types)
            msgs = []
            for conn, _, raw in reader.messages():
                if conn.topic == '/applied_joint_commands':
                    m = typestore.deserialize_cdr(raw, conn.msgtype)
                    msgs.append((list(m.names), list(m.effort)))

        def permute_to_canonical(names, efforts):
            lookup = dict(zip(names, efforts, strict=True))
            return np.array([lookup[n] for n in G1_CANONICAL_ACTION_JOINT_ORDER])

        permuted = np.stack([
            permute_to_canonical(n, e) for n, e in msgs])
        assert np.any(permuted != 0), (
            'bag /applied_joint_commands has all-zero effort -- this test '
            'would pass trivially even if the converter zeroed every '
            'effort column')

        groups = ('left_leg', 'right_leg', 'waist', 'left_arm', 'right_arm',
                  'left_hand', 'right_hand')
        cols = [
            np.asarray(
                run_conversion_parquet_table[f'action.effort_{g}'].to_pylist())
            for g in groups
        ]
        rows = np.concatenate(cols, axis=1)
        assert rows.shape[1] == len(G1_CANONICAL_ACTION_JOINT_ORDER)

        for i, row in enumerate(rows):
            diffs = np.abs(permuted - row).max(axis=1)
            assert diffs.min() < 1e-4, (
                f'action.effort_<group> row {i} does not match any '
                f'/applied_joint_commands msg effort permuted into '
                f'G1_CANONICAL_ACTION_JOINT_ORDER — indicates effort is not '
                f'being read from the real bag.'
            )


class TestConvertOneBag:
    """Verify convert_one_bag writes a standalone single-episode shard."""

    def test_convert_one_bag_writes_shard(self, tmp_path):
        embodiment = load_embodiment(TEST_EMBODIMENT)
        shard_dir = tmp_path / 'shard'
        result = convert_one_bag(
            str(TEST_BAG_DIR), embodiment, task='t', fps=TEST_FPS,
            shard_dir=str(shard_dir), codec_opts=software_codec_opts())

        assert result.ok, result.error
        assert result.num_frames > 0
        assert result.shard_dir == str(shard_dir)
        assert (shard_dir / 'meta' / 'info.json').exists()

        info = json.loads((shard_dir / 'meta' / 'info.json').read_text())
        assert info['total_episodes'] == 1
        assert info['total_frames'] == result.num_frames

    def test_convert_one_bag_returns_error_result_on_failure(self, tmp_path):
        embodiment = load_embodiment(TEST_EMBODIMENT)
        shard_dir = tmp_path / 'shard'
        result = convert_one_bag(
            str(tmp_path / 'does_not_exist'), embodiment, task='t',
            fps=TEST_FPS, shard_dir=str(shard_dir),
            codec_opts=software_codec_opts())

        assert not result.ok
        assert result.num_frames == 0
        assert result.error


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

    def test_decoded_frame_count_matches_messages_with_software_codec_opts(self):
        """Same invariant, but with an explicit (forced-software) CodecOpts
        threaded through -- this is the path convert_one_bag actually uses.
        """
        reader = RosbagReader(str(TEST_BAG_DIR))
        reader.construct_dataset()

        frames = reader.decode_video_frames(
            EXPECTED_CAMERA_TOPIC, codec_opts=software_codec_opts())
        timestamps = reader.compressed_camera_image_timestamps[
            EXPECTED_CAMERA_TOPIC]

        assert len(frames) == len(timestamps) == EXPECTED_COMPRESSED_IMAGE_MESSAGES

    @pytest.mark.skipif(
        not _hw_codec_openable('h264_cuvid', 'r'),
        reason='usable nvdec (h264_cuvid) hardware not available here')
    def test_nvdec_decode_matches_software_frame_count(self):
        """When nvdec hardware is actually available, the hw path must
        recover the same number of frames as software (pixel-exact
        equality isn't asserted -- nvdec's chroma/scaling pipeline can
        differ slightly from libavcodec's software decoder).
        """
        reader = RosbagReader(str(TEST_BAG_DIR))
        reader.construct_dataset()
        codec_opts = resolve_codec_opts(
            HwCodecCaps(nvdec=True, nvenc=False), nvdec='true')

        frames = reader.decode_video_frames(
            EXPECTED_CAMERA_TOPIC, codec_opts=codec_opts)

        assert len(frames) == EXPECTED_COMPRESSED_IMAGE_MESSAGES
        assert all(f.ndim == 3 and f.shape[2] == 3 for f in frames)

    def test_nvdec_runtime_failure_falls_back_to_software_under_auto(
        self, monkeypatch, caplog,
    ):
        reader = RosbagReader(str(TEST_BAG_DIR))
        reader.construct_dataset()
        monkeypatch.setattr(
            RosbagReader, '_decode_h264_nvdec',
            staticmethod(lambda data: (_ for _ in ()).throw(
                RuntimeError('simulated nvdec failure'))))
        codec_opts = resolve_codec_opts(
            HwCodecCaps(nvdec=True, nvenc=False), nvdec='auto')

        with caplog.at_level(logging.WARNING):
            frames = reader.decode_video_frames(
                EXPECTED_CAMERA_TOPIC, codec_opts=codec_opts)

        assert len(frames) == EXPECTED_COMPRESSED_IMAGE_MESSAGES
        assert any('nvdec' in rec.message for rec in caplog.records)

    def test_nvdec_runtime_failure_raises_under_true(self, monkeypatch):
        reader = RosbagReader(str(TEST_BAG_DIR))
        reader.construct_dataset()
        monkeypatch.setattr(
            RosbagReader, '_decode_h264_nvdec',
            staticmethod(lambda data: (_ for _ in ()).throw(
                RuntimeError('simulated nvdec failure'))))
        codec_opts = resolve_codec_opts(
            HwCodecCaps(nvdec=True, nvenc=False), nvdec='true')

        with pytest.raises(RuntimeError, match='simulated nvdec failure'):
            reader.decode_video_frames(
                EXPECTED_CAMERA_TOPIC, codec_opts=codec_opts)


class TestNvencEncodeCompat:
    """Verify the nvenc encode hook's wiring, fallback and hard-error paths."""

    def test_use_nvenc_false_is_a_noop(self):
        with mcap_to_lerobot._nvenc_encode_compat(software_codec_opts()):
            from lerobot.datasets import lerobot_dataset
            assert (
                lerobot_dataset.encode_video_frames.__module__
                == 'lerobot.datasets.video_utils')

    def test_nvenc_failure_falls_back_to_software_under_auto(
        self, monkeypatch, tmp_path, caplog,
    ):
        codec_opts = mcap_to_lerobot_codec.CodecOpts(
            use_nvdec=False, use_nvenc=True, nvenc_required=False)
        monkeypatch.setattr(
            mcap_to_lerobot, '_encode_video_frames_hw',
            lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('simulated nvenc failure')))

        imgs_dir = tmp_path / 'imgs'
        imgs_dir.mkdir()
        Image.new('RGB', (32, 32)).save(imgs_dir / 'frame_000000.png')
        video_path = tmp_path / 'out' / 'episode_000000.mp4'

        with caplog.at_level(logging.WARNING):
            with mcap_to_lerobot._nvenc_encode_compat(codec_opts):
                from lerobot.datasets import lerobot_dataset
                lerobot_dataset.encode_video_frames(
                    imgs_dir, video_path, 10, overwrite=True)

        assert video_path.exists()
        assert any('nvenc' in rec.message for rec in caplog.records)

    def test_nvenc_failure_raises_under_true(self, monkeypatch, tmp_path):
        codec_opts = mcap_to_lerobot_codec.CodecOpts(
            use_nvdec=False, use_nvenc=True, nvenc_required=True)
        monkeypatch.setattr(
            mcap_to_lerobot, '_encode_video_frames_hw',
            lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('simulated nvenc failure')))

        imgs_dir = tmp_path / 'imgs'
        imgs_dir.mkdir()
        Image.new('RGB', (32, 32)).save(imgs_dir / 'frame_000000.png')
        video_path = tmp_path / 'out' / 'episode_000000.mp4'

        with mcap_to_lerobot._nvenc_encode_compat(codec_opts):
            from lerobot.datasets import lerobot_dataset
            with pytest.raises(RuntimeError, match='simulated nvenc failure'):
                lerobot_dataset.encode_video_frames(
                    imgs_dir, video_path, 10, overwrite=True)

    def test_patch_is_restored_after_context_exits(self):
        codec_opts = mcap_to_lerobot_codec.CodecOpts(
            use_nvdec=False, use_nvenc=True, nvenc_required=False)
        from lerobot.datasets import lerobot_dataset
        original = lerobot_dataset.encode_video_frames
        with mcap_to_lerobot._nvenc_encode_compat(codec_opts):
            assert lerobot_dataset.encode_video_frames is not original
        assert lerobot_dataset.encode_video_frames is original

    @pytest.mark.skipif(
        not _hw_codec_openable('h264_nvenc', 'w'),
        reason='usable nvenc (h264_nvenc) hardware not available here')
    def test_convert_one_bag_with_real_nvenc_produces_playable_video(
        self, tmp_path,
    ):
        """End-to-end: when nvenc hardware is actually available, a shard
        built with use_nvenc=True still produces a valid, decodable video
        with the expected frame count -- proving the encode bypasses
        LeRobot's h264_nvenc whitelist rejection successfully.
        """
        embodiment = load_embodiment(TEST_EMBODIMENT)
        shard_dir = tmp_path / 'shard'
        codec_opts = resolve_codec_opts(
            HwCodecCaps(nvdec=False, nvenc=True), nvenc='true')

        result = convert_one_bag(
            str(TEST_BAG_DIR), embodiment, task='t', fps=TEST_FPS,
            shard_dir=str(shard_dir), codec_opts=codec_opts)

        assert result.ok, result.error
        video_files = list((shard_dir / 'videos').rglob('*.mp4'))
        assert video_files, 'expected at least one encoded episode video'
        with av.open(str(video_files[0])) as container:
            decoded = list(container.decode(video=0))
        assert len(decoded) == result.num_frames


class TestEmbodimentCli:
    """Verify --embodiment parsing and validation."""

    def test_embodiment_defaults_to_unitree_g1(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot', '--rosbags-dir', 'bags_dir',
            '--output-dir', 'output_dir', '--task', 'task',
        ])
        args = mcap_to_lerobot.parse_args()
        assert args.embodiment == 'unitree_g1'

    def test_embodiment_valid_value_parses(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot', '--rosbags-dir', 'bags_dir',
            '--output-dir', 'output_dir', '--task', 'task',
            '--embodiment', 'unitree_h2_sharpa_wave',
        ])
        args = mcap_to_lerobot.parse_args()
        assert args.embodiment == 'unitree_h2_sharpa_wave'

    def test_embodiment_unknown_value_errors_listing_valid_names(
        self, monkeypatch, capsys,
    ):
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot', '--rosbags-dir', 'bags_dir',
            '--output-dir', 'output_dir', '--task', 'task',
            '--embodiment', 'not_a_real_embodiment',
        ])
        with pytest.raises(SystemExit):
            mcap_to_lerobot.parse_args()
        stderr = capsys.readouterr().err
        assert 'not_a_real_embodiment' in stderr
        for name in available_embodiments():
            assert name in stderr


class TestCodecCli:
    """Verify the --nvenc/--nvdec CLI flags parse and wire through main()."""

    def test_nvenc_nvdec_default_to_auto(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot', '--rosbags-dir', 'bags_dir',
            '--output-dir', 'output_dir', '--task', 'task',
        ])
        args = mcap_to_lerobot.parse_args()
        assert args.nvenc == 'auto'
        assert args.nvdec == 'auto'
        assert args.num_workers == os.cpu_count()

    def test_num_workers_nvenc_nvdec_parse_explicit_values(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot', '--rosbags-dir', 'bags_dir',
            '--output-dir', 'output_dir', '--task', 'task',
            '--num-workers', '3', '--nvenc', 'true', '--nvdec', 'false',
        ])
        args = mcap_to_lerobot.parse_args()
        assert args.num_workers == 3
        assert args.nvenc == 'true'
        assert args.nvdec == 'false'

    def test_nvenc_true_errors_when_hardware_unavailable(
        self, monkeypatch, tmp_path,
    ):
        """--nvenc true on a machine without the codec must fail clearly,
        before any bag is opened -- simulate "no GPU" by stubbing
        probe_hw_codecs regardless of what this machine actually has.
        """
        monkeypatch.setattr(
            mcap_to_lerobot, 'probe_hw_codecs',
            lambda: HwCodecCaps(nvdec=False, nvenc=False))
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot',
            '--rosbags-dir', str(tmp_path / 'bags'),
            '--output-dir', str(tmp_path / 'output'),
            '--task', 'task', '--nvenc', 'true',
        ])

        with pytest.raises(RuntimeError, match='nvenc'):
            mcap_to_lerobot.main()

    def test_nvdec_true_errors_when_hardware_unavailable(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            mcap_to_lerobot, 'probe_hw_codecs',
            lambda: HwCodecCaps(nvdec=False, nvenc=False))
        monkeypatch.setattr(sys, 'argv', [
            'mcap-to-lerobot',
            '--rosbags-dir', str(tmp_path / 'bags'),
            '--output-dir', str(tmp_path / 'output'),
            '--task', 'task', '--nvdec', 'true',
        ])

        with pytest.raises(RuntimeError, match='nvdec'):
            mcap_to_lerobot.main()


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


class TestH2SharpaWaveConversion:
    """Verify H2 goes through the same config-driven path as G1."""

    def test_required_embodiments_are_packaged_and_immutable(self):
        required = {'unitree_g1', 'unitree_h2_sharpa_wave'}
        assert required <= set(available_embodiments())

        embodiment = load_embodiment('unitree_h2_sharpa_wave')
        with pytest.raises(AttributeError):
            embodiment.action_joints.append('unexpected_joint')
        with pytest.raises(TypeError):
            embodiment.action_modalities['unexpected'] = (0, 1)

    def test_g1_public_defaults_are_unchanged(self):
        parameters = inspect.signature(create_lerobot_dataset).parameters
        assert 'rosbags_dirs' in parameters
        assert parameters['robot_type'].default == 'unitree_g1'

    def test_merges_finger_actions_and_reloads_dataset(self, tmp_path, monkeypatch):
        embodiment = load_embodiment('unitree_h2_sharpa_wave')
        stamp = structures.Timestamp(seconds=1, nanoseconds=0)
        camera_topic = '/camera/color/image_compressed'
        arm_names = embodiment.action_joints[:14]
        finger_names = embodiment.action_joints[14:]

        state = structures.JointState(
            position=list(range(58)), velocity=[], effort=[],
            names=embodiment.state_joints, timestamp=stamp)
        arms = structures.JointState(
            position=list(range(100, 114)), velocity=[],
            effort=list(range(300, 314)),
            names=arm_names, timestamp=stamp)
        fingers = structures.JointState(
            position=list(range(200, 244)), velocity=[],
            effort=list(range(400, 444)),
            names=finger_names, timestamp=stamp)

        class SyntheticH2Reader:
            def __init__(self, _bag_path, max_staleness_s=None):
                self.max_staleness_s = max_staleness_s
                self.robot_joint_states = [state]
                self.joint_commands = [arms]
                self.joint_state_topics = {
                    '/joint_states': [state],
                    '/xr_teleop/finger_joints': [fingers],
                }
                self.compressed_camera_image_topics = [camera_topic]
                self.compressed_camera_image_timestamps = {camera_topic: [stamp]}
                self.navigate_commands = []
                self.base_height_commands = []
                self.record_data = [structures.RecordData(topic_stamps=[
                    structures.TopicStamp('joint_states', stamp, 0.0),
                    structures.TopicStamp('applied_joint_commands', stamp, 0.0),
                    structures.TopicStamp(camera_topic, stamp, 0.0),
                ])]

            def construct_dataset(self):
                pass

            def decode_video_frames(self, topic):
                assert topic == camera_topic
                return [np.zeros((64, 64, 3), dtype=np.uint8)]

            def joint_messages_for_topic(self, topic):
                if topic == '/applied_joint_commands':
                    return self.joint_commands
                return self.joint_state_topics.get(topic, [])

        bags_dir = tmp_path / 'bags'
        episode_dir = bags_dir / 'episode_000'
        episode_dir.mkdir(parents=True)
        (episode_dir / 'metadata.yaml').write_text('', encoding='utf-8')
        monkeypatch.setattr(mcap_to_lerobot, 'RosbagReader', SyntheticH2Reader)

        output_dir = tmp_path / 'dataset'
        create_lerobot_dataset(
            rosbags_dirs=[str(bags_dir)],
            output_dir=str(output_dir),
            task='synthetic h2 action merge',
            fps=30,
            robot_type='unitree_h2_sharpa_wave',
            embodiment_name='unitree_h2_sharpa_wave',
        )

        dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(output_dir))
        assert len(dataset) == 1
        # Tabular sample from the reloaded hf_dataset (see the note in
        # TestMcapToLerobotConversion.test_dataset_reloadable): avoids decoding
        # the episode video, which is not hermetic under Bazel.
        sample = dataset.hf_dataset[0]
        assert np.asarray(sample['observation.state']).shape == (58,)
        assert np.asarray(sample['action']).tolist() == list(range(100, 114)) + list(
            range(200, 244))
        # Effort is split per action_modalities group (arms carry real
        # torques from /applied_joint_commands; fingers are zero-filled —
        # see left_hand_effort/right_hand_effort's value: zero in the
        # embodiment YAML).
        assert np.asarray(sample['action.effort_left_arm']).tolist() == list(
            range(300, 307))
        assert np.asarray(sample['action.effort_right_arm']).tolist() == list(
            range(307, 314))
        assert np.asarray(sample['action.effort_left_hand']).tolist() == [0.0] * 22
        assert np.asarray(sample['action.effort_right_hand']).tolist() == [0.0] * 22
        assert np.asarray(sample['action.navigate_command']).shape == (3,)
        assert np.allclose(
            np.asarray(sample['action.base_height_command']), [0.72])

    def test_drops_action_when_finger_sample_is_stale(self):
        embodiment = load_embodiment('unitree_h2_sharpa_wave')
        camera_topic = '/camera/color/image_compressed'
        command_stamp = structures.Timestamp(seconds=2, nanoseconds=0)
        finger_stamp = structures.Timestamp(seconds=1, nanoseconds=0)
        state = structures.JointState(
            position=list(range(58)), velocity=[], effort=[],
            names=embodiment.state_joints, timestamp=command_stamp)
        arms = structures.JointState(
            position=list(range(100, 114)), velocity=[], effort=[],
            names=embodiment.action_joints[:14], timestamp=command_stamp)
        fingers = structures.JointState(
            position=list(range(200, 244)), velocity=[], effort=[],
            names=embodiment.action_joints[14:], timestamp=finger_stamp)

        reader = SimpleNamespace(
            compressed_camera_image_topics=[camera_topic],
            robot_joint_states=[state],
            joint_commands=[arms],
            compressed_camera_image_timestamps={camera_topic: [command_stamp]},
            navigate_commands=[],
            base_height_commands=[],
            max_staleness_s=0.1,
            record_data=[structures.RecordData(topic_stamps=[
                structures.TopicStamp('joint_states', command_stamp, 0.0),
                structures.TopicStamp(
                    'applied_joint_commands', command_stamp, 0.0),
                structures.TopicStamp(camera_topic, command_stamp, 0.0),
            ])],
            joint_messages_for_topic=lambda topic: (
                [fingers] if topic == '/xr_teleop/finger_joints' else []),
        )

        trajectory = mcap_to_lerobot.build_synchronized_trajectory(
            reader,
            {camera_topic: [np.zeros((64, 64, 3), dtype=np.uint8)]},
            embodiment,
        )
        assert trajectory == []

    def test_causal_zoh_source_is_indexed_once(self):
        embodiment = load_embodiment('unitree_h2_sharpa_wave')
        camera_topic = '/camera/color/image_compressed'
        stamps = [
            structures.Timestamp(seconds=1, nanoseconds=0),
            structures.Timestamp(seconds=2, nanoseconds=0),
        ]
        states = [
            structures.JointState(
                position=list(range(58)), velocity=[], effort=[],
                names=embodiment.state_joints, timestamp=stamp)
            for stamp in stamps
        ]
        arms = [
            structures.JointState(
                position=list(range(100, 114)), velocity=[], effort=[],
                names=embodiment.action_joints[:14], timestamp=stamp)
            for stamp in stamps
        ]
        fingers = [
            structures.JointState(
                position=list(range(200 + 100 * index, 244 + 100 * index)),
                velocity=[], effort=[], names=embodiment.action_joints[14:],
                timestamp=stamp)
            for index, stamp in enumerate(stamps)
        ]

        source_reads = 0

        def joint_messages_for_topic(topic):
            nonlocal source_reads
            assert topic == '/xr_teleop/finger_joints'
            source_reads += 1
            return list(reversed(fingers))

        reader = SimpleNamespace(
            compressed_camera_image_topics=[camera_topic],
            robot_joint_states=states,
            joint_commands=arms,
            compressed_camera_image_timestamps={camera_topic: stamps},
            navigate_commands=[],
            base_height_commands=[],
            max_staleness_s=1.0,
            record_data=[
                structures.RecordData(topic_stamps=[
                    structures.TopicStamp('joint_states', stamp, 0.0),
                    structures.TopicStamp(
                        'applied_joint_commands', stamp, 0.0),
                    structures.TopicStamp(camera_topic, stamp, 0.0),
                ])
                for stamp in stamps
            ],
            joint_messages_for_topic=joint_messages_for_topic,
        )

        trajectory = mcap_to_lerobot.build_synchronized_trajectory(
            reader,
            {
                camera_topic: [
                    np.zeros((64, 64, 3), dtype=np.uint8) for _ in stamps
                ],
            },
            embodiment,
        )

        assert len(trajectory) == 2
        assert source_reads == 1
        assert trajectory[0].action.data[14:] == list(range(200, 244))
        assert trajectory[1].action.data[14:] == list(range(300, 344))


class TestRunConversionDeterminism:
    """run_conversion must produce the same dataset at any worker count.

    Builds the same 2-bag input twice (both copies of the bundled test bag,
    so both runs process bit-identical source data) and converts it once
    with num_workers=1 and once with num_workers=4. The two outputs must
    agree on episode count/ordering, feature schema, and per-feature
    stats -- proving the ProcessPoolExecutor driver's MAP/REDUCE split
    doesn't depend on scheduling or worker count for its result.
    """

    STATS_TOLERANCE = 1e-6

    @staticmethod
    @pytest.fixture(scope='class')
    def two_bag_dirs(tmp_path_factory):
        bags_dir = tmp_path_factory.mktemp('run_conversion_src') / 'bags'
        for name in ('episode_000', 'episode_001'):
            shutil.copytree(TEST_BAG_DIR, bags_dir / name)
        return bags_dir

    @staticmethod
    def _run(tmp_path_factory, bags_dir, tmp_name, num_workers):
        embodiment = load_embodiment(TEST_EMBODIMENT)
        output_dir = tmp_path_factory.mktemp(tmp_name) / 'dataset'
        summary = run_conversion(
            rosbags_dirs=[str(bags_dir)],
            output_dir=str(output_dir),
            embodiment=embodiment,
            task=EXPECTED_TASK,
            fps=TEST_FPS,
            num_workers=num_workers,
            codec_opts=software_codec_opts(),
        )
        assert summary.converted == 2
        assert summary.skipped == 0
        assert summary.skipped_paths == []
        return output_dir

    @staticmethod
    @pytest.fixture(scope='class')
    def serial_output(tmp_path_factory, two_bag_dirs):
        return TestRunConversionDeterminism._run(
            tmp_path_factory, two_bag_dirs, 'w1', num_workers=1)

    @staticmethod
    @pytest.fixture(scope='class')
    def parallel_output(tmp_path_factory, two_bag_dirs):
        return TestRunConversionDeterminism._run(
            tmp_path_factory, two_bag_dirs, 'w4', num_workers=4)

    @staticmethod
    def _episode_indices(dataset_dir):
        lines = (
            dataset_dir / 'meta' / 'episodes.jsonl').read_text().splitlines()
        return sorted(json.loads(line)['episode_index'] for line in lines if line)

    def test_parallel_equals_serial(self, serial_output, parallel_output):
        serial_info = json.loads(
            (serial_output / 'meta' / 'info.json').read_text())
        parallel_info = json.loads(
            (parallel_output / 'meta' / 'info.json').read_text())

        assert parallel_info['total_episodes'] == serial_info['total_episodes'] == 2
        assert parallel_info['total_frames'] == serial_info['total_frames']
        assert parallel_info['total_chunks'] == serial_info['total_chunks']
        assert parallel_info['fps'] == serial_info['fps']
        assert parallel_info['features'] == serial_info['features']

        assert self._episode_indices(serial_output) == [0, 1]
        assert self._episode_indices(parallel_output) == [0, 1]

        serial_ds = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(serial_output))
        parallel_ds = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(parallel_output))

        assert len(serial_ds) == len(parallel_ds)
        assert set(serial_ds.meta.stats.keys()) == set(
            parallel_ds.meta.stats.keys())
        for key in serial_ds.meta.stats:
            for stat_name in ('min', 'max', 'mean', 'std', 'count'):
                serial_val = np.asarray(serial_ds.meta.stats[key][stat_name])
                parallel_val = np.asarray(
                    parallel_ds.meta.stats[key][stat_name])
                assert serial_val.shape == parallel_val.shape, (
                    f'{key}.{stat_name} shape mismatch: '
                    f'{serial_val.shape} vs {parallel_val.shape}')
                assert np.allclose(
                    serial_val, parallel_val, atol=self.STATS_TOLERANCE), (
                    f'{key}.{stat_name} differs beyond tolerance: '
                    f'{serial_val} vs {parallel_val}')


class TestRunConversionFailureHandling:
    """A bad bag must be counted, not silently dropped or fatal."""

    def test_one_bad_bag_is_counted_and_good_bags_still_convert(
        self, tmp_path,
    ):
        embodiment = load_embodiment(TEST_EMBODIMENT)
        bags_dir = tmp_path / 'bags'
        shutil.copytree(TEST_BAG_DIR, bags_dir / 'episode_000')
        broken_dir = bags_dir / 'episode_001'
        broken_dir.mkdir()
        (broken_dir / 'metadata.yaml').write_text('', encoding='utf-8')

        output_dir = tmp_path / 'dataset'
        summary = run_conversion(
            rosbags_dirs=[str(bags_dir)],
            output_dir=str(output_dir),
            embodiment=embodiment,
            task=EXPECTED_TASK,
            fps=TEST_FPS,
            num_workers=2,
            codec_opts=software_codec_opts(),
        )

        assert summary.converted == 1
        assert summary.skipped == 1
        assert summary.skipped_paths == [str(broken_dir)]

        info = json.loads((output_dir / 'meta' / 'info.json').read_text())
        assert info['total_episodes'] == 1


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
