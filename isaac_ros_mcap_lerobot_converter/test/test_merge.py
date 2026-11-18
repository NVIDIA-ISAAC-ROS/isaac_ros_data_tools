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
"""Equivalence test for merge_shards.

Builds the same 2-episode LeRobot dataset two ways:

(a) sequentially, via the existing create_lerobot_dataset over two bag
    directories in one process, and
(b) as the parallel MAP/REDUCE split — convert_one_bag on each bag into
    its own single-episode shard, followed by merge_shards — and asserts
    the two outputs are equivalent: same episode count, contiguous
    episode_index, identical feature schema, and per-feature stats equal
    within a small numerical tolerance.

The two duplicate episode directories are built from the single bundled
test bag (test/data/episode_001), so both pipelines process bit-identical
source data; any observed stats difference is attributable to the merge
logic, not to sampling differences. fps=30 is >5% off the bag's detected
native rate (~27.8 Hz), so sequential_dataset resamples via causal ZOH
(create_lerobot_dataset's own tolerance check); merged_dataset calls
convert_one_bag directly, so it passes resample_to_rate=TEST_FPS
explicitly to take the same resample branch and stay comparable.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from isaac_ros_mcap_lerobot_converter.codec import software_codec_opts
from isaac_ros_mcap_lerobot_converter.embodiments import load_embodiment
from isaac_ros_mcap_lerobot_converter.mcap_to_lerobot import (
    convert_one_bag,
    create_lerobot_dataset,
)
from isaac_ros_mcap_lerobot_converter.merge import merge_shards
import numpy as np
import pytest

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

TEST_BAG_DIR = Path(__file__).parent / 'data' / 'episode_001'
TEST_EMBODIMENT = 'unitree_g1'
TEST_FPS = 30
TEST_TASK = 'grasp with right hand and close left fingers'
STATS_TOLERANCE = 1e-6


def _make_two_bag_dirs(dest_root: Path) -> Path:
    """Duplicate the bundled bag into two sibling episode directories."""
    bags_dir = dest_root / 'bags'
    for name in ('episode_000', 'episode_001'):
        shutil.copytree(TEST_BAG_DIR, bags_dir / name)
    return bags_dir


@pytest.fixture(scope='module')
def sequential_dataset(tmp_path_factory):
    """The existing single-process path: two bags merged by
    create_lerobot_dataset in one call.
    """
    bags_dir = _make_two_bag_dirs(tmp_path_factory.mktemp('seq_src'))
    output_dir = tmp_path_factory.mktemp('seq_out') / 'dataset'
    create_lerobot_dataset(
        rosbags_dirs=[str(bags_dir)],
        output_dir=str(output_dir),
        task=TEST_TASK,
        fps=TEST_FPS,
        robot_type=TEST_EMBODIMENT,
        embodiment_name=TEST_EMBODIMENT,
    )
    return output_dir


@dataclass
class MergeResult:
    """The merged dataset plus the shard dirs it was built from, so tests
    can check the merge copied (rather than moved) shard video files.
    """

    output_dir: Path
    shard_dirs: list[str]


@pytest.fixture(scope='module')
def merged_dataset(tmp_path_factory) -> MergeResult:
    """The MAP/REDUCE path: convert_one_bag per bag, then merge_shards.

    Passes resample_to_rate=TEST_FPS explicitly -- this test calls
    convert_one_bag directly rather than through run_conversion, so it has
    to make the same resample-vs-tolerance decision itself, to stay
    comparable with sequential_dataset.
    """
    bags_dir = _make_two_bag_dirs(tmp_path_factory.mktemp('par_src'))
    embodiment = load_embodiment(TEST_EMBODIMENT)

    shard_dirs = []
    for i, bag_dir in enumerate(sorted(bags_dir.iterdir())):
        shard_dir = tmp_path_factory.mktemp(f'shard_{i}') / 'shard'
        result = convert_one_bag(
            str(bag_dir), embodiment, task=TEST_TASK, fps=TEST_FPS,
            shard_dir=str(shard_dir), codec_opts=software_codec_opts(),
            resample_to_rate=TEST_FPS)
        assert result.ok, result.error
        shard_dirs.append(str(shard_dir))

    output_dir = tmp_path_factory.mktemp('merged_out') / 'dataset'
    merge_shards(shard_dirs, str(output_dir), embodiment, TEST_TASK)
    return MergeResult(output_dir=output_dir, shard_dirs=shard_dirs)


def _read_info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / 'meta' / 'info.json').read_text())


def _read_episodes(dataset_dir: Path) -> list[dict]:
    lines = (dataset_dir / 'meta' / 'episodes.jsonl').read_text().splitlines()
    return [json.loads(line) for line in lines if line]


class TestMergeShardsEquivalence:
    """Merging 2 shards must equal the existing sequential conversion."""

    def test_episode_count_and_contiguous_index(self, merged_dataset):
        info = _read_info(merged_dataset.output_dir)
        assert info['total_episodes'] == 2

        episodes = _read_episodes(merged_dataset.output_dir)
        indices = sorted(e['episode_index'] for e in episodes)
        assert indices == [0, 1]

    def test_episode_videos_relocated_and_shards_untouched(
        self, merged_dataset,
    ):
        """Regression guard for the top merge risk: a bug that indexes the
        merged video path by the shard-local episode index (always 0, since
        every shard holds exactly one episode) instead of the global index
        would silently overwrite episode 0's video and never write episode
        1's, leaving episode_000001.mp4 missing. Both must exist here.

        The two shards are converted from the same source bag, so the
        merged videos may be byte-identical -- that alone doesn't prove
        anything about *paths*, so this also checks the source shard videos
        are still present and unchanged (copied, not moved/consumed).
        """
        video_key = f'observation.images.{self._camera_name()}'
        video_dir = (
            merged_dataset.output_dir / 'videos' / 'chunk-000' / video_key)
        video_0 = video_dir / 'episode_000000.mp4'
        video_1 = video_dir / 'episode_000001.mp4'

        assert video_0.is_file(), f'missing merged video: {video_0}'
        assert video_1.is_file(), f'missing merged video: {video_1}'
        assert video_0.stat().st_size > 0
        assert video_1.stat().st_size > 0

        for shard_dir in merged_dataset.shard_dirs:
            shard_video = (
                Path(shard_dir) / 'videos' / 'chunk-000' / video_key
                / 'episode_000000.mp4')
            assert shard_video.is_file(), (
                f'shard video missing after merge (moved instead of '
                f'copied?): {shard_video}')
            assert shard_video.stat().st_size > 0

    @staticmethod
    def _camera_name() -> str:
        embodiment = load_embodiment(TEST_EMBODIMENT)
        return embodiment.cameras[0].name

    def test_feature_schema_matches_sequential(
        self, sequential_dataset, merged_dataset,
    ):
        seq_info = _read_info(sequential_dataset)
        merged_info = _read_info(merged_dataset.output_dir)

        assert merged_info['features'] == seq_info['features']
        assert merged_info['total_episodes'] == seq_info['total_episodes']
        assert merged_info['total_frames'] == seq_info['total_frames']
        assert merged_info['total_chunks'] == seq_info['total_chunks']
        assert merged_info['total_videos'] == seq_info['total_videos']
        assert merged_info['fps'] == seq_info['fps']

    def test_stats_equal_within_tolerance(
        self, sequential_dataset, merged_dataset,
    ):
        seq_dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(sequential_dataset))
        merged_ds = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot',
            root=str(merged_dataset.output_dir))

        assert set(seq_dataset.meta.stats.keys()) == set(
            merged_ds.meta.stats.keys())
        for key in seq_dataset.meta.stats:
            for stat_name in ('min', 'max', 'mean', 'std', 'count'):
                seq_val = np.asarray(seq_dataset.meta.stats[key][stat_name])
                merged_val = np.asarray(merged_ds.meta.stats[key][stat_name])
                assert seq_val.shape == merged_val.shape, (
                    f'{key}.{stat_name} shape mismatch: '
                    f'{seq_val.shape} vs {merged_val.shape}')
                assert np.allclose(
                    seq_val, merged_val, atol=STATS_TOLERANCE), (
                    f'{key}.{stat_name} differs beyond tolerance: '
                    f'{seq_val} vs {merged_val}')

    def test_dataset_reloadable_with_matching_frame_count(
        self, sequential_dataset, merged_dataset,
    ):
        seq_dataset = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot', root=str(sequential_dataset))
        merged_ds = LeRobotDataset(
            repo_id='isaac_ros_mcap_lerobot',
            root=str(merged_dataset.output_dir))

        assert merged_ds.meta.total_episodes == 2
        assert len(merged_ds) == merged_ds.meta.total_frames
        assert len(merged_ds) == len(seq_dataset)

        seq_sample = seq_dataset[0]
        merged_sample = merged_ds[0]
        assert np.allclose(
            np.asarray(seq_sample['observation.state']),
            np.asarray(merged_sample['observation.state']))
        assert np.allclose(
            np.asarray(seq_sample['action']),
            np.asarray(merged_sample['action']))

        # Episode 0 is a decoy: its shard-local index (0) equals its global
        # index (0), so a merge bug that used the local index everywhere
        # would still pass every assertion above. Episode 1 is the only one
        # that requires actual renumbering (shard-local 0 -> global 1), for
        # both its index column and its video path -- sample a frame from
        # it here so that renumbering is genuinely exercised.
        ep1_start = merged_ds.meta.episodes[0]['length']
        assert seq_dataset.meta.episodes[0]['length'] == ep1_start, (
            'episode 0 length differs between pipelines; the shared '
            f'index {ep1_start} would not point at episode 1 in both')

        seq_ep1_sample = seq_dataset[ep1_start]
        merged_ep1_sample = merged_ds[ep1_start]
        assert int(seq_ep1_sample['episode_index']) == 1
        assert int(merged_ep1_sample['episode_index']) == 1

        assert np.allclose(
            np.asarray(seq_ep1_sample['observation.state']),
            np.asarray(merged_ep1_sample['observation.state']))
        assert np.allclose(
            np.asarray(seq_ep1_sample['action']),
            np.asarray(merged_ep1_sample['action']))

        video_key = f'observation.images.{self._camera_name()}'
        assert np.array_equal(
            np.asarray(seq_ep1_sample[video_key]),
            np.asarray(merged_ep1_sample[video_key])), (
            'episode-1 video frame differs between pipelines -- this '
            'decodes the merged dataset\'s renumbered '
            f'{video_key}/episode_000001.mp4, so a mispathed/overwritten '
            'video would surface here')

    def test_tasks_deduplicated(self, merged_dataset):
        tasks_lines = (
            merged_dataset.output_dir / 'meta'
            / 'tasks.jsonl').read_text().splitlines()
        tasks = [json.loads(line) for line in tasks_lines if line]
        # Both episodes share the same task label -> exactly one task entry.
        assert len(tasks) == 1
        assert tasks[0]['task'] == TEST_TASK


class TestMergeShardsErrors:
    def test_empty_shard_list_raises(self, tmp_path):
        embodiment = load_embodiment(TEST_EMBODIMENT)
        with pytest.raises(ValueError):
            merge_shards(
                [], str(tmp_path / 'out'), embodiment, TEST_TASK)
