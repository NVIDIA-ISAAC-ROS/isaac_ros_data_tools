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
"""Merge single-episode LeRobot shards into one final dataset (the REDUCE).

convert_one_bag (the MAP unit) writes each MCAP bag as a standalone
single-episode LeRobotDataset rooted at its own shard directory, with no
shared state between shards. merge_shards concatenates those shards, in
the given order, into one final on-disk LeRobotDataset at output_dir:
episodes are renumbered globally, index/task_index columns in each
episode's parquet are rebased, video files are relocated into the target
chunk layout, tasks are deduplicated, and per-episode stats are combined.

LeRobot 0.3.2 ships no dataset-merge utility, so this module implements a
focused merge directly against the documented v2.1 on-disk layout, reusing
LeRobotDatasetMetadata for the bookkeeping it already gets right (frame
and chunk counters, writing meta/*.jsonl, and combining stats via
lerobot.datasets.compute_stats.aggregate_stats, the same aggregation
save_episode performs incrementally in the sequential path). No video
frames or full parquet columns are re-read; only the three per-frame
bookkeeping columns (index, episode_index, task_index) are rewritten, and
their stats are recomputed from the rewritten arrays via
lerobot.datasets.compute_stats.get_feature_stats.
"""

from __future__ import annotations

import logging
import shutil

from isaac_ros_mcap_lerobot_converter.embodiments import EmbodimentConfig
from isaac_ros_mcap_lerobot_converter.mcap_to_lerobot import build_features
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from lerobot.common.datasets.compute_stats import get_feature_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

logger = logging.getLogger(__name__)

_REPO_ID = 'isaac_ros_mcap_lerobot'

# The three per-frame bookkeeping columns whose values depend on an
# episode's position in the merged dataset and must be rebased/remapped
# rather than copied verbatim from the shard.
_POSITIONAL_COLUMNS = ('index', 'episode_index', 'task_index')


def merge_shards(
    shard_dirs: list[str],
    output_dir: str,
    embodiment: EmbodimentConfig,
    task: str,
) -> None:
    """Concatenate single-episode shards into one final LeRobot dataset.

    Args:
        shard_dirs: Shard root directories, each written by convert_one_bag.
            Episodes are renumbered globally in this exact order --
            callers that need deterministic output must pass a pre-sorted
            list.
        output_dir: Root directory for the merged dataset. Must not
            already exist.
        embodiment: The embodiment config the shards were converted with;
            used to (re)derive the merged dataset's feature schema via
            build_features so meta/info.json doesn't just trust whatever a
            shard happened to write.
        task: Unused -- each shard already carries its own task string(s)
            in meta/tasks.jsonl, so the merge is driven by that per-shard
            metadata. Accepted for interface symmetry with convert_one_bag.
    """
    del task  # merge is metadata-driven, not label-driven (see docstring).

    if not shard_dirs:
        raise ValueError('merge_shards requires at least one shard directory')

    shard_metas = [
        LeRobotDatasetMetadata(repo_id=_REPO_ID, root=shard_dir)
        for shard_dir in shard_dirs
    ]

    first_info = shard_metas[0].info
    features = build_features(embodiment)
    for camera in embodiment.cameras:
        key = f'observation.images.{camera.name}'
        first_shape = tuple(first_info['features'][key]['shape'])
        # The merged metadata claims first_shape for every episode, so any
        # shard recorded at a different camera resolution/config would yield
        # a dataset whose later episodes don't match their declared shape.
        for shard_dir, shard_meta in zip(shard_dirs[1:], shard_metas[1:]):
            shard_shape = tuple(shard_meta.info['features'][key]['shape'])
            if shard_shape != first_shape:
                raise ValueError(
                    f'camera {key!r} shape mismatch across shards: '
                    f'{shard_dirs[0]} has {first_shape} but {shard_dir} has '
                    f'{shard_shape}; shards recorded at different camera '
                    'resolutions cannot be merged into one dataset')
        features[key]['shape'] = first_shape

    out_meta = LeRobotDatasetMetadata.create(
        repo_id=_REPO_ID,
        fps=first_info['fps'],
        features=features,
        robot_type=first_info['robot_type'],
        root=output_dir,
    )

    global_index = 0
    frame_offset = 0
    for shard_dir, shard_meta in zip(shard_dirs, shard_metas):
        local_to_global_task = _register_tasks(shard_meta, out_meta)

        for local_idx in sorted(shard_meta.episodes.keys()):
            episode = shard_meta.episodes[local_idx]
            episode_length = episode['length']
            episode_tasks = episode['tasks']
            episode_stats = {
                key: dict(value)
                for key, value in shard_meta.episodes_stats[local_idx].items()
            }

            _relocate_parquet(
                shard_meta, local_idx, out_meta, global_index, frame_offset,
                local_to_global_task, episode_stats, episode_length)
            _relocate_videos(shard_meta, local_idx, out_meta, global_index)

            if global_index == 0 and out_meta.video_keys:
                # Reads codec/height/width/etc. from the just-relocated
                # episode-0 video via ffprobe; mirrors what
                # LeRobotDataset.encode_episode_videos does after encoding
                # the first episode.
                out_meta.update_video_info()

            out_meta.save_episode(
                global_index, episode_length, episode_tasks, episode_stats)

            frame_offset += episode_length
            global_index += 1

    if global_index == 0:
        raise ValueError(f'No episodes found in any shard: {shard_dirs}')

    logger.info(
        'Merged %d shard(s) into %d episode(s) (%d frames) at %s',
        len(shard_dirs), global_index, frame_offset, output_dir)


def _register_tasks(
    shard_meta: LeRobotDatasetMetadata,
    out_meta: LeRobotDatasetMetadata,
) -> dict[int, int]:
    """Dedup-register a shard's tasks into out_meta.

    Returns the shard's local task_index -> merged task_index mapping,
    needed to rewrite each episode's task_index parquet column.
    """
    mapping = {}
    for local_task_index, task_str in shard_meta.tasks.items():
        global_task_index = out_meta.get_task_index(task_str)
        if global_task_index is None:
            out_meta.add_task(task_str)
            global_task_index = out_meta.get_task_index(task_str)
        mapping[local_task_index] = global_task_index
    return mapping


def _relocate_parquet(
    shard_meta: LeRobotDatasetMetadata,
    local_idx: int,
    out_meta: LeRobotDatasetMetadata,
    global_idx: int,
    frame_offset: int,
    local_to_global_task: dict[int, int],
    episode_stats: dict[str, dict],
    episode_length: int,
) -> None:
    """Copy one episode's parquet into the merged layout, rebasing
    index/episode_index/task_index, and update episode_stats in place for
    those three columns to match the rewritten values (every other
    feature's stats are unaffected by renumbering and are reused as-is --
    no frame re-reads).
    """
    src = shard_meta.root / shard_meta.get_data_file_path(local_idx)
    dest = out_meta.root / out_meta.get_data_file_path(global_idx)
    dest.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(src)
    num_rows = table.num_rows
    # Not an assert: a silent divergence here would offset every later
    # episode's global index column, and asserts are stripped under -O.
    if episode_length != num_rows:
        raise ValueError(
            f'episode {local_idx} in shard {shard_meta.root}: '
            f'shard metadata reports length {episode_length} but parquet '
            f'{src} has {num_rows} rows; a silent divergence here would '
            "offset every later episode's global index column")

    index_type = table.schema.field('index').type
    episode_index_type = table.schema.field('episode_index').type
    task_index_type = table.schema.field('task_index').type

    new_index = np.arange(frame_offset, frame_offset + num_rows)
    new_episode_index = np.full(num_rows, global_idx)

    old_task_index = table.column('task_index').to_numpy()
    new_task_index = np.array(
        [local_to_global_task[int(v)] for v in old_task_index])

    for name, values, arrow_type in (
        ('index', new_index, index_type),
        ('episode_index', new_episode_index, episode_index_type),
        ('task_index', new_task_index, task_index_type),
    ):
        field_index = table.schema.get_field_index(name)
        table = table.set_column(
            field_index, name, pa.array(values, type=arrow_type))
        episode_stats[name] = get_feature_stats(values, axis=0, keepdims=True)

    pq.write_table(table, dest)


def _relocate_videos(
    shard_meta: LeRobotDatasetMetadata,
    local_idx: int,
    out_meta: LeRobotDatasetMetadata,
    global_idx: int,
) -> None:
    """Copy one episode's per-camera video files into the merged layout.

    Copies rather than moves -- shards may be inspected by tests/callers
    after a merge, and a partial failure mid-copy must not leave the
    source shard unusable.
    """
    for video_key in out_meta.video_keys:
        src = shard_meta.root / shard_meta.get_video_file_path(
            local_idx, video_key)
        dest = out_meta.root / out_meta.get_video_file_path(
            global_idx, video_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
