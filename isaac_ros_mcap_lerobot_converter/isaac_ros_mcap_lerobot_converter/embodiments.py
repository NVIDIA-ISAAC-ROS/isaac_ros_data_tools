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
"""Load embodiment definitions for MCAP-to-LeRobot conversion."""

from __future__ import annotations

import copyreg
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


def _rebuild_mappingproxy(mapping: dict) -> MappingProxyType:
    return MappingProxyType(mapping)


def _reduce_mappingproxy(proxy: MappingProxyType):
    return _rebuild_mappingproxy, (dict(proxy),)


# MappingProxyType's default reduction is broken -- pickle resolves it via
# __module__, which (mis)reports 'builtins' where 'mappingproxy' isn't
# actually exposed, so pickling raises PicklingError. EmbodimentConfig's
# modality maps and ActionSource.fields are all MappingProxyType, and
# EmbodimentConfig instances must cross a ProcessPoolExecutor boundary (the
# parallel driver submits embodiment to worker processes), so register an
# explicit reducer here. This is a process-global registration -- it
# affects every MappingProxyType in the process for its lifetime.
copyreg.pickle(MappingProxyType, _reduce_mappingproxy)


@dataclass(frozen=True)
class ActionSource:
    """Topic-level action source declared by an embodiment YAML file."""

    topic: str
    groups: tuple[str, ...]
    sync: str | None
    fields: Mapping[str, str]


@dataclass(frozen=True)
class CameraConfig:
    """Image topic exported by an embodiment."""

    name: str
    topic: str
    element_names: tuple[str, ...]


@dataclass(frozen=True)
class ModalityConfig:
    """Single observation or action modality declared in YAML."""

    name: str
    type: str
    source_topic: str | None
    element_names: tuple[str, ...]
    sync: str | None
    value: str | None
    default: tuple[float, ...] | None


@dataclass(frozen=True)
class EmbodimentConfig:
    """Static embodiment data used to build LeRobot features and modalities."""

    name: str
    state_joints: tuple[str, ...]
    action_joints: tuple[str, ...]
    state_modalities: Mapping[str, tuple[int, int]]
    action_modalities: Mapping[str, tuple[int, int]]
    action_sources: tuple[ActionSource, ...]
    cameras: tuple[CameraConfig, ...]
    observations: tuple[ModalityConfig, ...]
    actions: tuple[ModalityConfig, ...]


def _embodiment_dir():
    return resources.files(__package__).joinpath('embodiments')


def available_embodiments() -> list[str]:
    """Return names of packaged embodiment YAML files."""
    return sorted(path.stem for path in _embodiment_dir().iterdir()
                  if path.name.endswith('.yaml'))


def load_embodiment(name: str) -> EmbodimentConfig:
    """Load and validate a packaged embodiment YAML file."""
    return _load_embodiment_cached(name)


@lru_cache(maxsize=None)
def _load_embodiment_cached(name: str) -> EmbodimentConfig:
    path = _embodiment_dir().joinpath(f'{name}.yaml')
    if not path.is_file():
        raise ValueError(
            f"Unknown embodiment '{name}'. Available embodiments: "
            f"{available_embodiments()}"
        )
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Embodiment file {path} must contain a mapping')
    return _parse_embodiment(data, Path(path.name))


def _parse_embodiment(data: dict[str, Any], path: Path) -> EmbodimentConfig:
    name = _required_str(data, 'name', path)
    observations = _parse_modality_section(
        _required_mapping(data, 'observations', path), path)
    actions = _parse_modality_section(
        _required_mapping(data, 'actions', path), path)

    state_joints, state_modalities = _derive_joint_modalities(
        observations, kind='joint_position')
    action_joints, action_modalities = _derive_joint_modalities(
        actions, kind='joint_position')
    cameras = _derive_cameras(observations)
    action_sources = _derive_action_sources(actions)

    return EmbodimentConfig(
        name=name,
        state_joints=state_joints,
        action_joints=action_joints,
        state_modalities=state_modalities,
        action_modalities=action_modalities,
        action_sources=action_sources,
        cameras=cameras,
        observations=observations,
        actions=actions,
    )


def _parse_modality_section(
    raw_modalities: dict[str, Any],
    path: Path,
) -> tuple[ModalityConfig, ...]:
    modalities = []
    for name, raw_modality in raw_modalities.items():
        if not isinstance(name, str) or not isinstance(raw_modality, dict):
            raise ValueError(f'{path}: modality entries must be name: mapping')
        modality_type = _required_str(raw_modality, 'type', path)
        source_topic = raw_modality.get('source_topic')
        if source_topic is not None and not isinstance(source_topic, str):
            raise ValueError(f'{path}: {name}.source_topic must be a string')
        sync = raw_modality.get('sync')
        if sync is not None and not isinstance(sync, str):
            raise ValueError(f'{path}: {name}.sync must be a string')
        value = raw_modality.get('value')
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{path}: {name}.value must be a string')
        element_names = tuple(_required_str_list(
            raw_modality, 'element_names', path))
        default = raw_modality.get('default')
        if default is not None and (
            not isinstance(default, list)
            or not all(isinstance(v, (int, float)) for v in default)
        ):
            raise ValueError(f'{path}: {name}.default must be a number list')
        if default is not None and len(default) != len(element_names):
            raise ValueError(
                f'{path}: {name}.default must have {len(element_names)} values '
                'to match element_names')
        modalities.append(ModalityConfig(
            name=name,
            type=modality_type,
            source_topic=source_topic,
            element_names=element_names,
            sync=sync,
            value=value,
            default=tuple(float(v) for v in default) if default is not None else None,
        ))
    return tuple(modalities)


def _derive_joint_modalities(
    modalities: tuple[ModalityConfig, ...],
    kind: str,
) -> tuple[tuple[str, ...], Mapping[str, tuple[int, int]]]:
    joints = []
    ranges = {}
    for modality in modalities:
        if modality.type != kind:
            continue
        start = len(joints)
        joints.extend(modality.element_names)
        ranges[_modality_group_name(modality.name)] = (start, len(joints))
    return tuple(joints), MappingProxyType(ranges)


def _derive_cameras(
    observations: tuple[ModalityConfig, ...],
) -> tuple[CameraConfig, ...]:
    cameras = []
    for modality in observations:
        if modality.type != 'image':
            continue
        if modality.source_topic is None:
            raise ValueError(f'Image modality {modality.name} needs source_topic')
        cameras.append(CameraConfig(
            name=_modality_group_name(modality.name),
            topic=modality.source_topic,
            element_names=modality.element_names,
        ))
    return tuple(cameras)


def _derive_action_sources(
    actions: tuple[ModalityConfig, ...],
) -> tuple[ActionSource, ...]:
    grouped_sources: dict[tuple[str, str | None], list[str]] = {}
    for modality in actions:
        if modality.type != 'joint_position' or modality.source_topic is None:
            continue
        key = (modality.source_topic, modality.sync)
        grouped_sources.setdefault(key, []).append(_modality_group_name(modality.name))

    return tuple(
        ActionSource(
            topic=topic,
            groups=tuple(groups),
            sync=sync,
            fields=MappingProxyType({'position': 'action'}),
        )
        for (topic, sync), groups in grouped_sources.items()
    )


def _modality_group_name(name: str) -> str:
    return name.split('.')[-1]


def _required_mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: required key {key!r} must be a mapping')
    return value


def _required_str(data: dict[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f'{path}: required key {key!r} must be a non-empty string')
    return value


def _required_str_list(data: dict[str, Any], key: str, path: Path) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f'{path}: required key {key!r} must be a string list')
    return value
