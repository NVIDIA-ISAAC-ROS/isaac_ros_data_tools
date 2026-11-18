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
"""Unit tests for hardware-codec probing and tri-state resolution."""

from isaac_ros_mcap_lerobot_converter.codec import (
    HwCodecCaps,
    probe_hw_codecs,
    resolve_codec_opts,
)
import pytest


def test_probe_returns_bools():
    caps = probe_hw_codecs()
    assert isinstance(caps.nvdec, bool) and isinstance(caps.nvenc, bool)


def test_false_forces_cpu():
    opts = resolve_codec_opts(
        HwCodecCaps(nvdec=True, nvenc=True), nvenc='false', nvdec='false')
    assert opts.use_nvdec is False and opts.use_nvenc is False


def test_auto_uses_hw_when_available_and_falls_back_when_not():
    opts = resolve_codec_opts(
        HwCodecCaps(nvdec=True, nvenc=False), nvenc='auto', nvdec='auto')
    assert opts.use_nvdec is True and opts.use_nvenc is False  # auto falls back per-direction


def test_true_requires_hw_or_errors():
    # true = require the GPU codec; error (not silent CPU fallback) if unavailable.
    with pytest.raises(Exception):
        resolve_codec_opts(HwCodecCaps(nvdec=False, nvenc=False), nvenc='true')
    opts = resolve_codec_opts(
        HwCodecCaps(nvdec=True, nvenc=True), nvenc='true', nvdec='true')
    assert opts.use_nvenc is True and opts.use_nvdec is True
