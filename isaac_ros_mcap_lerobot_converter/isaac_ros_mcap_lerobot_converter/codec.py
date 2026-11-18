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
"""Hardware-codec selection for MCAP-to-LeRobot video decode/encode.

probe_hw_codecs() checks, once per process, whether this machine's
PyAV/ffmpeg build exposes the NVIDIA nvdec (h264_cuvid) decoder and nvenc
(h264_nvenc) encoder. resolve_codec_opts() turns that capability probe
plus the CLI's tri-state --nvdec/--nvenc flags (auto/true/false) into a
plain CodecOpts that decode/encode call sites act on without needing to
know about probing or CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass

import av


@dataclass(frozen=True)
class HwCodecCaps:
    """Whether this process's PyAV/ffmpeg build can use the NVIDIA codecs."""

    nvdec: bool
    nvenc: bool


@dataclass(frozen=True)
class CodecOpts:
    """Codec backend selection passed down to decode/encode call sites.

    *_required records whether the corresponding flag was passed as true
    (as opposed to resolved via auto) — decode/encode call sites use this
    to decide whether a runtime hardware failure (the capability probe
    succeeded, but the actual decode/encode call still failed, e.g. a busy
    or misbehaving GPU) should be a hard error (true) or fall back to
    software with a logged warning (auto).
    """

    use_nvdec: bool
    use_nvenc: bool
    nvdec_required: bool = False
    nvenc_required: bool = False


def _has_codec(name: str, mode: str) -> bool:
    try:
        av.codec.Codec(name, mode)
        return True
    except Exception:
        return False


def probe_hw_codecs() -> HwCodecCaps:
    """Probe whether nvdec/nvenc codecs are available via this PyAV build."""
    return HwCodecCaps(
        nvdec=_has_codec('h264_cuvid', 'r'),
        nvenc=_has_codec('h264_nvenc', 'w'),
    )


def software_codec_opts() -> CodecOpts:
    """Return codec options that force pure-software decode and encode."""
    return CodecOpts(use_nvdec=False, use_nvenc=False)


def _resolve_one(available: bool, mode: str, name: str) -> bool:
    if mode == 'false':
        return False
    if mode == 'true':
        if not available:
            raise RuntimeError(
                f'--{name} true requested but the {name} GPU codec is not '
                f'available in this PyAV/ffmpeg build')
        return True
    if mode == 'auto':
        return available  # use if available, else software fallback
    raise ValueError(f'--{name} must be one of auto/true/false, got {mode!r}')


def resolve_codec_opts(
    caps: HwCodecCaps,
    nvenc: str = 'auto',
    nvdec: str = 'auto',
) -> CodecOpts:
    """Resolve tri-state --nvenc/--nvdec flags against caps.

    Each of nvenc/nvdec is one of:
      * 'false' — always use the software path.
      * 'true' — require the GPU codec; raises RuntimeError if caps says
        it's unavailable (a hard error, not a silent fallback).
      * 'auto' (default) — use the GPU codec if available, else fall back
        to software.
    """
    return CodecOpts(
        use_nvdec=_resolve_one(caps.nvdec, nvdec, 'nvdec'),
        use_nvenc=_resolve_one(caps.nvenc, nvenc, 'nvenc'),
        nvdec_required=(nvdec == 'true'),
        nvenc_required=(nvenc == 'true'),
    )
