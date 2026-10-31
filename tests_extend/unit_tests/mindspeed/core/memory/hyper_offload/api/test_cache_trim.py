# Copyright 2026 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Tests for replay-time accelerator cache trimming."""

from unittest.mock import patch

import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession


def test_opt_in_cache_trim_runs_once_before_replay():
    session = OffloadSession(OffloadConfig(trim_cache_before_replay=True))
    with session:
        _ = torch.ones(2) + 1

    with patch("mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache") as empty_cache:
        with session:
            _ = torch.ones(2) + 1
        empty_cache.assert_called_once_with()
        assert session.last_trim_seconds >= 0.0


def test_cache_trim_is_disabled_by_default():
    session = OffloadSession(OffloadConfig())
    with session:
        _ = torch.ones(2) + 1

    with patch("mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache") as empty_cache:
        with session:
            _ = torch.ones(2) + 1
        empty_cache.assert_not_called()
