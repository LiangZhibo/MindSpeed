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

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule


def test_replay_trims_cache_before_entry():
    session = OffloadSession(OffloadConfig())
    session.executor = ReplayExecutor(session.residency_manager, ResidencySchedule(), [])
    session.mode = "replay"

    with patch("mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache") as empty_cache:
        with session:
            empty_cache.assert_called_once_with()
        empty_cache.assert_called_once_with()
        assert session.last_trim_seconds >= 0.0
