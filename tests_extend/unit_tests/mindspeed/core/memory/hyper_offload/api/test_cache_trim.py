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

    original_reset = session.executor.reset

    def reset_after_trim():
        empty_cache.assert_called_once_with()
        original_reset()

    # Transfer-stream synchronization is independent of cache trimming and
    # must be mocked so that exiting the session also works on CPU-only hosts.
    with (
        patch("mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache") as empty_cache,
        patch.object(session.residency_manager, "wait_for_transfers") as wait_for_transfers,
        patch.object(session.executor, "reset", side_effect=reset_after_trim),
    ):
        with session:
            empty_cache.assert_called_once_with()
        empty_cache.assert_called_once_with()
        wait_for_transfers.assert_called_once_with()
        assert session.last_trim_seconds >= 0.0
