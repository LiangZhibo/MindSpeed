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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.api.session import ActivationDispatchMode, _empty_accelerator_cache
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule


def test_replay_trims_cache_before_entry():
    session = OffloadSession(OffloadConfig())
    session.executor = ReplayExecutor(session.residency_manager, ResidencySchedule(), [])
    session.mode = "replay"

    original_reset = session.executor.reset

    def reset_after_trim():
        empty_cache.assert_called_once_with(retain_physical=True)
        original_reset()

    # Transfer-stream synchronization is independent of cache trimming and
    # must be mocked so that exiting the session also works on CPU-only hosts.
    with (
        patch("mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache") as empty_cache,
        patch.object(session.residency_manager, "wait_for_transfers") as wait_for_transfers,
        patch.object(session.executor, "reset", side_effect=reset_after_trim),
    ):
        with session:
            empty_cache.assert_called_once_with(retain_physical=True)
        empty_cache.assert_called_once_with(retain_physical=True)
        wait_for_transfers.assert_called_once_with()
        assert session.last_trim_seconds >= 0.0


def test_replay_can_retain_inactive_cache_without_skipping_executor_reset():
    session = OffloadSession(OffloadConfig(trim_before_replay=False))
    session.executor = ReplayExecutor(session.residency_manager, ResidencySchedule(), [])
    session.mode = 'replay'
    session.last_trim_seconds = 123.0
    with patch('mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache') as trim, \
         patch.object(session.residency_manager, 'wait_for_transfers'), \
         patch.object(session.executor, 'reset', wraps=session.executor.reset) as reset:
        with session:
            reset.assert_called_once_with()
            assert session.last_trim_seconds == 0.0
        trim.assert_not_called()


def test_allocation_dispatch_reclaims_inactive_npu_cache_automatically():
    session = OffloadSession(OffloadConfig())
    dispatch_mode = ActivationDispatchMode(session)
    total_bytes = 64 * 1024**3
    inactive_bytes = 1024**3

    with patch("mindspeed.core.memory.hyper_offload.api.session.torch.npu", create=True) as npu:
        npu.is_initialized.return_value = True
        npu.current_device.return_value = 0
        npu.get_device_properties.return_value.total_memory = total_bytes
        npu.memory_stats_as_nested_dict.return_value = {
            "reserved_bytes": {"all": {"current": total_bytes * 95 // 100}},
            "allocated_bytes": {"all": {"current": total_bytes * 95 // 100 - inactive_bytes}},
        }
        with patch.object(session.executor, "dispatch", return_value="allocation") as dispatch:
            result = dispatch_mode.__torch_dispatch__(
                torch.ops.aten.empty.memory_format, (), ([8],), {}
            )

    assert result == "allocation"
    dispatch.assert_called_once()
    npu.empty_virt_addr_cache.assert_called_once_with()
    npu.empty_cache.assert_not_called()


def _npu_backend(*, reserved_fraction=.95, inactive_bytes=1024**3):
    total = 64 * 1024**3
    reserved = int(total * reserved_fraction)
    return SimpleNamespace(
        is_initialized=lambda: True, current_device=lambda: 0,
        get_device_properties=lambda device: SimpleNamespace(total_memory=total),
        memory_stats_as_nested_dict=Mock(return_value={
            'reserved_bytes': {'all': {'current': reserved}},
            'allocated_bytes': {'all': {'current': reserved-inactive_bytes}},
        }), empty_cache=Mock(), empty_virt_addr_cache=Mock(),
    )


@pytest.mark.parametrize('missing', [False, True])
def test_unavailable_virtual_trim_falls_back_once_and_keeps_replay_trim(missing):
    npu = _npu_backend()
    if missing:
        del npu.empty_virt_addr_cache
    else:
        npu.empty_virt_addr_cache.side_effect = RuntimeError('Unsupported config for empty_virt_addr_cache')
    session = OffloadSession(OffloadConfig())
    with patch('torch.npu', npu, create=True):
        session._reclaim_before_allocation(torch.ops.aten.empty.memory_format)
        session._reclaim_before_allocation(torch.ops.aten.empty.memory_format)
        assert session._virtual_cache_supported is False
        npu.empty_cache.assert_called_once_with()
        # Ordinary replay must retain its fallback even after pressure checks stop.
        assert _empty_accelerator_cache() is False
        assert npu.empty_cache.call_count == 2
    if not missing:
        npu.empty_virt_addr_cache.assert_called_once_with()


def test_virtual_trim_does_not_hide_unrelated_backend_failures():
    npu = _npu_backend()
    npu.empty_virt_addr_cache.side_effect = RuntimeError('device stream failed')
    with patch('torch.npu', npu, create=True), pytest.raises(RuntimeError, match='device stream failed'):
        _empty_accelerator_cache(retain_physical=True)
    npu.empty_cache.assert_not_called()


@pytest.mark.parametrize('fraction,inactive', [(.90, 1024**3), (.95, 511*1024**2)])
def test_pressure_reclaim_requires_both_capacity_conditions(fraction, inactive):
    npu = _npu_backend(reserved_fraction=fraction, inactive_bytes=inactive)
    session = OffloadSession(OffloadConfig(record_cache_reclaim=True))
    with patch('torch.npu', npu, create=True):
        session._reclaim_before_allocation(torch.ops.aten.empty.memory_format)
    npu.empty_virt_addr_cache.assert_not_called()
    assert session.cache_reclaim_stats['checks'] == 1
    assert session.cache_reclaim_stats['calls'] == 0


def test_pressure_observation_stops_without_disabling_runtime_reclaim():
    npu = _npu_backend()
    session = OffloadSession(OffloadConfig(record_cache_reclaim=True))
    with patch('torch.npu', npu, create=True):
        session._reclaim_before_allocation(torch.ops.aten.sin.default)
        npu.memory_stats_as_nested_dict.assert_not_called()
        with patch('mindspeed.core.memory.hyper_offload.api.session.time.perf_counter',
                   side_effect=[1., 2., 4., 5.]):
            session._reclaim_before_allocation(torch.ops.aten.empty.memory_format)
        assert session.cache_reclaim_stats == dict(checks=1, calls=1, check_seconds=4., reclaim_seconds=2.)
        session.record_cache_reclaim = False
        session._reclaim_before_allocation(torch.ops.aten.empty.memory_format)
        assert session.cache_reclaim_stats == dict(checks=1, calls=1, check_seconds=4., reclaim_seconds=2.)
        assert npu.empty_virt_addr_cache.call_count == 2
        assert session.last_trim_seconds == 0.


def test_pressure_reclaim_runs_before_fsdp_tracking_bypass():
    npu = _npu_backend()
    session = OffloadSession(OffloadConfig())
    mode = ActivationDispatchMode(session)
    operation = Mock(return_value='gathered')
    operation.__str__ = Mock(return_value='fsdp.all_gather_copy_in.default')
    with patch('torch.npu', npu, create=True), \
         patch('mindspeed.core.memory.hyper_offload.api.session._inside_fsdp_internal', return_value=True), \
         patch.object(session.executor, 'dispatch') as dispatch:
        assert mode.__torch_dispatch__(operation, (), (), {}) == 'gathered'
    npu.empty_virt_addr_cache.assert_called_once_with()
    dispatch.assert_not_called()
