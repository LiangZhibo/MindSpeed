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
"""Atomic invariants and fault injection for the offload runtime."""

import gc
from unittest.mock import Mock, patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from mindspeed.core.memory.hyper_offload.runtime.pinned_memory import PinnedMemoryPool
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
from .test_correctness import manager, shadow, evict


@pytest.fixture
def host_pool():
    original_empty = torch.empty
    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original_empty(*args, **kwargs)
    with patch('torch.empty', side_effect=empty), patch('torch.Tensor.is_pinned', return_value=True):
        yield PinnedMemoryPool(4096)


@pytest.mark.parametrize('size_bytes', [1, 1024, 2048])
def test_full_pool_waits_for_larger_pending_bucket(host_pool, size_bytes):
    value = host_pool.acquire(4096)
    address = value.data_ptr()
    event = Mock()
    event.query.return_value = False
    host_pool.release(value, event)
    restored = host_pool.acquire(size_bytes)
    assert restored.data_ptr() == address
    assert restored.numel() == size_bytes
    event.synchronize.assert_called_once()
    assert host_pool.total_allocated == 4096


def test_allocation_failure_does_not_consume_pool_capacity(host_pool):
    with patch('torch.empty', side_effect=RuntimeError('injected allocation failure')):
        with pytest.raises(RuntimeError, match='injected'):
            host_pool.acquire(4096)
    assert host_pool.total_allocated == 0
    assert host_pool.acquire(4096).numel() == 4096


@pytest.mark.parametrize('size', [-1, 1.5, True])
def test_invalid_allocation_size_fails_without_allocating(host_pool, size):
    with pytest.raises((ValueError, TypeError)):
        host_pool.acquire(size)
    assert host_pool.total_allocated == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize('view', ['scalar', 'empty', 'offset', 'transpose', 'expand', 'stride'])
def test_storage_roundtrip_preserves_view_and_full_backing(manager, dtype, view):
    root = torch.arange(24, dtype=dtype).reshape(4, 6)
    values = {
        'scalar': root[1, 2], 'empty': root[1:1], 'offset': root[1:3, 2:5],
        'transpose': root.t(), 'expand': root[:1].expand(3, 6), 'stride': root[:, ::2],
    }
    original = values[view]
    value = shadow(manager, 1, original)
    expected = original.clone()
    for _ in range(3):
        evict(manager, 1)
        actual = value.resolve()
        assert tuple(actual.shape) == tuple(original.shape)
        assert actual.stride() == original.stride()
        assert actual.storage_offset() == original.storage_offset()
        assert actual.untyped_storage().nbytes() == root.untyped_storage().nbytes()
        torch.testing.assert_close(actual, expected)
    manager.clear_runtime()
    torch.testing.assert_close(value.resolve(), expected)


@pytest.mark.parametrize('operation', ['transpose_', 't_', 'unsqueeze_', 'squeeze_', 'as_strided_'])
def test_metadata_mutation_matches_native_tensor(manager, operation):
    source = torch.arange(6.).reshape(1, 2, 3) if operation == 'squeeze_' else torch.arange(6.).reshape(2, 3)
    expected = source.clone()
    value = shadow(manager, 1, source)
    alias = value.view(-1)
    args = {
        'transpose_': (0, 1), 't_': (), 'unsqueeze_': (0,), 'squeeze_': (0,),
        'as_strided_': ((3, 2), (1, 3), 0),
    }[operation]
    result = getattr(value, operation)(*args)
    getattr(expected, operation)(*args)
    assert result is value
    assert value.shape == expected.shape
    assert value.stride() == expected.stride()
    evict(manager, 1)
    torch.testing.assert_close(value.resolve(), expected)
    torch.testing.assert_close(alias.resolve(), torch.arange(6.))


@pytest.mark.parametrize('operation', ['resize_', 'resize_as_', 'set_'])
def test_storage_rebinding_is_explicitly_rejected_before_mutation(manager, operation):
    value = shadow(manager, 1, torch.arange(6.))
    args = {'resize_': (12,), 'resize_as_': (torch.ones(12),), 'set_': (torch.ones(12),)}[operation]
    with pytest.raises(NotImplementedError, match='storage'):
        getattr(value, operation)(*args)
    torch.testing.assert_close(value.resolve(), torch.arange(6.))


@pytest.mark.parametrize('after_reload', [False, True])
def test_tracker_maps_shadow_and_raw_alias_to_one_storage(manager, after_reload):
    tracker = ActivationTracker()
    source = torch.arange(6.)
    tracker.register_op_activations([], [source])
    sid = tracker._ensure_id(source)
    value = shadow(manager, sid, source)
    if after_reload:
        evict(manager, sid)
    raw = value.resolve()
    tracker.register_op_activations([value], [raw.view(2, 3)])
    assert tracker._ensure_id(value) == sid
    assert tracker._ensure_id(raw) == sid
    assert tracker.storage_sizes[sid] == 24
    assert tracker._activation_sids == {sid}


def test_resolve_keeps_h2d_event_for_host_recycling(manager):
    value = shadow(manager, 1, torch.ones(2))
    manager.copy_d2h(1)
    d2h, h2d = Mock(), Mock()
    value._buffer.host_event, value._buffer.device_event = d2h, h2d
    with patch('torch.accelerator.current_stream', return_value=Mock()), \
         patch('torch.Tensor.record_stream'):
        value.resolve()
    with patch.object(manager._host_pool, 'release') as release:
        manager.release_host(1)
    assert release.call_args.kwargs['event'] is h2d


def test_d2h_copy_failure_returns_host_allocation(manager):
    value = shadow(manager, 1, torch.ones(2))
    with patch('torch.Tensor.copy_', side_effect=RuntimeError('injected copy failure')):
        with pytest.raises(RuntimeError, match='injected'):
            manager.copy_d2h(1)
    assert value._buffer.host_buffer is None
    assert len(manager._host_pool.free) == 1
    manager.copy_d2h(1)
    assert not manager._host_pool.free


@pytest.fixture
def cpu_session():
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer'), \
         patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
               return_value=(10., 10.)), \
         patch.object(ResidencyManager, 'wait_for_transfers'), \
         patch.object(ResidencyManager, 'sync_all_transfers'):
        yield OffloadSession(OffloadConfig(max_resident_activation_mb=1))


@pytest.mark.parametrize('same_session', [False, True])
def test_nested_session_is_rejected_without_disturbing_outer(cpu_session, same_session):
    outer = cpu_session
    inner = outer if same_session else OffloadSession(OffloadConfig())
    with outer:
        torch.ones(2)
        with pytest.raises(RuntimeError, match='nested'):
            inner.__enter__()
        assert OffloadSession.get_active() is outer
        torch.ones(2)
    assert OffloadSession.get_active() is None


def test_dispatch_enter_failure_restores_active_context(cpu_session):
    with patch('mindspeed.core.memory.hyper_offload.api.session.ActivationDispatchMode.__enter__',
               side_effect=RuntimeError('injected enter failure')):
        with pytest.raises(RuntimeError, match='injected'):
            cpu_session.__enter__()
    assert OffloadSession.get_active() is None
    assert cpu_session._dispatch_context is None
    with cpu_session:
        torch.ones(2)


@pytest.mark.parametrize('stage', ['wait', 'profile', 'planner'])
def test_finalize_failure_resets_runtime_and_can_retry(cpu_session, stage):
    target = {
        'wait': (cpu_session.residency_manager, 'wait_for_transfers'),
        'profile': (cpu_session.executor, 'finish'),
        'planner': (cpu_session.planner, 'build'),
    }[stage]
    with patch.object(*target, side_effect=RuntimeError('injected finalize failure')):
        with pytest.raises(RuntimeError, match='injected'):
            with cpu_session:
                torch.ones(2)
    assert OffloadSession.get_active() is None
    assert cpu_session.executor.op_idx == -1
    assert not cpu_session.residency_manager._residency
    with cpu_session:
        torch.ones(2)


def test_reset_drops_cached_inputs(cpu_session):
    source = torch.ones(1024)
    reference = __import__('weakref').ref(source)
    with cpu_session:
        torch.sin(source)
    with cpu_session:
        torch.sin(source)
    cpu_session.executor.reset()
    del source
    gc.collect()
    assert reference() is None


def test_failed_pending_wait_keeps_pool_ownership(host_pool):
    value = host_pool.acquire(4096)
    event = Mock()
    event.query.return_value = False
    event.synchronize.side_effect = RuntimeError('injected event failure')
    host_pool.release(value, event)
    with pytest.raises(RuntimeError, match='injected'):
        host_pool.acquire(1)
    event.synchronize.side_effect = None
    assert host_pool.acquire(1).data_ptr() == value.data_ptr()
    assert host_pool.total_allocated == 4096


def test_oversize_bucket_can_satisfy_smaller_request(host_pool):
    with patch('mindspeed.core.memory.hyper_offload.runtime.pinned_memory._BUCKET_SIZES', [1, 2]):
        value = host_pool.acquire(8)
        host_pool.release(value)
        assert host_pool.acquire(4).data_ptr() == value.data_ptr()
        assert host_pool.total_allocated == 8


def test_cleanup_failure_preserves_original_error_and_blocks_reuse(cpu_session):
    with patch.object(cpu_session.residency_manager, 'sync_all_transfers',
                      side_effect=RuntimeError('injected cleanup failure')):
        with pytest.raises(ValueError, match='original model failure'):
            with cpu_session:
                torch.ones(2)
                raise ValueError('original model failure')
    assert OffloadSession.get_active() is None
    assert cpu_session.executor.op_idx == -1
    with pytest.raises(RuntimeError, match='cleanup failure'):
        cpu_session.__enter__()


def test_mode_exit_failure_restores_context_and_cleans_runtime(cpu_session):
    from mindspeed.core.memory.hyper_offload.api.session import ActivationDispatchMode
    original_exit = ActivationDispatchMode.__exit__
    def fail_after_exit(self, *args):
        original_exit(self, *args)
        raise RuntimeError('injected mode exit failure')
    with patch.object(ActivationDispatchMode, '__exit__', fail_after_exit):
        with pytest.raises(RuntimeError, match='injected'):
            with cpu_session:
                torch.ones(2)
    assert OffloadSession.get_active() is None
    assert cpu_session.executor.op_idx == -1
    with cpu_session:
        torch.ones(2)


@pytest.mark.parametrize('initial_size', [0, 1])
def test_out_resize_fails_explicitly_and_prevents_stale_reads(manager, initial_size):
    value = shadow(manager, 1, torch.zeros(initial_size))
    with pytest.raises(NotImplementedError, match='storage resizing'):
        torch.add(torch.ones(3), 2, out=value)
    with pytest.raises(NotImplementedError, match='discard'):
        value.resolve()


def test_multi_output_resize_invalidates_all_outputs(manager):
    values = shadow(manager, 1, torch.empty(0))
    indices = shadow(manager, 2, torch.empty(0, dtype=torch.int64))
    with pytest.raises(NotImplementedError, match='storage resizing'):
        torch.sort(torch.tensor([3., 1., 2.]), out=(values, indices))
    for value in (values, indices):
        with pytest.raises(NotImplementedError, match='discard'):
            value.resolve()


def test_new_d2h_event_protects_host_after_an_older_h2d(manager):
    from unittest.mock import MagicMock
    value = shadow(manager, 1, torch.ones(2))
    manager.copy_d2h(1)
    old_reader = Mock()
    value._buffer.device_event = old_reader
    with patch('torch.Tensor.is_pinned', return_value=True), \
         patch('torch.accelerator.current_stream', return_value=Mock()), \
         patch('torch.Tensor.record_stream'):
        value.add_(1)
        stream = MagicMock()
        stream.device = torch.device('cpu')
        manager._copy_stream = stream
        latest_writer = Mock()
        with patch('torch.Event', return_value=latest_writer):
            manager.copy_d2h(1)
    with patch.object(manager._host_pool, 'release') as release:
        manager.release_host(1)
    assert release.call_args.kwargs['event'] is latest_writer


def test_h2d_copy_failure_leaves_valid_host_state_for_retry(manager):
    value = shadow(manager, 1, torch.arange(4.))
    evict(manager, 1)
    with patch('torch.Tensor.copy_', side_effect=RuntimeError('injected H2D failure')):
        with pytest.raises(RuntimeError, match='injected'):
            manager.copy_h2d(1)
    assert value._buffer.device_buffer is None
    manager.copy_h2d(1)
    torch.testing.assert_close(value.resolve(), torch.arange(4.))


def test_tracker_records_maximum_size_after_storage_growth():
    tracker = ActivationTracker()
    raw = torch.empty(0)
    tracker.register_op_activations([], [raw])
    sid = tracker._ensure_id(raw)
    raw.resize_(12)
    assert tracker._ensure_id(raw) == sid
    assert tracker.storage_sizes[sid] >= 48


def test_dispatch_does_not_retain_resolved_views_until_cyclic_gc(manager):
    import weakref
    value = shadow(manager, 1, torch.ones(1024))
    refs = []
    resolve = value.resolve
    def record():
        raw = resolve()
        refs.append(weakref.ref(raw))
        return raw
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with patch.object(value, 'resolve', side_effect=record):
            value.sin()
        assert all(ref() is None for ref in refs)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize('direction', ['d2h', 'h2d'])
def test_failed_transfer_and_failed_drain_keep_a_host_fence(manager, direction):
    from unittest.mock import MagicMock
    value = shadow(manager, 1, torch.arange(4.))
    if direction == 'h2d':
        evict(manager, 1)
    stream = MagicMock()
    stream.device = torch.device('cpu')
    stream.synchronize.side_effect = RuntimeError('injected drain failure')
    manager._copy_stream = stream
    event = Mock()
    event.record.side_effect = RuntimeError('original transfer failure')
    with patch('torch.Tensor.is_pinned', return_value=True), \
         patch('torch.accelerator.current_stream', return_value=Mock()), \
         patch('torch.Tensor.record_stream'), patch('torch.Event', return_value=event), \
         patch.object(manager._host_pool, 'release') as release:
        with pytest.raises(RuntimeError, match='original transfer failure'):
            getattr(manager, 'copy_' + direction)(1)
        if direction == 'h2d':
            manager.release_host(1)
        assert release.call_args.kwargs['event'] is stream
    assert stream.synchronize.called
