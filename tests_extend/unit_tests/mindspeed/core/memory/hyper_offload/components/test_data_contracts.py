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
"""Value flags, borrowed storage, ownership and profiling failure contracts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import math

import pytest
import torch

from mindspeed.core.memory.hyper_offload.execution.validation import signature
from mindspeed.core.memory.hyper_offload.runtime.pinned_memory import PinnedMemoryPool
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
from mindspeed.core.memory.hyper_offload.runtime.timer import DeviceTimer
from mindspeed.core.memory.hyper_offload.runtime.bandwidth import profile_transfer_bandwidth
from ..test_correctness import manager, shadow, evict


@pytest.mark.parametrize('flag', ['conj', 'neg', 'both'])
def test_lazy_value_flags_survive_shadow_and_offload(manager, flag):
    source = torch.tensor([1+2j, 3-4j])
    if flag in ('conj', 'both'): source = source.conj()
    if flag in ('neg', 'both'): source = torch._neg_view(source)
    expected = source.clone()
    value = shadow(manager, 1, source)
    assert value.is_conj() == source.is_conj()
    assert value.is_neg() == source.is_neg()
    torch.testing.assert_close(value + 0, expected)
    evict(manager, 1)
    torch.testing.assert_close(value.resolve(), expected)


@pytest.mark.parametrize('flag', ['conj', 'neg'])
def test_signature_distinguishes_value_flags(flag):
    plain = torch.tensor([1+2j])
    lazy = plain.conj() if flag == 'conj' else torch._neg_view(plain)
    assert signature(plain) != signature(lazy)


def test_rebinding_storage_retires_old_host_snapshot(manager):
    value = shadow(manager, 1, torch.tensor([1., 2.]))
    manager.copy_d2h(1)
    manager.bind(1, torch.tensor([11., 12.]))
    evict(manager, 1)
    torch.testing.assert_close(value.resolve(), torch.tensor([11., 12.]))


@pytest.fixture
def pool():
    original = torch.empty
    def empty(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return original(*args, **kwargs)
    with patch('torch.empty', side_effect=empty), patch('torch.Tensor.is_pinned', return_value=True):
        yield PinnedMemoryPool(4096)


@pytest.mark.parametrize('kind', ['double', 'foreign', 'other_pool'])
def test_pool_rejects_unowned_release_without_aliasing_allocations(pool, kind):
    value = pool.acquire(16)
    if kind == 'double':
        pool.release(value)
        invalid = value
    elif kind == 'foreign': invalid = torch.empty(16, dtype=torch.uint8)
    else: invalid = PinnedMemoryPool(4096).acquire(16)
    with pytest.raises(ValueError, match='owned|leased|released'):
        pool.release(invalid)
    if kind != 'double': pool.release(value)
    first, second = pool.acquire(16), pool.acquire(16)
    assert first.untyped_storage()._cdata != second.untyped_storage()._cdata


def test_timer_rejects_overlapping_measurement_and_can_reset():
    event = Mock()
    with patch('torch.accelerator.is_available', return_value=True), \
         patch('torch.accelerator.current_stream', return_value=Mock()), \
         patch('torch.Event', return_value=event):
        timer = DeviceTimer()
        timer.start()
        with pytest.raises(RuntimeError, match='already|active'):
            timer.start()
        timer.reset()
        timer.start()
        event.elapsed_time.return_value = 1.25
        assert timer.stop() == 1.25


def test_failed_timer_stop_clears_start_for_next_measurement():
    start, end = Mock(), Mock()
    end.synchronize.side_effect = RuntimeError('injected event failure')
    with patch('torch.accelerator.is_available', return_value=True), \
         patch('torch.accelerator.current_stream', return_value=Mock()), \
         patch('torch.Event', side_effect=[start, end]):
        timer = DeviceTimer()
        timer.start()
        with pytest.raises(RuntimeError, match='injected'):
            timer.stop()
    assert timer._start_event is None


@pytest.mark.parametrize('timing', [0., -1., float('nan'), float('inf')])
def test_bandwidth_bad_timing_uses_finite_conservative_fallback(timing):
    original = torch.empty
    with patch('torch.accelerator.is_available', return_value=True), \
         patch('torch.accelerator.current_accelerator', return_value=torch.device('cpu')), \
         patch('mindspeed.core.memory.hyper_offload.runtime.bandwidth.torch.empty',
               side_effect=lambda *a, **k: original(8, dtype=torch.uint8)), \
         patch('mindspeed.core.memory.hyper_offload.runtime.bandwidth.DeviceTimer') as timer:
        timer.return_value.stop.return_value = timing
        result = profile_transfer_bandwidth()
    assert result == (1., 1.)
    assert all(math.isfinite(x) for x in result)


def test_bandwidth_allocation_oom_does_not_abort_a_successful_warmup():
    with patch('torch.accelerator.is_available', return_value=True), \
         patch('torch.accelerator.current_accelerator', return_value=torch.device('cpu')), \
         patch('torch.empty', side_effect=torch.OutOfMemoryError('injected OOM')):
        assert profile_transfer_bandwidth() == (1., 1.)


@pytest.mark.parametrize('flag', ['conj', 'neg', 'both'])
def test_mutation_through_lazy_flag_updates_underlying_storage(manager, flag):
    plain = torch.tensor([1+2j, 3-4j])
    reference = plain.clone()
    actual_view, expected_view = plain, reference
    if flag in ('conj', 'both'):
        actual_view, expected_view = actual_view.conj(), expected_view.conj()
    if flag in ('neg', 'both'):
        actual_view, expected_view = torch._neg_view(actual_view), torch._neg_view(expected_view)
    value = shadow(manager, 1, actual_view)
    manager.copy_d2h(1)
    value.add_(2+1j)
    expected_view.add_(2+1j)
    torch.testing.assert_close(plain, reference)
    evict(manager, 1)
    torch.testing.assert_close(value.resolve(), expected_view)


def test_pool_concurrent_leases_never_share_live_storage(pool):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    barrier = threading.Barrier(4)
    pointers = [None] * 4
    def worker(index):
        for _ in range(10):
            value = pool.acquire(16)
            value.fill_(index)
            pointers[index] = value.data_ptr()
            barrier.wait(timeout=10)
            assert len(set(pointers)) == 4
            assert value.eq(index).all()
            barrier.wait(timeout=10)
            pool.release(value)
            barrier.wait(timeout=10)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        for future in futures: future.result(timeout=30)
    assert pool.total_allocated == 4096


@pytest.mark.parametrize('host_only', [False, True])
def test_surviving_output_does_not_keep_entire_host_pool_alive(host_only):
    import weakref
    import gc
    from ..test_correctness import BytePool
    def produce():
        owner = ResidencyManager(4096)
        owner._host_pool = BytePool()
        owner._copy_stream = SimpleNamespace(device=torch.device('meta'))
        value = shadow(owner, 1, torch.tensor([1., 2.]))
        owner.copy_d2h(1)
        if host_only:
            owner.release_device(1)
        else:
            owner.clear_runtime()
        return value, weakref.ref(owner._host_pool)
    value, pool_ref = produce()
    gc.collect()
    assert pool_ref() is None
    torch.testing.assert_close(value.resolve(), torch.tensor([1., 2.]))
    value.add_(2)
    torch.testing.assert_close(value.resolve(), torch.tensor([3., 4.]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_zero_byte_storage_roundtrip_is_not_a_missing_buffer(manager, dtype):
    original = torch.empty(0, dtype=dtype)
    value = shadow(manager, 1, original)
    for _ in range(3):
        evict(manager, 1)
        manager.copy_h2d(1)
        manager.release_host(1)
        assert value.resolve().shape == (0,)
        assert value.resolve().dtype == dtype
    manager.clear_runtime()


def test_signature_handles_non_strided_and_metadata_only_tensors():
    sparse = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([2.]), (2, 2))
    meta = torch.empty(2, device='meta')
    assert signature(sparse) == signature(sparse)
    assert signature(meta) == signature(torch.empty(2, device='meta'))
    assert signature(sparse) != signature(sparse.to_dense())


@pytest.mark.parametrize('preexisting', [False, True])
@pytest.mark.parametrize('failure', ['storage', 'allocation', 'view'])
def test_failed_bind_does_not_create_or_corrupt_registry_entries(manager, preexisting, failure):
    value = None
    if preexisting:
        value = shadow(manager, 1, torch.tensor([1., 2.]))
        manager.copy_d2h(1)
    original_buffer = manager._residency.get(1)
    raw = torch.tensor([11., 12.])
    if failure == 'storage':
        raw = SimpleNamespace(device=torch.device('cpu'), untyped_storage=Mock(side_effect=RuntimeError('injected bind failure')))
        context = patch.object(raw, 'device', torch.device('cpu'))
    elif failure == 'allocation': context = patch('torch.empty', side_effect=RuntimeError('injected bind failure'))
    else: context = patch('torch.Tensor.set_', side_effect=RuntimeError('injected bind failure'))
    with context:
        with pytest.raises(RuntimeError, match='injected'):
            manager.bind(1, raw)
    assert manager._residency.get(1) is original_buffer
    if value is not None:
        evict(manager, 1)
        torch.testing.assert_close(value.resolve(), torch.tensor([1., 2.]))
    manager.clear_runtime()
