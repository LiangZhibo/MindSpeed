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
"""Storage, view and event invariants on a real accelerator."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch.utils._python_dispatch import _disable_current_modes

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


@pytest.fixture
def storage():
    manager = ResidencyManager(1024 * 1024)
    yield manager
    manager.sync_all_transfers()
    manager.clear_runtime()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize('view', ['scalar', 'empty', 'offset', 'transpose', 'expand', 'stride'])
def test_real_storage_views_roundtrip(storage, dtype, view):
    device = torch.accelerator.current_accelerator()
    root = torch.arange(24, dtype=dtype, device=device).reshape(4, 6)
    original = {
        'scalar': root[1, 2], 'empty': root[1:1], 'offset': root[1:3, 2:5],
        'transpose': root.t(), 'expand': root[:1].expand(3, 6), 'stride': root[:, ::2],
    }[view]
    expected = original.cpu().clone()
    value = ShadowTensor(original, storage.bind(1, original), 1)
    for _ in range(3):
        storage.copy_d2h(1)
        storage.release_device(1)
        storage.copy_h2d(1)
        actual = value.resolve()
        assert actual.stride() == original.stride()
        assert actual.storage_offset() == original.storage_offset()
        storage.release_host(1)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize('operation', ['transpose_', 't_', 'unsqueeze_', 'squeeze_', 'as_strided_'])
def test_real_metadata_mutation_and_gradients(operation):
    device = torch.accelerator.current_accelerator()
    args = {'transpose_': (0, 1), 't_': (), 'unsqueeze_': (0,), 'squeeze_': (0,),
            'as_strided_': ((3, 2), (1, 3), 0)}[operation]
    shape = (1, 2, 3) if operation == 'squeeze_' else (2, 3)
    source = torch.arange(6., device=device).reshape(shape).requires_grad_()
    ref = source.detach().clone().requires_grad_()
    value_ref = ref * 2
    getattr(value_ref, operation)(*args)
    loss_ref = value_ref.square().sum()
    loss_ref.backward()
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    for _ in range(3):
        source.grad = None
        with session:
            value = source * 2
            result = getattr(value, operation)(*args)
            assert result is value
            assert value.shape == value_ref.shape
            assert value.stride() == value_ref.stride()
            loss = value.square().sum()
            loss.backward()
        torch.testing.assert_close(loss.detach().cpu(), loss_ref.detach().cpu())
        torch.testing.assert_close(source.grad.detach().cpu(), ref.grad.detach().cpu())


@pytest.mark.parametrize('after_reload', [False, True])
def test_real_trace_preserves_logical_storage_identity(after_reload):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(12., device=device).reshape(3, 4).requires_grad_()
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    with session:
        value = source * 2
        sid = value.storage_id
        if after_reload:
            with _disable_current_modes():
                session.residency_manager.copy_d2h(sid)
                session.residency_manager.release_device(sid)
                value.resolve()
        transposed = value.t()
        sliced = transposed[:, 1:]
        assert sid == transposed.storage_id == sliced.storage_id
        assert session.executor._tracker._ensure_id(value) == sid
        loss = sliced.square().sum()
        loss.backward()
    source.grad = None
    with session:
        value = source * 2
        transposed = value.t()
        sliced = transposed[:, 1:]
        sliced.square().sum().backward()
    expected = torch.zeros_like(source)
    expected[1:] = source.detach()[1:] * 8
    torch.testing.assert_close(source.grad.detach().cpu(), expected.cpu())


def test_real_resolve_keeps_host_reader_event(storage):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(1024., device=device)
    value = ShadowTensor(source, storage.bind(1, source), 1)
    storage.copy_d2h(1)
    storage.release_device(1)
    storage.copy_h2d(1)
    reader = value._buffer.device_event
    value.resolve()
    with patch.object(storage._host_pool, 'release', wraps=storage._host_pool.release) as release:
        storage.release_host(1)
    assert reader is not None
    assert release.call_args.kwargs['event'] is reader
    torch.testing.assert_close(value.resolve().cpu(), source.cpu())


def test_real_copy_failure_recycles_only_after_stream_completion(storage):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(1024., device=device)
    value = ShadowTensor(source, storage.bind(1, source), 1)
    event = Mock()
    event.record.side_effect = RuntimeError('injected record failure after copy enqueue')
    with patch('mindspeed.core.memory.hyper_offload.runtime.residency.torch.Event', return_value=event):
        with pytest.raises(RuntimeError, match='injected'):
            storage.copy_d2h(1)
    assert value._buffer.host_buffer is None
    allocated = storage._host_pool.total_allocated
    storage.copy_d2h(1)
    assert storage._host_pool.total_allocated == allocated
    storage.release_device(1)
    torch.testing.assert_close(value.resolve().cpu(), source.cpu())


@pytest.mark.parametrize('initial_size', [0, 1])
def test_real_out_resize_is_rejected_without_silent_stale_reads(storage, initial_size):
    device = torch.accelerator.current_accelerator()
    source = torch.zeros(initial_size, device=device)
    value = ShadowTensor(source, storage.bind(1, source), 1)
    with pytest.raises(NotImplementedError, match='storage resizing'):
        torch.add(torch.ones(3, device=device), 2, out=value)
    with pytest.raises(NotImplementedError, match='discard'):
        value.resolve()


def test_real_new_d2h_event_supersedes_old_h2d_reader(storage):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(1024., device=device)
    value = ShadowTensor(source, storage.bind(1, source), 1)
    storage.copy_d2h(1)
    storage.release_device(1)
    storage.copy_h2d(1)
    value.add_(1)
    storage.copy_d2h(1)
    writer = value._buffer.host_event
    with patch.object(storage._host_pool, 'release', wraps=storage._host_pool.release) as release:
        storage.release_host(1)
    assert release.call_args.kwargs['event'] is writer
    torch.testing.assert_close(value.resolve().cpu(), source.cpu() + 1)


def test_real_h2d_record_failure_drains_stream_before_host_reuse(storage):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(1024., device=device)
    value = ShadowTensor(source, storage.bind(1, source), 1)
    storage.copy_d2h(1)
    storage.release_device(1)
    event = Mock()
    event.record.side_effect = RuntimeError('injected H2D record failure')
    with patch('mindspeed.core.memory.hyper_offload.runtime.residency.torch.Event', return_value=event):
        with pytest.raises(RuntimeError, match='injected'):
            storage.copy_h2d(1)
    storage.copy_h2d(1)
    torch.testing.assert_close(value.resolve().cpu(), source.cpu())
