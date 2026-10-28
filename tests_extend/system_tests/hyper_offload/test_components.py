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
"""Component combinations on Ascend/CUDA without a full Megatron installation."""

from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload, megatron
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.runtime.pinned_memory import PinnedMemoryPool
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('mutate', [False, True])
def test_negative_view_values_and_transfer_versions(dtype, mutate):
    device = torch.accelerator.current_accelerator()
    source = torch.arange(8, device=device, dtype=dtype)
    reference = source.clone()
    original = torch._neg_view(source)
    expected = torch._neg_view(reference)
    manager = ResidencyManager(4096)
    value = ShadowTensor(original, manager.bind(1, original), 1)
    assert value.is_neg()
    manager.copy_d2h(1)
    if mutate:
        value.add_(2)
        expected.add_(2)
    manager.release_device(1)
    torch.testing.assert_close(value.resolve().cpu(), expected.cpu())
    torch.testing.assert_close(source.cpu(), reference.cpu())
    manager.clear_runtime()


@pytest.mark.parametrize('kind', ['greedy', 'peak', 'masked'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_negative_view_full_step_gradient_and_replay(kind, dtype):
    device = torch.accelerator.current_accelerator()
    planner = GreedyResidencyPlanner() if kind == 'greedy' else PeakResidencyPlanner(enable_communication_masking=kind == 'masked')
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1, planner=planner))
    tolerance = dict(rtol=.02, atol=.002) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-6)
    for _ in range(3):
        source = torch.randn(4, 8, device=device, dtype=dtype, requires_grad=True)
        reference = source.detach().clone().requires_grad_()
        expected = torch._neg_view(reference * 2).sin().float().square().sum()
        expected.backward()
        with session:
            actual = torch._neg_view(source * 2).sin().float().square().sum()
            actual.backward()
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), **tolerance)
        torch.testing.assert_close(source.grad.detach().cpu(), reference.grad.detach().cpu(), **tolerance)


@pytest.mark.parametrize('kind', ['partial', 'module', 'callable'])
@pytest.mark.parametrize('planner_name', ['greedy', 'peak', 'masked'])
def test_callable_opaque_mixed_outputs_forward_backward(kind, planner_name):
    device = torch.accelerator.current_accelerator()
    def operation(x, scale=2): return {'main': x.sin() * scale, 'aux': x.cos(), 'none': None, 'count': 7}
    class Module(torch.nn.Module):
        def forward(self, x): return operation(x)
    class Callable:
        def __call__(self, x): return operation(x)
    fn = {'partial': partial(operation, scale=2), 'module': Module(), 'callable': Callable()}[kind]
    wrapped = skip_offload(fn)
    planner = GreedyResidencyPlanner() if planner_name == 'greedy' else PeakResidencyPlanner(enable_communication_masking=planner_name == 'masked')
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1, planner=planner))
    for _ in range(3):
        x = torch.randn(4, 8, device=device, requires_grad=True)
        ref = x.detach().clone().requires_grad_()
        expected_values = operation(ref)
        expected = (expected_values['main'] + expected_values['aux']).sum()
        expected.backward()
        with session:
            values = wrapped(x)
            loss = (values['main'] + values['aux']).sum()
            loss.backward()
        assert values['none'] is None and values['count'] == 7
        torch.testing.assert_close(loss.detach().cpu(), expected.detach().cpu())
        torch.testing.assert_close(x.grad.detach().cpu(), ref.grad.detach().cpu())


def test_factory_lookup_reuses_real_session_and_bypasses_evaluation():
    device = torch.accelerator.current_accelerator()
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).to(device)
    reference = deepcopy(model)
    options = SimpleNamespace(hyper_offload=True, hyper_offload_planner='peak',
        hyper_offload_communication_masking=True, hyper_offload_device_mb=1, hyper_offload_host_mb=8)
    sessions = []
    def create(config):
        value = OffloadSession(config)
        sessions.append(value)
        return value
    def schedule(x, *, forward_only=False):
        loss = model(x).square().mean()
        if not forward_only: loss.backward()
        return loss
    original = megatron.wrap_schedule
    def wrap(fn, config): return original(fn, config, session_factory=create)
    with patch.dict(sys.modules, {'mindspeed.args_utils': SimpleNamespace(get_full_args=lambda: options)}), \
         patch.object(megatron, 'wrap_schedule', side_effect=wrap):
        factory = megatron.get_forward_backward_func_wrapper(lambda: schedule)
        previous = None
        for _ in range(4):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = torch.randn(4, 8, device=device)
            expected = reference(x).square().mean()
            expected.backward()
            current = factory()
            if previous is not None: assert current is previous
            actual = current(x)
            torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu())
            for a, b in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(a.grad.detach().cpu(), b.grad.detach().cpu())
            previous = current
            with torch.no_grad(): current(x, forward_only=True)
        assert len(sessions) == 1 and sessions[0].mode == 'replay'


def test_probe_oom_fallback_preserves_warmup_and_next_replay():
    from mindspeed.core.memory.hyper_offload.runtime.bandwidth import profile_transfer_bandwidth
    device = torch.accelerator.current_accelerator()
    def fail_probe():
        with patch('mindspeed.core.memory.hyper_offload.runtime.bandwidth.torch.empty',
                   side_effect=torch.OutOfMemoryError('injected probe allocation OOM')):
            return profile_transfer_bandwidth()
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    x = torch.randn(8, device=device, requires_grad=True)
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
               side_effect=fail_probe):
        with session:
            x.square().sum().backward()
    x.grad = None
    with session:
        x.square().sum().backward()
    torch.testing.assert_close(x.grad.detach().cpu(), x.detach().cpu() * 2)


def test_pool_ownership_with_real_pinned_allocations():
    pool = PinnedMemoryPool(4096)
    value = pool.acquire(16)
    pool.release(value)
    with pytest.raises(ValueError, match='released|leased'): pool.release(value)
    with pytest.raises(ValueError, match='owned'): pool.release(torch.empty(16, dtype=torch.uint8, pin_memory=True))
    a, b = pool.acquire(16), pool.acquire(16)
    assert a.data_ptr() != b.data_ptr()
    pool.release(a)
    pool.release(b)


def test_rebound_storage_does_not_restore_previous_values():
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(4096)
    old = torch.tensor([1., 2.], device=device)
    value = ShadowTensor(old, manager.bind(1, old), 1)
    manager.copy_d2h(1)
    manager.bind(1, torch.tensor([11., 12.], device=device))
    manager.copy_d2h(1)
    manager.release_device(1)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([11., 12.]))
    manager.clear_runtime()


def test_foreign_shadow_cannot_enter_another_session_without_cloning():
    device = torch.accelerator.current_accelerator()
    first, second = OffloadSession(OffloadConfig()), OffloadSession(OffloadConfig())
    source = torch.ones(8, device=device)
    with first: output = source.sin()
    with pytest.raises(RuntimeError, match='foreign ShadowTensor'):
        with second: output.cos()
    raw = output.clone()
    with second: result = raw.cos()
    torch.testing.assert_close(result.cpu(), source.sin().cos().cpu())


def test_opaque_backward_after_exit_is_rejected_before_trace_changes():
    device = torch.accelerator.current_accelerator()
    region = skip_offload(torch.sin)
    session = OffloadSession(OffloadConfig())
    original = session.executor
    source = torch.ones(8, device=device, requires_grad=True)
    with session: output = region(source)
    with pytest.raises(RuntimeError, match='originating session scope'):
        output.sum().backward()
    assert original.op_idx == -1 and not original._ops


@pytest.mark.parametrize('opaque', [False, True])
def test_failed_step_timer_and_scope_can_retry_on_real_device(opaque):
    device = torch.accelerator.current_accelerator()
    state = {'fail': True}
    def operation(x):
        value = x.sin()
        if state['fail']: raise ValueError('injected forward failure')
        return value
    operation = skip_offload(operation) if opaque else operation
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    x = torch.ones(8, device=device, requires_grad=True)
    with pytest.raises(ValueError, match='injected'):
        with session: operation(x)
    state['fail'] = False
    for _ in range(2):
        x.grad = None
        with session: operation(x).sum().backward()
        torch.testing.assert_close(x.grad.detach().cpu(), x.detach().cpu().cos())


@pytest.mark.parametrize('host_only', [False, True])
def test_output_survives_pool_destruction_without_retaining_cache(host_only):
    import weakref
    import gc
    device = torch.accelerator.current_accelerator()
    def produce():
        owner = ResidencyManager(4096)
        raw = torch.tensor([1., 2.], device=device)
        value = ShadowTensor(raw, owner.bind(1, raw), 1)
        owner.copy_d2h(1)
        if host_only: owner.release_device(1)
        else: owner.clear_runtime()
        owner.sync_all_transfers()
        return value, weakref.ref(owner._host_pool)
    value, pool_ref = produce()
    gc.collect()
    assert pool_ref() is None
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([1., 2.]))
    value.add_(2)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([3., 4.]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_empty_storage_roundtrip_with_real_host_pool(dtype):
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(4096)
    raw = torch.empty(0, dtype=dtype, device=device)
    value = ShadowTensor(raw, manager.bind(1, raw), 1)
    for _ in range(3):
        manager.copy_d2h(1)
        manager.release_device(1)
        manager.copy_h2d(1)
        manager.release_host(1)
        assert value.resolve().shape == (0,) and value.dtype == dtype
    manager.clear_runtime()


@pytest.mark.parametrize('kind', ['detach', 'constant', 'mixed'])
def test_opaque_non_differentiable_outputs_keep_native_gradient_flags(kind):
    device = torch.accelerator.current_accelerator()
    @skip_offload
    def region(x):
        if kind == 'detach': return x.detach().sin()
        if kind == 'constant': return torch.ones_like(x)
        return x.sin(), x.detach().cos()
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    for _ in range(3):
        source = torch.ones(8, device=device, requires_grad=True)
        with session:
            result = region(source)
            if kind == 'mixed':
                assert result[0].requires_grad and not result[1].requires_grad
                result[0].sum().backward()
            else: assert not result.requires_grad
        if kind == 'mixed':
            torch.testing.assert_close(source.grad.detach().cpu(), source.detach().cpu().cos())


def test_unused_opaque_nan_branch_does_not_pollute_live_gradients():
    device = torch.accelerator.current_accelerator()
    @skip_offload
    def region(x, unused): return x.sin(), x * float('nan'), unused * 3
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    for _ in range(3):
        x = torch.ones(8, device=device, requires_grad=True)
        unused = torch.ones(8, device=device, requires_grad=True)
        with session:
            outputs = region(x, unused)
            outputs[0].sum().backward()
        torch.testing.assert_close(x.grad.detach().cpu(), x.detach().cpu().cos())
        assert unused.grad is None


@pytest.mark.parametrize('preexisting', [False, True])
def test_failed_bind_allocation_is_transactional_on_real_device(preexisting):
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(4096)
    value = None
    if preexisting:
        raw = torch.tensor([1., 2.], device=device)
        value = ShadowTensor(raw, manager.bind(1, raw), 1)
        manager.copy_d2h(1)
    previous = manager._residency.get(1)
    replacement = torch.tensor([11., 12.], device=device)
    with patch('torch.empty', side_effect=RuntimeError('injected bind allocation failure')):
        with pytest.raises(RuntimeError, match='injected'):
            manager.bind(1, replacement)
    assert manager._residency.get(1) is previous
    if value is not None:
        manager.release_device(1)
        torch.testing.assert_close(value.resolve().cpu(), torch.tensor([1., 2.]))
    manager.clear_runtime()
