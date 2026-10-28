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
"""Factory reuse and opaque-region combinations omitted by direct wrapper tests."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload import megatron
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


@pytest.fixture
def cpu():
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer'), \
         patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
               return_value=(10., 10.)), \
         patch.object(ResidencyManager, 'wait_for_transfers'), \
         patch.object(ResidencyManager, 'sync_all_transfers'):
        yield


@pytest.mark.parametrize('callable_kind', ['partial', 'object', 'module'])
def test_opaque_accepts_callable_objects_and_partial(cpu, callable_kind):
    class Op:
        def __call__(self, x): return x.sin()
    class Module(torch.nn.Module):
        def forward(self, x): return x.sin()
    fn = {'partial': partial(torch.sin), 'object': Op(), 'module': Module()}[callable_kind]
    wrapped = skip_offload(fn)
    session = OffloadSession(OffloadConfig())
    for _ in range(2):
        x = torch.tensor([.5], requires_grad=True)
        with session:
            wrapped(x).sum().backward()
        torch.testing.assert_close(x.grad, x.detach().cos())


@pytest.mark.parametrize('structure', ['dict', 'tuple', 'list'])
def test_opaque_mixed_outputs_and_keyword_constants_preserve_gradients(cpu, structure):
    @skip_offload
    def region(x, *, scale=2):
        values = [x.sin() * scale, x.cos(), None, 7]
        return dict(zip(['a','b','none','count'], values)) if structure == 'dict' else tuple(values) if structure == 'tuple' else values
    session = OffloadSession(OffloadConfig())
    for _ in range(2):
        x = torch.tensor([.5], requires_grad=True)
        with session:
            output = region(x, scale=2)
            values = list(output.values()) if isinstance(output, dict) else output
            (values[0] + values[1]).sum().backward()
        assert values[2:] == (None, 7) if isinstance(values, tuple) else values[2:] == [None, 7]
        torch.testing.assert_close(x.grad, x.detach().cos() * 2 - x.detach().sin())


def args():
    return SimpleNamespace(hyper_offload=True, hyper_offload_planner='peak',
        hyper_offload_communication_masking=False, hyper_offload_device_mb=1, hyper_offload_host_mb=8)


def test_factory_repeated_lookup_reuses_wrapped_schedule_and_session():
    options = args()
    create = Mock()
    create.return_value.__enter__ = Mock()
    create.return_value.__exit__ = Mock(return_value=False)
    def schedule(*, forward_only=False): return forward_only
    original_wrap = megatron.wrap_schedule
    def wrap(fn, config): return original_wrap(fn, config, session_factory=create)
    with patch.dict(sys.modules, {'mindspeed.args_utils': SimpleNamespace(get_full_args=lambda: options)}), \
         patch.object(megatron, 'wrap_schedule', side_effect=wrap):
        factory = megatron.get_forward_backward_func_wrapper(lambda: schedule)
        first, second = factory(), factory()
        assert first is second
        assert first() is False and second() is False
        create.assert_called_once()
        options.hyper_offload_device_mb = 2
        assert factory() is not first
        options.hyper_offload = False
        assert factory() is schedule


@pytest.mark.parametrize('invalid', ['unknown_planner', 'greedy_masking'])
def test_config_builder_rejects_invalid_combinations(invalid):
    options = args()
    if invalid == 'unknown_planner': options.hyper_offload_planner = 'typo'
    else:
        options.hyper_offload_planner = 'greedy'
        options.hyper_offload_communication_masking = True
    with pytest.raises(ValueError, match='planner|masking'):
        megatron.config_from_args(options)


def test_generic_schedule_evaluation_is_bypassed_without_session_creation():
    create = Mock()
    def schedule(**kwargs): return kwargs['forward_only']
    wrapped = megatron.wrap_schedule(schedule, object(), session_factory=create)
    assert wrapped(forward_only=True) is True
    create.assert_not_called()


def test_backward_outside_owning_session_is_rejected(cpu):
    @skip_offload
    def region(x): return x.sin()
    session = OffloadSession(OffloadConfig())
    x = torch.ones(2, requires_grad=True)
    with session:
        result = region(x)
    with pytest.raises(RuntimeError, match='session|scope'):
        result.sum().backward()


def test_foreign_shadow_is_rejected_but_explicit_clone_is_supported(cpu):
    from ..test_correctness import BytePool, shadow
    owner = ResidencyManager(4096)
    owner._host_pool = BytePool()
    value = shadow(owner, 1, torch.ones(2))
    other = OffloadSession(OffloadConfig())
    with pytest.raises(RuntimeError, match='foreign ShadowTensor'):
        with other:
            value.sin()
    torch.testing.assert_close(value.resolve(), torch.ones(2))
    copied = value.clone()
    for _ in range(2):
        with other:
            torch.sin(copied)


def test_previous_step_opaque_graph_cannot_write_new_step_trace(cpu):
    @skip_offload
    def region(x): return x.sin()
    session = OffloadSession(OffloadConfig())
    x = torch.ones(2, requires_grad=True)
    with session:
        old = region(x)
    grad = torch.ones_like(old)
    with pytest.raises(RuntimeError, match='originating session scope'):
        with session:
            old.backward(gradient=grad)


def test_opaque_cannot_import_foreign_shadow_through_a_closure(cpu):
    from ..test_correctness import BytePool, shadow
    owner = ResidencyManager(4096)
    owner._host_pool = BytePool()
    value = shadow(owner, 1, torch.ones(2))
    @skip_offload
    def region(): return value
    session = OffloadSession(OffloadConfig())
    with pytest.raises(RuntimeError, match='foreign ShadowTensor'):
        with session: region()
    torch.testing.assert_close(value.resolve(), torch.ones(2))


@pytest.mark.parametrize('kind', ['detach', 'constant', 'mixed'])
def test_opaque_preserves_non_differentiable_output_flags(cpu, kind):
    @skip_offload
    def region(x):
        if kind == 'detach': return x.detach().sin()
        if kind == 'constant': return torch.ones_like(x)
        return x.sin(), x.detach().cos()
    session = OffloadSession(OffloadConfig())
    for _ in range(2):
        x = torch.tensor([.5], requires_grad=True)
        with session:
            result = region(x)
            if kind == 'mixed':
                assert result[0].requires_grad
                assert not result[1].requires_grad
                result[0].sum().backward()
            else:
                assert not result.requires_grad


def test_unused_opaque_outputs_do_not_backpropagate_nan_or_zero_grads(cpu):
    @skip_offload
    def region(x, unused): return x.sin(), x * float('nan'), unused * 3
    session = OffloadSession(OffloadConfig())
    for _ in range(2):
        x = torch.tensor([.5], requires_grad=True)
        unused = torch.ones(1, requires_grad=True)
        with session:
            outputs = region(x, unused)
            outputs[0].sum().backward()
        torch.testing.assert_close(x.grad, x.detach().cos())
        assert unused.grad is None
