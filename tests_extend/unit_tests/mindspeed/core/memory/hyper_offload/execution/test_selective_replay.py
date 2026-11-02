"""End-to-end contracts for plan-scoped bindings with real CPU byte copies."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import pytest

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType as Action, ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.test_correctness import BytePool


class CopyFirstActivation:
    def build(self, trace):
        assert trace.ops[0].name == 'sin.default'
        assert trace.ops[1].name == 'cos.default'
        assert trace.ops[2].name == 'view.default'
        sid = next(a.storage_id for a in trace.ops[0].accesses if a.kind == AccessKind.WRITE)
        self.sid = sid
        schedule = ResidencySchedule()
        schedule.add_post(0, Action.COPY_D2H, sid)
        schedule.add_post(0, Action.RELEASE_DEVICE, sid)
        schedule.add_pre(2, Action.COPY_H2D, sid)
        schedule.add_post(4, Action.RELEASE_HOST, sid)
        return schedule


class CPUActivationTracker(ActivationTracker):
    """Keep production identity rules; permit CPU activations in this test."""
    def get_activation_sid(self, tensor):
        if isinstance(tensor, ShadowTensor):
            return tensor.storage_id
        sid = self._storage_tracker.get(tensor.untyped_storage())
        return sid if sid in self._activation_sids else None


def compute(x, replace_right=None):
    a = x.sin()
    b = x.cos()
    left = a.view(-1)
    right = b.view(-1)
    loss = (left * (right if replace_right is None else replace_right)).sum()
    loss.backward()
    return a, b, left, loss


@pytest.mark.parametrize('opaque', [False, True])
def test_native_view_created_outside_tracking_keeps_original_input_contract(opaque):
    """FSDP may make an untracked native alias of a tracked layer output."""
    planner = CopyFirstActivation()
    session = OffloadSession(OffloadConfig(planner=planner))
    manager = session.residency_manager
    manager._host_pool = BytePool()
    manager._copy_stream = SimpleNamespace(device=torch.device('meta'), synchronize=lambda: None)
    manager.wait_for_transfers = lambda: None
    session.executor._timer = Mock()
    session.executor._timer.stop.return_value = 0.
    session.executor._tracker = CPUActivationTracker()

    def run(value):
        a, b = value.sin(), value.cos()
        left = a.view(-1)
        # Model the dispatch-disabled FSDP boundary: warmup's ShadowTensor
        # resolves to an ordinary view which intentionally has no binding.
        if opaque:
            escaped = skip_offload(lambda z: z * 2)(b).view(-1)
        else:
            with torch.utils._python_dispatch._disable_current_modes():
                escaped = b.view(-1)
        loss = (left * escaped).sum()
        loss.backward()
        return loss

    data = torch.linspace(-2, 2, 16).reshape(4, 4)
    reference = data.clone().requires_grad_()
    expected = run(reference)
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
               return_value=(10., 10.)):
        warmup = data.clone().requires_grad_()
        with session:
            run(warmup)
    value = data.clone().requires_grad_()
    with session:
        actual = run(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize('invalid_native_input', [False, True])
def test_partial_replay_preserves_gradients_aliases_and_retained_native_outputs(invalid_native_input):
    planner = CopyFirstActivation()
    session = OffloadSession(OffloadConfig(planner=planner))
    manager = session.residency_manager
    manager._host_pool = BytePool()
    manager._copy_stream = SimpleNamespace(device=torch.device('meta'), synchronize=lambda: None)
    manager.wait_for_transfers = lambda: None
    session.executor._timer = Mock()
    session.executor._timer.stop.return_value = 0.
    session.executor._tracker = CPUActivationTracker()
    data = torch.linspace(-2, 2, 16).reshape(4, 4)
    reference = data.clone().requires_grad_()
    expected = compute(reference)
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
               return_value=(10., 10.)):
        warmup = data.clone().requires_grad_()
        with session:
            compute(warmup)
    detach_sids = {
        sid for op in session.executor._guide if op.name == 'aten.detach.default'
        for sid in op.input_bindings.values()
    }
    assert session.executor._managed_sids == {planner.sid} | detach_sids
    # Gradient subclass identity must survive even when that storage never
    # transfers: Torch 2.7 otherwise elides an AccumulateGrad detach operation.
    assert detach_sids - {planner.sid}
    if invalid_native_input:
        value = data.clone().requires_grad_()
        replacement = value.flatten()
        with pytest.raises(RuntimeError, match='input metadata or bindings differ'):
            with session:
                compute(value, replace_right=replacement)
        return
    for _ in range(2):
        value = data.clone().requires_grad_()
        with patch.object(manager, 'copy_d2h', wraps=manager.copy_d2h) as d2h:
            with session:
                actual = compute(value)
            assert d2h.call_count == 1
        assert isinstance(actual[0], ShadowTensor)
        assert isinstance(actual[2], ShadowTensor)
        assert type(actual[1]) is torch.Tensor
        assert actual[0]._buffer is actual[2]._buffer
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        # The native output must survive later session resets and keep values.
        retained_native = actual[1]
        session.executor.reset()
        torch.testing.assert_close(retained_native, expected[1], rtol=0, atol=0)
