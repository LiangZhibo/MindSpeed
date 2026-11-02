"""Checkpoint inputs must remain visible through the opaque recompute body."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind
from mindspeed.core.memory.hyper_offload.tuning.regions import NamedRecomputeRegions
from .test_joint_session import cpu_runtime


@pytest.mark.parametrize('offload_inputs', [False, True])
def test_checkpoint_boundary_inputs_roundtrip_and_preserve_gradients(offload_inputs):
    class CapturePlanner:
        def build(self, trace):
            self.trace = trace
            fwd_id, fwd = next((i, op) for i, op in enumerate(trace.ops) if 'pointwise_fwd' in op.name)
            bwd_id, bwd = next((i, op) for i, op in enumerate(trace.ops) if 'pointwise_bwd' in op.name)
            written = {a.storage_id for a in fwd.accesses if a.kind == AccessKind.WRITE}
            read = {a.storage_id for a in bwd.accesses if a.kind == AccessKind.READ}
            self.inputs = written & read
            assert len(self.inputs) == (2 if offload_inputs else 0)
            assert not self.inputs.intersection(trace.retained_sids)
            # Force precisely the new boundary gaps. Peak selection is covered
            # separately and can legitimately choose no copies for this tiny
            # model; this test must actually execute the boundary roundtrips.
            schedule = ResidencySchedule()
            for sid in self.inputs:
                schedule.add_post(fwd_id, ResidencyActionType.COPY_D2H, sid)
                schedule.add_post(fwd_id, ResidencyActionType.RELEASE_DEVICE, sid)
                schedule.add_pre(bwd_id, ResidencyActionType.COPY_H2D, sid)
                schedule.add_post(bwd_id, ResidencyActionType.RELEASE_HOST, sid)
            return schedule

    torch.manual_seed(91)
    reference = torch.nn.ModuleList([torch.nn.Linear(16, 32), torch.nn.Linear(16, 32),
                                   torch.nn.Linear(32, 16)])
    model = deepcopy(reference)
    regions = NamedRecomputeRegions(model)

    def pointwise(a, b):
        return torch.nn.functional.silu(a) * b

    selected = regions.register_callable('pointwise', pointwise, offload_checkpoint_inputs=offload_inputs)
    regions.apply(('pointwise',))
    saved_kinds = []

    def pack(tensor):
        # Keep only metadata here; retaining an additional tensor would change
        # the residency being tested. The hook's returned tensor is its save.
        saved_kinds.append((tuple(tensor.shape), isinstance(tensor, ShadowTensor)))
        return tensor

    def forward(modules, x, body):
        a, b = modules[0](x), modules[1](x)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            value = body(a, b)
        return modules[2](value).sin().cos().tanh().square().sum()

    original_lookup = ActivationTracker.get_activation_sid

    def include_cpu_activations(tracker, tensor):
        if tensor.device.type == 'cpu' and not isinstance(tensor, ShadowTensor):
            sid = tracker._storage_tracker.get(tensor.untyped_storage())
            return sid if sid in tracker._activation_sids else None
        return original_lookup(tracker, tensor)

    inputs = torch.randn(3, 16)
    planner = CapturePlanner()
    with cpu_runtime(), patch.object(ActivationTracker, 'get_activation_sid', include_cpu_activations):
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=64, planner=planner))
        session.residency_manager._host_pool._pinning_exhausted = True
        copies = []
        manager = session.residency_manager
        original_d2h, original_h2d = manager.copy_d2h, manager.copy_h2d

        def copy_d2h(sid):
            copies.append(('d2h', sid))
            return original_d2h(sid)

        def copy_h2d(sid):
            copies.append(('h2d', sid))
            return original_h2d(sid)

        manager.copy_d2h, manager.copy_h2d = copy_d2h, copy_h2d
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            x = inputs.clone().requires_grad_()
            x_ref = inputs.clone().requires_grad_()
            expected = forward(reference, x_ref, pointwise)
            expected.backward()
            saved_kinds.clear()
            with session:
                actual = forward(model, x, selected)
                actual.backward()
            # The checkpoint saves exactly the two projection outputs. Moving
            # its input saver outside the opaque boundary preserves shadows.
            boundary_saves = [shadow for shape, shadow in saved_kinds if shape == (3, 32)]
            assert boundary_saves == [offload_inputs, offload_inputs]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
            for actual_p, expected_p in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=0, atol=0)
            del actual, expected, x, x_ref

        if offload_inputs:
            trace = planner.trace
            fwd_id, fwd = next((i, op) for i, op in enumerate(trace.ops) if 'pointwise_fwd' in op.name)
            bwd_id, bwd = next((i, op) for i, op in enumerate(trace.ops) if 'pointwise_bwd' in op.name)
            written = {a.storage_id for a in fwd.accesses if a.kind == AccessKind.WRITE}
            read = {a.storage_id for a in bwd.accesses if a.kind == AccessKind.READ}
            inputs = written & read
            assert len(inputs) == 2
            assert copies.count(('d2h', next(iter(inputs)))) == 2
            schedule = session.executor._schedule
            for sid in inputs:
                assert copies.count(('d2h', sid)) == copies.count(('h2d', sid)) == 2
                assert any(fwd_id <= index < bwd_id and action.storage_id == sid
                           and action.kind == ResidencyActionType.COPY_D2H
                           for index, actions in schedule.post.items() for action in actions)
                assert any(index <= bwd_id and action.storage_id == sid
                           and action.kind == ResidencyActionType.COPY_H2D
                           for index, actions in schedule.pre.items() for action in actions)
