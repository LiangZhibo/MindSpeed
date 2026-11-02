"""Opaque Host invalidation must be visible to the replay planner."""
from dataclasses import replace
from unittest.mock import patch

import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType
from mindspeed.core.memory.hyper_offload.planning.peak import PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from .test_joint_session import cpu_runtime


def test_opaque_input_snapshot_is_copied_after_invalidation_and_replays_exactly():
    class CapturePlanner:
        def build(self, trace):
            self.trace = trace
            # Warmup has plenty of CPU memory; force gap offload in replay.
            return PeakResidencyPlanner(enable_communication_masking=True).build(
                replace(trace, memory_limit_bytes=0))

    @skip_offload
    def opaque(x):
        return x.cos()

    def forward(x):
        a = x.sin()
        b = opaque(a)
        c = b.tanh().sin()
        return (c * a).sum()

    torch.manual_seed(48)
    inputs = torch.randn(16)
    expected_x = inputs.clone().requires_grad_()
    expected = forward(expected_x)
    expected.backward()
    planner = CapturePlanner()
    original_lookup = ActivationTracker.get_activation_sid

    def include_cpu_activations(tracker, tensor):
        # Production deliberately excludes CPU data from accelerator offload.
        # Enable only that eligibility check for this CPU residency test.
        if tensor.device.type == 'cpu' and not isinstance(tensor, ShadowTensor):
            sid = tracker._storage_tracker.get(tensor.untyped_storage())
            return sid if sid in tracker._activation_sids else None
        return original_lookup(tracker, tensor)

    with cpu_runtime(), patch.object(ActivationTracker, 'get_activation_sid', include_cpu_activations):
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=64, planner=planner))
        # This CPU-only numerical test uses real byte copies, without requiring
        # a pinned accelerator allocator. The NPU suite covers asynchronous DMA.
        session.residency_manager._host_pool._pinning_exhausted = True
        for _ in range(3):
            x = inputs.clone().requires_grad_()
            with session:
                actual = forward(x)
                actual.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
            del x, actual
        trace = planner.trace
        op_id, op = next((i, op) for i, op in enumerate(trace.ops) if 'opaque_fwd' in op.name)
        # The input is produced by sin, and is not returned by this region.
        produced = {a.storage_id for earlier in trace.ops[:op_id] for a in earlier.accesses
                    if a.kind == AccessKind.WRITE}
        inputs = produced.intersection(a.storage_id for a in op.accesses)
        assert len(inputs) == 1
        sid = next(iter(inputs))
        assert any(a.storage_id == sid and a.kind == AccessKind.WRITE for a in op.accesses)
        copies = [index for index, actions in session.executor._schedule.post.items()
                  for action in actions if action.kind == ResidencyActionType.COPY_D2H and action.storage_id == sid]
        assert copies and all(index >= op_id for index in copies)
