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
"""System test for peak offload planning and replay on a tiny network."""

from __future__ import annotations

import gc
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType
from mindspeed.core.memory.hyper_offload.planning.peak import PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import modeling
from mindspeed.core.memory.hyper_offload.runtime.residency import PhysicalBuffer


def _memory_backend():
    """Return the real NPU memory backend used by this ST."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu
    return None


def _synchronize_backend(backend) -> None:
    """Synchronize backend work before reading peak memory."""
    if hasattr(backend, "synchronize"):
        backend.synchronize()
        return
    backend.current_stream().synchronize()


def _measure_peak_memory(backend, step) -> int:
    """Measure real accelerator max allocated bytes for one step."""
    _synchronize_backend(backend)
    backend.empty_cache()
    backend.reset_peak_memory_stats()
    step()
    _synchronize_backend(backend)
    return int(backend.max_memory_allocated())


class _MockExecutorContext:
    """CPU-only patches for offload system tests."""

    def __init__(self) -> None:
        self._patches = []

    def start(self) -> None:
        """Start CPU-safe patches for warmup and replay helpers."""
        timer_patch = patch("mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer")
        mock_timer_cls = timer_patch.start()
        mock_timer_cls.return_value.start.return_value = None
        mock_timer_cls.return_value.stop.return_value = 0.0
        self._patches.append(timer_patch)

        bw_patch = patch(
            "mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
            return_value=(16.0, 16.0),
        )
        bw_patch.start()
        self._patches.append(bw_patch)

        stream_patch = patch("torch.Stream")
        stream_patch.start()
        self._patches.append(stream_patch)

        acc_stream_patch = patch("torch.accelerator.current_stream")
        mock_stream = acc_stream_patch.start()
        mock_stream.return_value.wait_stream.return_value = None
        mock_stream.return_value.synchronize.return_value = None
        self._patches.append(acc_stream_patch)

    def stop(self) -> None:
        """Stop all patches in reverse creation order."""
        for patcher in reversed(self._patches):
            patcher.stop()


class _RecordingPlanner(PeakResidencyPlanner):
    """Planner wrapper that keeps the warmup trace and emitted schedule."""

    def __init__(self) -> None:
        super().__init__(enable_communication_masking=True)
        self.trace = None
        self.schedule = None

    def build(self, trace):
        """Build a schedule and keep trace data for assertions."""
        self.trace = trace
        self.schedule = super().build(trace)
        return self.schedule


class _RecordingResidencyManager:
    """Record the residency actions executed by replay."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self._residency: dict[int, PhysicalBuffer] = {}

    @property
    def resident_bytes(self) -> int:
        """Return currently resident device bytes in the recording manager."""
        return sum(
            buffer.device_buffer.numel()
            for buffer in self._residency.values()
            if buffer.device_buffer is not None
        )

    def wait_for_transfers(self) -> None:
        """Match the residency manager interface for replay tests."""

    def sync_all_transfers(self) -> None:
        """Match the residency manager interface for replay tests."""

    def clear_runtime(self, retained_sids=None) -> None:
        """Clear recorded residency state."""
        self._residency.clear()

    def bind(self, sid: int, tensor: torch.Tensor):
        """Bind a storage id to the tensor device buffer."""
        self.calls.append(("bind", sid))
        storage = tensor.untyped_storage()
        dev_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        buffer = self._residency.setdefault(sid, PhysicalBuffer())
        buffer.device = tensor.device
        buffer.device_buffer = dev_view
        return buffer

    def copy_d2h(self, sid: int) -> None:
        """Record and perform a synchronous device-to-host copy."""
        self.calls.append(("copy_d2h", sid))
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(f"copy_d2h sid={sid}: no physical buffer registered")
        if buffer.host_buffer is not None:
            return
        if buffer.device_buffer is None:
            raise RuntimeError(f"copy_d2h sid={sid}: no device data to copy")
        buffer.host_buffer = buffer.device_buffer.cpu().clone()

    def copy_h2d(self, sid: int) -> None:
        """Record and perform a synchronous host-to-device copy."""
        self.calls.append(("copy_h2d", sid))
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(f"copy_h2d sid={sid}: no physical buffer registered")
        if buffer.device_buffer is not None:
            return
        if buffer.host_buffer is None:
            raise RuntimeError(f"copy_h2d sid={sid}: no host data to copy")
        if buffer.device is None:
            raise RuntimeError(f"copy_h2d sid={sid}: target device unknown")
        buffer.device_buffer = buffer.host_buffer.to(buffer.device)

    def release_device(self, sid: int) -> None:
        """Record device buffer release for a storage id."""
        self.calls.append(("release_device", sid))
        buffer = self._residency.get(sid)
        if buffer is None:
            return
        buffer.device_buffer = None
        if buffer.host_buffer is None:
            del self._residency[sid]

    def release_host(self, sid: int) -> None:
        """Record host buffer release for a storage id."""
        self.calls.append(("release_host", sid))
        buffer = self._residency.get(sid)
        if buffer is None:
            return
        buffer.host_buffer = None
        if buffer.device_buffer is None:
            del self._residency[sid]


class _TinyNet(nn.Module):
    """A tiny network that creates a short-lived activation gap."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim))
        self.w2 = nn.Parameter(torch.randn(dim, dim))
        self.w3 = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        """Run the tiny activation graph."""
        a = x @ self.w1
        b = torch.relu(a)
        c = b @ self.w2
        d = torch.relu(c)
        e = d @ self.w3
        return e + b


class _ActivationHeavyNet(nn.Module):
    """Network whose peak is dominated by saved intermediate activations."""

    def __init__(self, dim: int = 512, layers: int = 8) -> None:
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim))
            for _ in range(layers)
        ])

    def forward(self, x):
        """Run the activation-heavy layer stack."""
        for weight in self.weights:
            x = torch.relu(x @ weight)
        return x


class TestPeakSystem(unittest.TestCase):
    """End-to-end warmup trace -> peak planner -> replay action execution."""

    def setUp(self) -> None:
        self._ctx = _MockExecutorContext()
        self._ctx.start()

    def tearDown(self) -> None:
        self._ctx.stop()

    @staticmethod
    def baseline_peak(trace) -> int:
        """Return the unmasked trace peak computed by the planner model."""
        accesses_by_storage = modeling.group_accesses_by_storage(trace)
        resident_bytes, _, _, _, _ = modeling.build_profile_and_candidates(trace, accesses_by_storage)
        return max(resident_bytes, default=0)

    @staticmethod
    def simulate_device_peak(trace, schedule) -> int:
        """Simulate replay-time device residency from emitted actions."""
        produced_sids = set()
        resident_sids = set()
        peak = 0

        for op_id, op in enumerate(trace.ops):
            for action in schedule.pre_actions(op_id):
                if action.kind == ResidencyActionType.COPY_H2D:
                    resident_sids.add(action.storage_id)
            for access in op.accesses:
                if access.storage_id not in produced_sids:
                    produced_sids.add(access.storage_id)
                    resident_sids.add(access.storage_id)
            peak = max(peak, sum(trace.storage_sizes[sid] for sid in resident_sids))
            for action in schedule.post_actions(op_id):
                if action.kind == ResidencyActionType.RELEASE_DEVICE:
                    resident_sids.discard(action.storage_id)
        return peak

    @unittest.skipUnless(torch.accelerator.is_available(), "accelerator is not available")
    def test_peak_system_trace_to_schedule_to_replay(self) -> None:
        """Verify peak planning actions are replayed end to end."""
        device = torch.accelerator.current_accelerator()
        planner = _RecordingPlanner()
        config = OffloadConfig(max_resident_activation_mb=1, planner=planner)
        session = OffloadSession(config)

        dim = 80
        bs = 1024
        model = _TinyNet(dim=dim).to(device)
        x = torch.randn(bs, dim, device=device, requires_grad=True)

        with session:
            loss = model(x).sum()
            loss.backward()

        trace = planner.trace
        schedule = planner.schedule
        self.assertIsNotNone(trace)
        self.assertIsNotNone(schedule)
        self.assertGreater(len(trace.ops), 0)
        self.assertGreaterEqual(sum(len(v) for v in schedule.post.values()), 1)

        baseline_peak = self.baseline_peak(trace)
        masked_peak = self.simulate_device_peak(trace, schedule)
        self.assertGreater(baseline_peak, trace.memory_limit_bytes)
        self.assertLessEqual(masked_peak, trace.memory_limit_bytes)
        self.assertLess(masked_peak, baseline_peak)

        selected_copy_actions = [
            action
            for actions in schedule.post.values()
            for action in actions
            if action.kind == ResidencyActionType.COPY_D2H
        ]
        self.assertTrue(selected_copy_actions)
        copied_sids = {action.storage_id for action in selected_copy_actions}

        masked_pre_op = min(op_id for op_id, actions in schedule.pre.items() if actions)
        masked_pre_actions = schedule.pre_actions(masked_pre_op)
        self.assertEqual([action.kind for action in masked_pre_actions], [
            ResidencyActionType.COPY_H2D,
            ResidencyActionType.RELEASE_HOST,
        ])
        self.assertEqual(masked_pre_actions[0].storage_id, masked_pre_actions[1].storage_id)
        selected_sid = masked_pre_actions[0].storage_id
        self.assertIn(selected_sid, copied_sids)
        selected_copy_action = next(action for action in selected_copy_actions if action.storage_id == selected_sid)
        selected_accesses = [
            access
            for op in trace.ops
            for access in op.accesses
            if access.storage_id == selected_sid
        ]
        self.assertTrue(selected_accesses)
        self.assertLess(min(access.op_id for access in selected_accesses), selected_copy_action.op_id)
        self.assertLess(selected_copy_action.op_id, max(access.op_id for access in selected_accesses))

        unmasked_schedule = PeakResidencyPlanner().build(trace)
        unmasked_copy_op = next(
            op_id
            for op_id, actions in unmasked_schedule.post.items()
            for action in actions
            if action.kind == ResidencyActionType.COPY_D2H and action.storage_id == selected_sid
        )
        unmasked_release_op = next(
            op_id
            for op_id, actions in unmasked_schedule.post.items()
            for action in actions
            if action.kind == ResidencyActionType.RELEASE_DEVICE and action.storage_id == selected_sid
        )
        unmasked_pre_op = next(
            op_id
            for op_id, actions in unmasked_schedule.pre.items()
            for action in actions
            if action.kind == ResidencyActionType.COPY_H2D and action.storage_id == selected_sid
        )
        masked_release_op = next(
            op_id
            for op_id, actions in schedule.post.items()
            for action in actions
            if action.kind == ResidencyActionType.RELEASE_DEVICE and action.storage_id == selected_sid
        )
        self.assertEqual(unmasked_copy_op, selected_copy_action.op_id)
        self.assertGreaterEqual(masked_release_op, unmasked_release_op)
        self.assertGreater(masked_pre_op, masked_release_op)
        self.assertLessEqual(masked_pre_op, unmasked_pre_op)

        replay_manager = _RecordingResidencyManager()
        replay = ReplayExecutor(replay_manager, schedule, session.executor._guide)  # pylint: disable=protected-access
        session.executor = replay
        session.mode = "replay"

        x2 = torch.randn(bs, dim, device=device, requires_grad=True)
        model.zero_grad(set_to_none=True)
        with session:
            loss2 = model(x2).sum()
            loss2.backward()

        replay_calls = replay_manager.calls
        self.assertTrue(any(kind == "copy_d2h" for kind, _ in replay_calls))
        self.assertTrue(any(kind == "copy_h2d" for kind, _ in replay_calls))
        self.assertTrue(any(kind == "release_host" for kind, _ in replay_calls))
        self.assertTrue(any(kind == "release_device" for kind, _ in replay_calls))


class TestPeakSystemRealMemory(unittest.TestCase):
    """NPU-side peak memory regression for peak replay."""

    @staticmethod
    def _make_model_and_input(device, *, dim: int, bs: int, layers: int):
        """Create a deterministic activation-heavy model and input."""
        torch.manual_seed(0)
        model = _ActivationHeavyNet(dim=dim, layers=layers).to(device)
        x = torch.randn(bs, dim, device=device)
        return model, x

    @staticmethod
    def _run_step(model: nn.Module, x: torch.Tensor) -> None:
        """Run one forward and backward step."""
        model.zero_grad(set_to_none=True)
        x.grad = None
        loss = model(x).sum()
        loss.backward()

    @staticmethod
    def _schedule_memory_summary(trace, schedule) -> str:
        """Summarize selected residency actions and offloaded bytes."""
        copied_sids = {
            action.storage_id
            for actions in schedule.post.values()
            for action in actions
            if action.kind == ResidencyActionType.COPY_D2H
        }
        action_counts = {
            kind.name: sum(
                1
                for action_list in list(schedule.pre.values()) + list(schedule.post.values())
                for action in action_list
                if action.kind == kind
            )
            for kind in ResidencyActionType
        }
        offloaded_bytes = sum(trace.storage_sizes.get(sid, 0) for sid in copied_sids)
        largest_offloaded = max((trace.storage_sizes.get(sid, 0) for sid in copied_sids), default=0)
        return (
            f"action_counts={action_counts}, "
            f"offloaded_sids={len(copied_sids)}, "
            f"offloaded_bytes={offloaded_bytes / 1024**2:.2f} MiB, "
            f"largest_offloaded={largest_offloaded / 1024**2:.2f} MiB"
        )

    def _trace_path_peak_without_offload(self, backend, device, *, dim: int, bs: int, layers: int) -> int:
        """Measure the original trace path without warmup or replay offload."""
        model, x = self._make_model_and_input(device, dim=dim, bs=bs, layers=layers)
        try:
            return _measure_peak_memory(backend, lambda: self._run_step(model, x))
        finally:
            del model
            del x
            gc.collect()
            backend.empty_cache()

    def _peak_replay_peak(self, backend, device, *, dim: int, bs: int, layers: int, budget_mb: int) -> int:
        """Measure replay peak memory after building a peak schedule."""
        planner = _RecordingPlanner()
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=budget_mb, planner=planner))
        model, x = self._make_model_and_input(device, dim=dim, bs=bs, layers=layers)
        try:
            with session:
                self._run_step(model, x)

            trace = planner.trace
            schedule = planner.schedule
            self.assertIsNotNone(trace)
            self.assertIsNotNone(schedule)
            baseline_peak = TestPeakSystem.baseline_peak(trace)
            masked_peak = TestPeakSystem.simulate_device_peak(trace, schedule)
            self._last_trace_summary = (
                f"trace_baseline={baseline_peak / 1024**2:.2f} MiB, "
                f"trace_masked={masked_peak / 1024**2:.2f} MiB, "
                f"trace_budget={trace.memory_limit_bytes / 1024**2:.2f} MiB, "
                f"{self._schedule_memory_summary(trace, schedule)}"
            )
            self.assertGreater(
                baseline_peak,
                trace.memory_limit_bytes,
                self._last_trace_summary,
            )
            self.assertLess(
                masked_peak,
                baseline_peak,
                self._last_trace_summary,
            )

            def replay_step() -> None:
                with session:
                    self._run_step(model, x)

            model.zero_grad(set_to_none=True)
            x.grad = None
            return _measure_peak_memory(backend, replay_step)
        finally:
            del session
            del model
            del x
            gc.collect()
            backend.empty_cache()

    @unittest.skipUnless(torch.accelerator.is_available(), "accelerator is not available")
    def test_peak_replay_reduces_real_npu_peak_memory(self) -> None:
        """Verify peak replay reduces real accelerator peak memory."""
        backend = _memory_backend()
        if backend is None:
            self.skipTest("NPU memory backend is not available")
        for required in ("empty_cache", "reset_peak_memory_stats", "max_memory_allocated"):
            if not hasattr(backend, required):
                self.skipTest(f"accelerator memory backend does not support {required}")

        device = torch.accelerator.current_accelerator()
        dim = 512
        bs = 4096
        layers = 8
        budget_mb = 33

        trace_path_peak = self._trace_path_peak_without_offload(backend, device, dim=dim, bs=bs, layers=layers)
        peak_peak = self._peak_replay_peak(
            backend,
            device,
            dim=dim,
            bs=bs,
            layers=layers,
            budget_mb=budget_mb,
        )
        if trace_path_peak == 0 or peak_peak == 0:
            self.skipTest(
                "accelerator memory backend returned zero peak memory; "
                f"trace_path_peak={trace_path_peak}, peak_replay={peak_peak}"
            )

        self.assertLess(
            peak_peak,
            trace_path_peak,
            f"peak replay should reduce real accelerator peak memory versus the no-offload trace path: "
            f"peak_replay={peak_peak / 1024**2:.2f} MiB, "
            f"trace_path_peak={trace_path_peak / 1024**2:.2f} MiB, "
            f"{getattr(self, '_last_trace_summary', 'trace_summary=unavailable')}",
        )


if __name__ == "__main__":
    unittest.main()
