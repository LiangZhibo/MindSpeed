"""Finite joint tuning over ordinary HyperOffload warmup and replay.

Each recomputation structure gets its own trace. Masking/budget variants reuse
only the matching trace; returning to an already measured policy reuses its
guide and schedule, never its live activations. All candidates share exactly
one bounded Host pool and residency manager.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable

from mindspeed.core.memory.hyper_offload.api.config import OffloadConfig
from mindspeed.core.memory.hyper_offload.api.session import OffloadSession
from mindspeed.core.memory.hyper_offload.api.session import _empty_accelerator_cache
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.ir.replay import OpGuide
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace
from mindspeed.core.memory.hyper_offload.planning.peak import PeakResidencyPlanner
from .controller import JointPolicy, JointTuningController, StepObservation


@dataclass(frozen=True)
class _Plan:
    trace: ActivationTrace
    guide: list[OpGuide]
    schedule: ResidencySchedule


class _RecordingPlanner:
    def __init__(self, policy: JointPolicy):
        self.planner = PeakResidencyPlanner(
            enable_communication_masking=policy.masking,
            transfer_window_scale=policy.transfer_window_scale,
            limit_prefetch_window=policy.limit_prefetch_window,
        )
        self.trace = None
        self.seconds = 0.0

    def build(self, trace):
        started = time.perf_counter()
        try:
            result = self.planner.build(trace)
            self.trace = trace
            return result
        finally:
            self.seconds += time.perf_counter() - started


class JointOffloadSession(OffloadSession):
    """An OffloadSession with step-bounded policy selection.

    The caller supplies complete-step observations outside this context, after
    optimizer completion and synchronization. Distributed callers reduce these
    observations consistently before ``observe``. Do not call observe once
    stable. The session then follows the ordinary fixed replay path.
    """

    def __init__(
        self,
        controller: JointTuningController,
        apply_recompute: Callable[[int | tuple[str, ...]], None] | None = None,
        *,
        apply_policy: Callable[[JointPolicy], None] | None = None,
        host_mb: int = 65536,
    ):
        if (apply_recompute is None) == (apply_policy is None):
            raise ValueError('provide exactly one of apply_recompute or apply_policy')
        self.controller = controller
        self._apply_recompute = apply_recompute
        self._apply_policy = apply_policy
        self.executed_policy = controller.policy
        self._plans: dict[JointPolicy, _Plan] = {}
        self._traces: dict[tuple, _Plan] = {}
        self._awaiting_observation = False
        self.trace_collections = 0
        self.plan_builds = 0
        self.plan_seconds = 0.0
        self.policy_switches = 0
        self.post_offload_trim_calls = 0
        self.post_offload_trim_seconds = 0.0
        self._apply_regions(controller.policy)
        super().__init__(
            OffloadConfig(
                max_resident_activation_mb=controller.policy.resident_mb,
                max_offload_activation_mb=host_mb,
                planner=_RecordingPlanner(controller.policy),
                trim_before_replay=controller.policy.trim_before_replay,
            )
        )
        self.residency_manager.prefetch_on_copy_stream = controller.policy.prefetch_on_copy_stream
        self.residency_manager.defer_h2d_release = controller.policy.defer_h2d_release

    @property
    def stable(self) -> bool:
        return self.controller.stable

    def _apply_regions(self, policy: JointPolicy) -> None:
        if self._apply_policy is not None:
            self._apply_policy(policy)
            return
        if policy.resident_regions:
            raise ValueError('resident regions require an apply_policy callback')
        self._apply_recompute(policy.recompute_count if policy.recompute_regions is None else policy.recompute_regions)

    def _trim_after_offload(self) -> None:
        started = time.perf_counter()
        try:
            _empty_accelerator_cache()
        finally:
            self.post_offload_trim_calls += 1
            self.post_offload_trim_seconds += time.perf_counter() - started

    def _replay_executor(self, plan: _Plan, policy: JointPolicy) -> ReplayExecutor:
        return ReplayExecutor(
            self.residency_manager, plan.schedule, plan.guide,
            after_offload=self._trim_after_offload if policy.trim_after_offload else None,
        )

    def _switch_policy(self, policy: JointPolicy) -> None:
        # All previous DMA is finished and surviving shadow tensors detach to
        # valid device storage before replacing the executor/trace namespace.
        self.residency_manager.sync_all_transfers()
        self.executor.reset()
        self.trim_before_replay = policy.trim_before_replay
        self.residency_manager.prefetch_on_copy_stream = policy.prefetch_on_copy_stream
        self.residency_manager.defer_h2d_release = policy.defer_h2d_release
        self._apply_regions(policy)
        planner = _RecordingPlanner(policy)
        plan = self._plans.get(policy)
        if plan is None and policy.structure_key in self._traces:
            previous = self._traces[policy.structure_key]
            trace = replace(previous.trace, memory_limit_bytes=policy.resident_mb * 1024**2)
            schedule = planner.build(trace)
            plan = _Plan(trace, previous.guide, schedule)
            self._plans[policy] = plan
            self.plan_builds += 1
            self.plan_seconds += planner.seconds
        self.planner = planner
        if plan is not None:
            self.executor = self._replay_executor(plan, policy)
            self.mode = 'replay'
        else:
            self.executor = WarmupExecutor(self.residency_manager, policy.resident_mb * 1024**2)
            self.mode = 'warmup'
        self.executed_policy = policy
        self.policy_switches += 1

    def __enter__(self):
        if self._dispatch_context is not None or OffloadSession.get_active() is not None:
            raise RuntimeError('nested joint offload contexts are unsupported')
        if self._cleanup_failed:
            raise RuntimeError('cannot reuse a joint session after cleanup failure')
        if self._awaiting_observation:
            raise RuntimeError('observe the completed training step before entering the next tuning step')
        if self.controller.policy != self.executed_policy:
            self._switch_policy(self.controller.policy)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        was_warmup = self.mode == 'warmup'
        result = super().__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None:
            if was_warmup:
                plan = _Plan(self.planner.trace, self.executor._guide, self.executor._schedule)
                self._plans[self.executed_policy] = plan
                self._traces[self.executed_policy.structure_key] = plan
                if self.executed_policy.trim_after_offload:
                    self.executor = self._replay_executor(plan, self.executed_policy)
                self.trace_collections += 1
                self.plan_builds += 1
                self.plan_seconds += self.planner.seconds
            self._awaiting_observation = not self.stable
        return result

    def observe(self, observation: StepObservation) -> JointPolicy:
        if self._dispatch_context is not None or OffloadSession.get_active() is not None:
            raise RuntimeError('observe only after the full forward/backward context has exited')
        if self.stable:
            return self.controller.policy
        if not self._awaiting_observation and not self.stable:
            raise RuntimeError('no completed joint tuning step to observe')
        self._awaiting_observation = False
        policy = self.controller.observe(observation)
        if self.stable:
            # The selected policy was applied before its confirmation steps.
            # Release unused trace metadata; keep the current fixed executor.
            self._plans.clear()
            self._traces.clear()
            self.planner = self.planner.planner
        return policy
