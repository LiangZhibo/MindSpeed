"""Finite joint tuning over ordinary HyperOffload warmup and replay.

Each recomputation structure gets its own trace. Masking/budget variants reuse
only the matching trace; returning to an already measured policy reuses its
guide and schedule, never its live activations. All candidates share exactly
one bounded Host pool and residency manager.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import wraps
import time
import torch
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


def _timed(counter):
    """Record infrequent tuning work, including failed attempts."""
    def decorate(function):
        @wraps(function)
        def call(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return function(self, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                setattr(self, counter, getattr(self, counter) + elapsed)
                self.timing_events.append(dict(counter=counter,elapsed_seconds=elapsed,
                                               completed_step=self.controller.steps))
        return call
    return decorate


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
            shared_transfer_calendar=policy.shared_transfer_calendar,
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
        codecs: dict | None = None,
        transfer_size_calibration: bool = False,
    ):
        if (apply_recompute is None) == (apply_policy is None):
            raise ValueError('provide exactly one of apply_recompute or apply_policy')
        if not isinstance(transfer_size_calibration,bool):
            raise ValueError('transfer_size_calibration must be boolean')
        self.transfer_size_calibration = transfer_size_calibration
        self._codecs = dict(codecs or {})
        if "raw" in self._codecs:
            raise ValueError("raw is reserved for uncompressed transport")
        for key, codec in self._codecs.items():
            if not isinstance(key, str) or not key.strip() or codec is None:
                raise ValueError("codec registry requires nonempty keys and codec instances")
            OffloadConfig(codec=codec)  # Validate before changing model regions.
        self._codecs["raw"] = None
        initial_codec = self._codec_for(controller.policy)
        self.timing_events = []
        self.switch_seconds = 0.0
        self.switch_drain_seconds = 0.0
        self.apply_regions_seconds = 0.0
        self.codec_prepare_seconds = 0.0
        self.warmup_context_seconds = 0.0
        self.warmup_finish_seconds = 0.0
        self._warmup_started = None
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
        self._policy_replay_entries = 0
        self.last_entry_trim_performed = False
        self.post_offload_trim_calls = 0
        self.post_offload_trim_seconds = 0.0
        self._apply_regions(controller.policy)
        super().__init__(
            OffloadConfig(
                max_resident_activation_mb=controller.policy.resident_mb,
                max_offload_activation_mb=host_mb,
                planner=_RecordingPlanner(controller.policy),
                codec=initial_codec,
                trim_before_replay=controller.policy.trim_before_replay,
                record_cache_reclaim=not controller.stable,
            )
        )
        if self.transfer_size_calibration:
            self.executor = self._warmup_executor(controller.policy)
        self.residency_manager.prefetch_on_copy_stream = controller.policy.prefetch_on_copy_stream
        self.residency_manager.defer_h2d_release = controller.policy.defer_h2d_release
        self._prepare_codec(initial_codec)

    def _transfer_probe_budget(self):
        limit = self.controller.limits.device_peak_bytes
        if limit is None or not torch.accelerator.is_available():return 0
        device = torch.accelerator.current_accelerator()
        stats = torch.get_device_module(device).memory_stats()
        allocated = stats.get('allocated_bytes.all.current')
        if not isinstance(allocated,int) or allocated<0:return 0
        pool = self.residency_manager._host_pool
        return max(0,min(limit-allocated,pool.max_host_bytes-pool.total_allocated,
                         self.controller.policy.resident_mb*1024**2))

    def _warmup_executor(self, policy):
        return WarmupExecutor(self.residency_manager,policy.resident_mb*1024**2,
            transfer_probe_budget=self._transfer_probe_budget if self.transfer_size_calibration else None)

    @property
    def stable(self) -> bool:
        return self.controller.stable

    @_timed("apply_regions_seconds")
    def _apply_regions(self, policy: JointPolicy) -> None:
        if self._apply_policy is not None:
            self._apply_policy(policy)
            return
        if policy.resident_regions:
            raise ValueError('resident regions require an apply_policy callback')
        self._apply_recompute(policy.recompute_count if policy.recompute_regions is None else policy.recompute_regions)

    def _codec_for(self, policy):
        try:
            return self._codecs[policy.codec_key]
        except KeyError:
            raise ValueError(f"unregistered codec profile: {policy.codec_key}") from None

    @_timed("codec_prepare_seconds")
    def _prepare_codec(self, codec):
        prepare = getattr(codec, "prepare", None)
        if prepare is not None:
            stream = self.residency_manager._get_copy_stream()
            prepare(stream.device, stream)

    @_timed("switch_drain_seconds")
    def _drain_previous_policy(self):
        self.residency_manager.sync_all_transfers()
        self.executor.reset()

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

    @_timed("switch_seconds")
    def _switch_policy(self, policy: JointPolicy) -> None:
        # All previous DMA is finished and surviving shadow tensors detach to
        # valid device storage before replacing the executor/trace namespace.
        codec = self._codec_for(policy)  # Reject unknown profiles before mutation.
        self._drain_previous_policy()
        self._prepare_codec(codec)
        self.residency_manager._codec = codec
        self.residency_manager._transfer_node = None
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
            self.executor = self._warmup_executor(policy)
            self.mode = 'warmup'
        self.executed_policy = policy
        self._policy_replay_entries = 0
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
        # Fixed cadence continues after freeze without measurements or search.
        policy = self.executed_policy
        was_replay = self.mode == 'replay'
        trim = policy.trim_before_replay and self._policy_replay_entries % policy.trim_replay_interval == 0
        self.last_entry_trim_performed = False
        self.trim_before_replay = trim
        try:
            entered = super().__enter__()
        finally:
            self.trim_before_replay = policy.trim_before_replay
        if was_replay:
            self._policy_replay_entries += 1
            self.last_entry_trim_performed = trim
        self._warmup_started = time.perf_counter() if self.mode == "warmup" else None
        return entered

    def __exit__(self, exc_type, exc_val, exc_tb):
        was_warmup = self.mode == 'warmup'
        finish_started = time.perf_counter() if was_warmup else None
        try:
            result = super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            if was_warmup:
                self.warmup_finish_seconds += time.perf_counter() - finish_started
                if self._warmup_started is not None:
                    # Inclusive training+tracing interval, not pure instrumentation overhead.
                    self.warmup_context_seconds += time.perf_counter() - self._warmup_started
                self._warmup_started = None
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
            self.record_cache_reclaim = False
            self._plans.clear()
            self._traces.clear()
            self.planner = self.planner.planner
        return policy
