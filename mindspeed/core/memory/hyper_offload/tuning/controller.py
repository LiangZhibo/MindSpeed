"""Small bounded search state machine, independent of Torch and model hooks.

The integration applies ``policy`` only between complete training iterations,
refreshes the trace whenever recomputation changes, and supplies a max-rank
observation. A warmup/settling iteration is excluded from performance scoring
but is always included in memory qualification and in the tuning step budget.
There is no background search after the controller enters ``stable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median
import time
from typing import Callable


@dataclass(frozen=True)
class JointPolicy:
    recompute_count: int
    resident_mb: int
    masking: bool = True
    transfer_window_scale: float | None = None
    recompute_regions: tuple[str, ...] | None = None
    limit_prefetch_window: bool = True
    resident_regions: tuple[str, ...] = ()
    prefetch_on_copy_stream: bool = False
    trim_before_replay: bool = True
    defer_h2d_release: bool = False
    trim_after_offload: bool = False

    def __post_init__(self):
        if not isinstance(self.limit_prefetch_window, bool):
            raise ValueError('limit_prefetch_window must be a boolean')
        if not isinstance(self.prefetch_on_copy_stream, bool):
            raise ValueError('prefetch_on_copy_stream must be a boolean')
        if not isinstance(self.trim_before_replay, bool):
            raise ValueError('trim_before_replay must be a boolean')
        if not isinstance(self.defer_h2d_release, bool):
            raise ValueError('defer_h2d_release must be a boolean')
        if not isinstance(self.trim_after_offload, bool):
            raise ValueError('trim_after_offload must be a boolean')
        for name in ('recompute_count', 'resident_mb'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.transfer_window_scale is not None:
            if not self.masking or not math.isfinite(self.transfer_window_scale) or self.transfer_window_scale < 0:
                raise ValueError('transfer windows require masking and a finite nonnegative scale')
        if self.recompute_regions is not None:
            if self.recompute_count:
                raise ValueError('named regions and a layer prefix cannot be selected together')
            if not isinstance(self.recompute_regions, tuple) or any(
                not isinstance(name, str) or not name or any(not part for part in name.split('.'))
                for name in self.recompute_regions
            ):
                raise ValueError('recompute_regions must be a tuple of nonempty dotted names')
            if len(set(self.recompute_regions)) != len(self.recompute_regions):
                raise ValueError('duplicate recompute region')
            object.__setattr__(self, 'recompute_regions', tuple(sorted(self.recompute_regions)))
            names = set(self.recompute_regions)
            for name in names:
                parts = name.split('.')
                if any('.'.join(parts[:i]) in names for i in range(1, len(parts))):
                    raise ValueError('parent and child recompute regions overlap')
        if not isinstance(self.resident_regions, tuple) or any(
            not isinstance(name, str) or not name or any(not part for part in name.split('.'))
            for name in self.resident_regions
        ):
            raise ValueError('resident_regions must be a tuple of nonempty dotted names')
        if len(set(self.resident_regions)) != len(self.resident_regions):
            raise ValueError('duplicate resident region')
        if self.resident_regions and self.recompute_count:
            raise ValueError('resident regions require named recomputation instead of a layer prefix')
        object.__setattr__(self, 'resident_regions', tuple(sorted(self.resident_regions)))
        if set(self.resident_regions).intersection(self.recompute_regions or ()):
            raise ValueError('a region cannot be both recomputed and resident')
        names = set(self.resident_regions).union(self.recompute_regions or ())
        for name in names:
            parts = name.split('.')
            if any('.'.join(parts[:i]) in names for i in range(1, len(parts))):
                raise ValueError('parent and child regions overlap')

    @property
    def structure_key(self) -> tuple:
        # Equal cardinality is not equal execution structure.
        recompute = (
            ('prefix', self.recompute_count) if self.recompute_regions is None else ('regions', self.recompute_regions)
        )
        return recompute + (self.resident_regions,)


@dataclass(frozen=True)
class TuningLimits:
    max_steps: int = 10
    max_candidates: int = 3
    samples_per_candidate: int = 2
    confirmation_steps: int = 2
    device_peak_bytes: int | None = None
    host_peak_bytes: int | None = None
    max_search_seconds: float = 1.0
    minimum_relative_gain: float = 0.005

    def __post_init__(self):
        for name in ('max_steps', 'max_candidates', 'samples_per_candidate', 'confirmation_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_steps < 1 + self.samples_per_candidate + self.confirmation_steps:
            raise ValueError('step budget must include initial warmup, measurement and confirmation')
        if not math.isfinite(self.max_search_seconds) or self.max_search_seconds <= 0:
            raise ValueError('search time budget must be finite and positive')
        if not math.isfinite(self.minimum_relative_gain) or not 0 <= self.minimum_relative_gain < 1:
            raise ValueError('minimum_relative_gain must be in [0,1)')
        for name in ('device_peak_bytes', 'host_peak_bytes'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f'{name} must be nonnegative integer bytes or None')


@dataclass(frozen=True)
class StepObservation:
    elapsed_seconds: float
    device_peak_bytes: int
    host_peak_bytes: int
    valid: bool = True


@dataclass
class Trial:
    policy: JointPolicy
    samples: list[float] = field(default_factory=list)
    device_peak_bytes: int = 0
    host_peak_bytes: int = 0
    feasible: bool = True

    @property
    def score(self) -> float:
        return median(self.samples) if self.feasible and self.samples else math.inf


class JointTuningController:
    """Evaluate a few joint policies and irrevocably freeze within a step cap.

    ``propose`` receives completed trials and returns one next candidate or
    None. It must be a cheap, synchronous policy proposal, not a GPU benchmark.
    The same reduced observations must be supplied on all participating ranks.
    A failed training iteration itself must be handled by the training loop;
    this controller never treats a partially executed step as recoverable.
    """

    def __init__(
        self,
        initial: JointPolicy,
        propose: Callable[[tuple[Trial, ...]], JointPolicy | None],
        limits: TuningLimits | None = None,
    ):
        self.limits = limits or TuningLimits()
        self.policy = initial
        self.phase = 'warmup'
        self.steps = 0
        self.frozen_at_step: int | None = None
        self.search_seconds = 0.0
        self.stop_reason: str | None = None
        self.trials: list[Trial] = []
        self.confirmation: list[StepObservation] = []
        self._trial = Trial(initial)
        self._propose = propose

    @property
    def stable(self) -> bool:
        return self.phase == 'stable'

    def _feasible(self, observation: StepObservation) -> bool:
        limits = self.limits
        return (
            observation.valid
            and math.isfinite(observation.elapsed_seconds)
            and observation.elapsed_seconds > 0
            and observation.device_peak_bytes >= 0
            and observation.host_peak_bytes >= 0
            and (limits.device_peak_bytes is None or observation.device_peak_bytes <= limits.device_peak_bytes)
            and (limits.host_peak_bytes is None or observation.host_peak_bytes <= limits.host_peak_bytes)
        )

    def observe(self, observation: StepObservation) -> JointPolicy:
        """Finish one full iteration; return the policy for the next one.

        Stable calls return immediately: no sampling, selection or re-planning.
        Ongoing workload validation belongs to the ordinary replay executor.
        """
        if self.stable:
            return self.policy
        started = time.perf_counter()
        try:
            self.steps += 1
            if self.steps > self.limits.max_steps:
                raise RuntimeError('tuning step budget exhausted without a qualified policy')
            feasible = self._feasible(observation)
            if self.phase in ('restore', 'confirm'):
                if not feasible:
                    raise RuntimeError(
                        'selected policy failed confirmation; training must stop or restore a checkpoint'
                    )
                if self.phase == 'restore':
                    self.phase = 'confirm'
                else:
                    self.confirmation.append(observation)
                    if len(self.confirmation) >= self.limits.confirmation_steps:
                        self.phase = 'stable'
                        self.frozen_at_step = self.steps
                return self.policy

            trial = self._trial
            trial.device_peak_bytes = max(trial.device_peak_bytes, observation.device_peak_bytes)
            trial.host_peak_bytes = max(trial.host_peak_bytes, observation.host_peak_bytes)
            trial.feasible &= feasible
            if not feasible:
                self._finish_trial()
            elif self.phase == 'warmup':
                self.phase = 'measure'
            else:
                trial.samples.append(observation.elapsed_seconds)
                if len(trial.samples) >= self.limits.samples_per_candidate:
                    self._finish_trial()
            return self.policy
        finally:
            self.search_seconds += time.perf_counter() - started

    def _finish_trial(self):
        self.trials.append(self._trial)
        # Preserve the earlier measured incumbent for near ties. With only a
        # few samples, a tiny timing difference is insufficient reason to
        # replace it. This is a configurable decision margin, not a statistical
        # confidence claim, and requires no extra tuning iterations.
        best = self.trials[0]
        for trial in self.trials[1:]:
            if trial.score < best.score * (1 - self.limits.minimum_relative_gain):
                best = trial
        if not math.isfinite(best.score):
            raise RuntimeError('no measured policy satisfies the memory and validity constraints')
        # Reserve one candidate warmup, its samples, a possible return-to-best
        # settling iteration and final confirmation before requesting any work.
        needed = 1 + self.limits.samples_per_candidate + 1 + self.limits.confirmation_steps
        can_search = (
            self.steps + needed <= self.limits.max_steps
            and len(self.trials) < self.limits.max_candidates
            and self.search_seconds < self.limits.max_search_seconds
        )
        candidate = None
        if can_search:
            started = time.perf_counter()
            candidate = self._propose(tuple(self.trials))
            proposal_seconds = time.perf_counter() - started
            if self.search_seconds + proposal_seconds >= self.limits.max_search_seconds:
                candidate = None
                self.stop_reason = 'search_time_budget'
            if candidate is not None and not isinstance(candidate, JointPolicy):
                raise TypeError('propose must return JointPolicy or None')
            if candidate is not None and any(candidate == trial.policy for trial in self.trials):
                candidate = None
                self.stop_reason = 'duplicate_proposal'
        if candidate is not None:
            self.policy = candidate
            self._trial = Trial(candidate)
            self.phase = 'warmup'
            return
        self.stop_reason = self.stop_reason or ('proposal_exhausted' if can_search else 'tuning_budget')
        self.phase = 'confirm' if self.policy == best.policy else 'restore'
        self.policy = best.policy
