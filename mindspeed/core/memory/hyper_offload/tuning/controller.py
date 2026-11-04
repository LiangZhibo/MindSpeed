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
    codec_key: str = "raw"
    trim_replay_interval: int = 1
    shared_transfer_calendar: bool = False

    def __post_init__(self):
        if not isinstance(self.shared_transfer_calendar, bool):
            raise ValueError('shared_transfer_calendar must be a boolean')
        if self.shared_transfer_calendar and not self.masking:
            raise ValueError('the shared transfer calendar requires communication masking')
        if (isinstance(self.trim_replay_interval, bool) or not isinstance(self.trim_replay_interval, int)
                or self.trim_replay_interval < 1):
            raise ValueError('trim_replay_interval must be a positive integer')
        if not self.trim_before_replay and self.trim_replay_interval != 1:
            raise ValueError('periodic trimming requires trim_before_replay')
        if not isinstance(self.codec_key, str) or not self.codec_key.strip():
            raise ValueError("codec_key must be a nonempty immutable transport profile key")
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
    timing_noise_multiplier: float = 0.0
    validate_confirmation_timing: bool = False
    confirm_best_observed: bool = False

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
        if not math.isfinite(self.timing_noise_multiplier) or self.timing_noise_multiplier < 0:
            raise ValueError('timing_noise_multiplier must be finite and nonnegative')
        if not isinstance(self.validate_confirmation_timing, bool):
            raise ValueError('validate_confirmation_timing must be a boolean')
        if not isinstance(self.confirm_best_observed, bool):
            raise ValueError('confirm_best_observed must be a boolean')
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
    def balanced_samples(self) -> list[float]:
        """One mean per complete execution cycle, including expensive phases.

        Within-cycle trim/no-trim differences are expected work, not timing
        noise. A single cycle has no observed between-cycle dispersion; this
        is not a confidence claim and measured tuning still requires a new
        complete confirmation cycle before accepting a periodic winner.
        """
        cycle = self.policy.trim_replay_interval if self.policy.trim_before_replay else 1
        if cycle == 1:
            return self.samples
        if len(self.samples) % cycle:
            return []  # A cheap partial cycle cannot receive a finite score.
        return [math.fsum(self.samples[i:i+cycle]) / cycle
                for i in range(0,len(self.samples),cycle)]

    @property
    def score(self) -> float:
        values = self.balanced_samples
        return median(values) if self.feasible and values else math.inf

    @property
    def spread(self) -> float:
        score = self.score
        return median(abs(value-score) for value in self.balanced_samples) if math.isfinite(score) else math.inf


class JointTuningController:
    """Evaluate a few joint policies and irrevocably freeze within a step cap.

    ``propose`` receives completed trials and returns one next candidate or
    None. It must be a cheap, synchronous policy proposal, not a GPU benchmark.
    The same reduced observations must be supplied on all participating ranks.
    Distributed callers may also supply ``search_time_reducer`` to return the
    maximum elapsed search time across ranks before each budget decision.
    A failed training iteration itself must be handled by the training loop;
    this controller never treats a partially executed step as recoverable.
    """

    def __init__(
        self,
        initial: JointPolicy,
        propose: Callable[[tuple[Trial, ...]], JointPolicy | None],
        limits: TuningLimits | None = None,
        *,
        search_time_reducer: Callable[[float], float] | None = None,
    ):
        if search_time_reducer is not None and not callable(search_time_reducer):
            raise TypeError('search_time_reducer must be callable')
        self._search_time_reducer = search_time_reducer
        self.limits = limits or TuningLimits()
        self.policy = initial
        self.phase = 'warmup'
        self.steps = 0
        self.frozen_at_step: int | None = None
        self.search_seconds = 0.0
        self.stop_reason: str | None = None
        self.trials: list[Trial] = []
        self.confirmation: list[StepObservation] = []
        self.confirmation_target = self.limits.confirmation_steps
        self.confirmation_fallback = False
        self.confirmation_reference_index = 0
        self.selection_log: list[dict] = []
        self._trial = Trial(initial)
        self._propose = propose
        if 1 + self._measurement_steps(initial) + self._confirmation_steps(initial) > self.limits.max_steps:
            raise ValueError('step budget must cover complete initial execution cycles')

    @staticmethod
    def _cycle_steps(policy):
        return policy.trim_replay_interval if policy.trim_before_replay else 1

    def _measurement_steps(self, policy):
        cycle = self._cycle_steps(policy)
        return ((self.limits.samples_per_candidate + cycle - 1) // cycle) * cycle

    def _confirmation_steps(self, policy):
        cycle = self._cycle_steps(policy)
        return ((self.limits.confirmation_steps + cycle - 1) // cycle) * cycle

    def _search_steps(self, best, candidate=None):
        candidate = best if candidate is None else candidate
        periodic = max(self._cycle_steps(best), self._cycle_steps(candidate)) > 1
        if not periodic:
            return 1 + self._measurement_steps(candidate) + 1 + self._confirmation_steps(best)
        # Candidate wins: no restore, but a full periodic confirmation and a
        # possible fallback. Incumbent wins: restore it and confirm its own
        # cycle. These branches are mutually exclusive; do not charge both a
        # periodic winner's confirmation and an unnecessary winner restore.
        validate = self.limits.validate_confirmation_timing
        initial = self.trials[0].policy
        candidate_finish = self._confirmation_steps(candidate) + int(validate and candidate != initial)
        incumbent_confirmation = self._confirmation_steps(best)
        if validate and best != initial and self._cycle_steps(best) == 1:
            incumbent_confirmation = 1  # Existing nonperiodic budget contract.
        incumbent_finish = 1 + incumbent_confirmation + int(validate and best != initial)
        return 1 + self._measurement_steps(candidate) + max(candidate_finish, incumbent_finish)

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
            if self.phase in ('restore', 'confirm', 'fallback'):
                if not feasible:
                    raise RuntimeError(
                        'selected policy failed confirmation; training must stop or restore a checkpoint'
                    )
                if self.phase == 'fallback':
                    # This incumbent already has complete warmup and measured
                    # samples. A full restored step validates capacity before
                    # the hard deadline; no new search or trace is required.
                    self.selection_log.append(dict(step=self.steps,phase='fallback_validation',
                        elapsed_seconds=observation.elapsed_seconds,device_peak_bytes=observation.device_peak_bytes,
                        host_peak_bytes=observation.host_peak_bytes,accepted=True))
                    self.phase = 'stable'
                    self.frozen_at_step = self.steps
                elif self.phase == 'restore':
                    self.phase = 'confirm'
                else:
                    self.confirmation.append(observation)
                    if len(self.confirmation) >= self.confirmation_target:
                        baseline = self.trials[self.confirmation_reference_index]
                        measured = Trial(self.policy, samples=[x.elapsed_seconds for x in self.confirmation])
                        reject = (self.limits.validate_confirmation_timing and self.policy != baseline.policy
                                  and not self._beats(measured, baseline))
                        self.selection_log.append(dict(step=self.steps, phase='confirmation',
                            samples=list(measured.samples), score=measured.score, spread=measured.spread,
                            accepted=not reject))
                        if reject:
                            if self.steps >= self.limits.max_steps:
                                raise RuntimeError('no budget remains to validate the timing fallback')
                            self.policy = baseline.policy
                            self.confirmation_fallback = True
                            self.stop_reason = 'confirmation_timing_fallback'
                            self.phase = 'fallback'
                        else:
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
                if len(trial.samples) >= self._measurement_steps(trial.policy):
                    self._finish_trial()
            return self.policy
        finally:
            self.search_seconds += time.perf_counter() - started

    def _beats(self, challenger: Trial, incumbent: Trial) -> bool:
        if not math.isfinite(challenger.score):
            return False
        if not math.isfinite(incumbent.score):
            return True
        margin = incumbent.score * self.limits.minimum_relative_gain
        margin += self.limits.timing_noise_multiplier * (challenger.spread + incumbent.spread)
        # This is a bounded screening guard, not a confidence interval from
        # two samples. Independent stable-step runs establish final speed.
        return challenger.score < incumbent.score - margin

    def _budget_seconds(self, local_seconds):
        # Distributed callers must agree before entering a proposal collective
        # and before deciding whether to execute its returned candidate. Local
        # wall clocks alone can take different branches at the time boundary.
        if self._search_time_reducer is None:
            return local_seconds
        result = self._search_time_reducer(local_seconds)
        if (isinstance(result, bool) or not isinstance(result, (int, float))
                or not math.isfinite(result) or result < local_seconds):
            raise ValueError('search time reduction must return a finite upper bound')
        return result

    def _finish_trial(self):
        self.trials.append(self._trial)
        # Preserve the earlier measured incumbent for near ties. With only a
        # few samples, a tiny timing difference is insufficient reason to
        # replace it. This is a configurable decision margin, not a statistical
        # confidence claim, and requires no extra tuning iterations.
        best = self.trials[0]
        best_index, prior_best_index = 0, 0
        for index, trial in enumerate(self.trials[1:], 1):
            if self._beats(trial, best):
                prior_best_index, best_index = best_index, index
                best = trial
        if not math.isfinite(best.score):
            raise RuntimeError('no measured policy satisfies the memory and validity constraints')
        # Reserve one candidate warmup, its samples, a possible return-to-best
        # settling iteration and final confirmation before requesting any work.
        needed = self._search_steps(best.policy)
        used_seconds = self._budget_seconds(self.search_seconds)
        can_search = (
            self.steps + needed <= self.limits.max_steps
            and len(self.trials) < self.limits.max_candidates
            and used_seconds < self.limits.max_search_seconds
        )
        candidate = None
        if used_seconds >= self.limits.max_search_seconds:
            self.stop_reason = 'search_time_budget'
        if can_search:
            started = time.perf_counter()
            candidate = self._propose(tuple(self.trials))
            proposal_seconds = time.perf_counter() - started
            if self._budget_seconds(self.search_seconds + proposal_seconds) >= self.limits.max_search_seconds:
                candidate = None
                self.stop_reason = 'search_time_budget'
            if candidate is not None and not isinstance(candidate, JointPolicy):
                raise TypeError('propose must return JointPolicy or None')
            if candidate is not None and any(candidate == trial.policy for trial in self.trials):
                candidate = None
                self.stop_reason = 'duplicate_proposal'
            if candidate is not None and self.steps + self._search_steps(best.policy, candidate) > self.limits.max_steps:
                candidate = None
                self.stop_reason = 'periodic_cycle_budget'
        if candidate is not None:
            self.policy = candidate
            self._trial = Trial(candidate)
            self.phase = 'warmup'
            return
        self.stop_reason = self.stop_reason or ('proposal_exhausted' if can_search else 'tuning_budget')
        self.confirmation_reference_index = prior_best_index if self._cycle_steps(best.policy) > 1 else 0
        strict_best_index = best_index
        tentative = (min(self.trials, key=lambda trial:trial.score)
                     if self.limits.confirm_best_observed else self.trials[-1])
        tentative_restore = int(tentative.policy != self.policy)
        tentative_confirmation = (
            self.limits.validate_confirmation_timing
            and tentative is not best
            and (self.limits.confirm_best_observed or not tentative_restore)
            and tentative.score < best.score
            and (self.limits.confirm_best_observed or self._cycle_steps(best.policy) == 1)
            and self.steps + tentative_restore + self._confirmation_steps(tentative.policy) + 1
                <= self.limits.max_steps
        )
        if tentative_confirmation:
            # Spend one complete confirmation window on a lower-score
            # challenger. Exploration may revisit an earlier observation,
            # including the restore step in its bounded finish cost.
            # The same gain/noise margin still gates acceptance, and a full
            # validation step is reserved for the measured fallback policy.
            self.confirmation_reference_index = best_index
            best = tentative
        restoring = self.policy != best.policy
        self.phase = 'restore' if restoring else 'confirm'
        self.policy = best.policy
        self.confirmation_target = self._confirmation_steps(best.policy)
        if self.limits.validate_confirmation_timing and best.policy != self.trials[0].policy:
            # Reserve one complete restore/validation step for the already
            # measured initial policy if independent confirmation loses its
            # gain. Returning to an earlier candidate may leave one confirm
            # sample at the 12-step cap; record the actual target explicitly.
            available = self.limits.max_steps - self.steps - int(restoring) - 1
            if self._cycle_steps(best.policy) > 1 and available < self.confirmation_target:
                raise RuntimeError('step budget cannot confirm a complete periodic execution cycle')
            self.confirmation_target = min(self.confirmation_target, available)
            if self.confirmation_target < 1:
                raise RuntimeError('step budget cannot confirm the selected policy')
        self.selection_log.append(dict(step=self.steps, phase='selection',
            selected_index=next(i for i,t in enumerate(self.trials) if t is best),
            scores=[t.score if math.isfinite(t.score) else None for t in self.trials],
            spreads=[t.spread if math.isfinite(t.spread) else None for t in self.trials],
            confirmation_target=self.confirmation_target,
            confirmation_cycle_steps=self._cycle_steps(best.policy),
            confirmation_reference_index=self.confirmation_reference_index,
            strict_best_index=strict_best_index,
            tentative_confirmation=tentative_confirmation))
