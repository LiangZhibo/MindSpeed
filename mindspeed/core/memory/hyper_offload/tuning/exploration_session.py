"""Candidate-aware, step-bounded observation and generic structural search."""
from __future__ import annotations

from dataclasses import asdict, replace
import time

import torch
import torch.distributed as dist

from .controller import JointTuningController
from .capacity_exploration import CapacityFirstNeighborhood
from .phase_capacity import compact_phase_report, merge_phase_reports
from .phase_memory import PhaseMemoryObserver
from .profiling import RegionProfiler
from .peak_windows import PeakWindowModel
from .codec_calibration import ActivationSamples, CalibrationResult, calibrate_activations, calibration_capability
from .calibrated_transport import common_profiles
from .selective_codec import FrozenNodeCodec, codec_scopes
from .session import JointOffloadSession
from ..api.session import OffloadSession


class _ExplorationProfiler(RegionProfiler):
    def __init__(self, regions, phase_provider):
        super().__init__(regions)
        self._phase_provider = phase_provider

    def end(self, token, result):
        super().end(token, result)
        if token is not None:
            self._phase_provider().observe_callable(token[0].name, result)


class ExplorationOffloadSession(JointOffloadSession):
    """Explore several legal cuts, retire every observer before stable replay.

    A single complete model forward and its backward belong to each context.
    Only a new structure receives region/memory observation. Existing traces
    remain valid solely for the exact corresponding structure. Distributed
    observations must be reduced over ``process_group`` by the training loop.
    A failed training step is not recoverable by pretending it never occurred.
    """
    def __init__(self, initial, *, model, regions, limits, host_mb=65536,
                 process_group=None, transfer_size_calibration=True, beam_width=3,
                 codecs=None, enable_compression=True, compression_library=None, calibration_limits=None,
                 device_window_profile=None):
        if initial.recompute_regions is None or initial.codec_key != 'raw':
            raise ValueError('exploration requires named regions and an initial raw transport')
        if limits.device_peak_bytes is None or limits.max_steps > 60:
            raise ValueError('exploration requires a device capacity and at most 60 tuning steps')
        if not isinstance(enable_compression, bool):
            raise ValueError('enable_compression must be boolean')
        self.model, self.regions = model, regions
        self.initial_policy = initial
        self.process_group = process_group
        self._host_limit_bytes = host_mb * 1024**2
        self._observations, self._global_reports, self._queue_models = {}, {}, {}
        self._trim_samples = {}
        self._root_handle = self.region_profiler = self.phase_observer = None
        self.region_report = self.phase_report = self.global_phase_report = None
        self.proposal_compute_seconds = self.proposal_consensus_seconds = 0.0
        self.queue_analysis_seconds = 0.0
        self.observer_collections = 0
        self.initial_trim_samples_seconds = []
        self._compression_library, capability = calibration_capability(compression_library) if enable_compression else (None, 'disabled')
        self.compression_report = dict(status=capability)
        self._calibration_pending = enable_compression
        self._compression_samples = ActivationSamples(calibration_limits) if self._compression_library else None
        self._codec_profiles = ()
        self._codec_backend = None
        self._auto_scopes = {}
        self._announced_scopes = set()
        self._codec_keys_by_structure = {}
        self.calibration_consensus_seconds = 0.0
        self.calibration_host_peak_bytes = 0
        limits = replace(limits, timing_noise_multiplier=1.0, validate_confirmation_timing=True,
                         confirm_best_observed=True)
        options = dict(beam_width=beam_width, minimum_relative_gain=limits.minimum_relative_gain,
                       timing_noise_multiplier=limits.timing_noise_multiplier)
        if device_window_profile is None:
            self.neighborhood = CapacityFirstNeighborhood(self._evidence, **options)
        else:
            from .device_windows import DeviceWindowProfile
            from .device_window_search import DeviceWindowNeighborhood
            profile = DeviceWindowProfile(device_window_profile)
            if profile.device_mb != initial.resident_mb:
                raise ValueError('profile managed-memory budget differs from this run')
            required_names = set(profile.recompute + profile.resident) | {r.name for r in profile.groups}
            if not required_names <= set(regions.names):
                raise ValueError('profile anchor contains unregistered regions')
            initial = replace(initial, recompute_regions=profile.recompute, resident_regions=profile.resident,
                              transfer_window_scale=profile.window_scale,
                              shared_transfer_calendar=profile.prefer_shared_calendar)
            self.initial_policy = initial
            self.neighborhood = DeviceWindowNeighborhood(self._evidence, profile, **options)
        controller = JointTuningController(initial, self._propose, limits,
                                          search_time_reducer=self._synchronize_search_seconds)
        super().__init__(controller, apply_policy=regions.apply_policy, host_mb=host_mb,
                         transfer_size_calibration=transfer_size_calibration, codecs=codecs)

    def _finish_memory(self, report):
        self.phase_report = report

    def _finish_forward(self, model, args, output):
        self._root_handle.remove()
        self._root_handle = None
        self.region_report = self.region_profiler.finish_forward()
        self.phase_observer.finish_forward()

    def close_tuning_observers(self):
        if self._root_handle is not None:
            self._root_handle.remove()
            self._root_handle = None
        if self.region_profiler is not None:
            self.region_profiler.close()
        if self.phase_observer is not None:
            self.phase_observer.close()
        if self._compression_samples is not None:
            self._compression_samples.close()

    def _codec_for(self, policy):
        scope = self._auto_scopes.get(policy.codec_key)
        if scope is not None and scope.structure_key != policy.structure_key:
            raise ValueError('native codec nodes belong to a different execution structure')
        return super()._codec_for(policy)

    def __enter__(self):
        # Reject misuse before switching executors or installing fresh hooks.
        if self._dispatch_context is not None or OffloadSession.get_active() is not None:
            raise RuntimeError('nested joint offload contexts are unsupported')
        if self._cleanup_failed:
            raise RuntimeError('cannot reuse a joint session after cleanup failure')
        if self._awaiting_observation:
            raise RuntimeError('observe the completed training step before entering the next tuning step')
        if not self.stable and self.controller.policy != self.executed_policy:
            self._switch_policy(self.controller.policy)
        if not self.stable and self.mode == 'warmup':
            self.region_report = self.phase_report = None
            try:
                self.region_profiler = _ExplorationProfiler(self.regions, lambda: self.phase_observer).start()
                self.phase_observer = PhaseMemoryObserver(self, self.regions, on_finish=self._finish_memory)
                self._root_handle = self.model.register_forward_hook(self._finish_forward)
                if self._calibration_pending and self._compression_samples is not None:
                    host_limit = self.controller.limits.host_peak_bytes
                    self._compression_samples.install(self.residency_manager,
                        host_limit if host_limit is not None else self._host_limit_bytes)
            except BaseException:
                self.close_tuning_observers()
                raise
        try:
            return super().__enter__()
        except BaseException:
            self.close_tuning_observers()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        was_warmup = self.mode == 'warmup'
        try:
            result = super().__exit__(exc_type, exc_val, exc_tb)
            if exc_type is None and was_warmup:
                self._observations[self.executed_policy.structure_key] = (self.region_report, self.phase_report)
                self.observer_collections += 1
                if self._calibration_pending:
                    self._finish_calibration()
            return result
        finally:
            self.close_tuning_observers()

    def observe(self, observation):
        if self.stable:
            return self.controller.policy
        if self.controller.steps == 0:
            observation = replace(observation, host_peak_bytes=max(observation.host_peak_bytes,
                                                                  self.calibration_host_peak_bytes))
        if self._awaiting_observation and not self.stable and self.controller.phase == 'measure':
            samples = self._trim_samples.setdefault(self.executed_policy, [])
            if len(samples) < self.controller.limits.samples_per_candidate:
                samples.append(self.last_trim_seconds)
            if self.executed_policy == self.initial_policy:
                self.initial_trim_samples_seconds = samples
        result = super().observe(observation)
        if self.stable:
            # Reports remain available for explicit diagnostics, while heavy
            # queue/trace objects and any hooks stop participating in training.
            self.close_tuning_observers()
            self._queue_models.clear()
            self.neighborhood.retire_evidence()
            if self._codec_backend is not None:
                self.compression_report['runtime_stats_at_freeze'] = dict(self._codec_backend.stats)
            if self.executed_policy.codec_key == 'raw' and self._codec_backend is not None:
                self._codec_backend.close()
                for key in self._auto_scopes:
                    self._codecs.pop(key, None)
                self._codec_backend = None
        return result

    def _evidence(self, policy):
        plan = self._plans.get(policy)
        observed = self._observations.get(policy.structure_key)
        if plan is None or observed is None or observed[0] is None:
            return None
        host_limit = self.controller.limits.host_peak_bytes
        return dict(trace=plan.trace, report=observed[0], names=self.regions.names,
            conflicts=self.regions._conflicts,
            offloaded_checkpoint_inputs=self.regions.offloaded_checkpoint_inputs,
            phase_report=self._global_reports.get(policy.structure_key),
            replay_trim_samples_seconds=tuple(self._trim_samples.get(policy, ())),
            device_limit_bytes=self.controller.limits.device_peak_bytes,
            host_limit_bytes=host_limit if host_limit is not None else self._host_limit_bytes,
            codec_candidates=self._codec_keys_by_structure.get(policy.structure_key, ()),
            queue_summary=self._queue_models.get(policy.structure_key, (None, None))[1],
            queue_model=self._queue_models.get(policy.structure_key, (None, None))[0])

    def _finish_calibration(self):
        # This is still the excluded initial warmup, and its elapsed time and
        # device high-water remain part of the caller's full-step observation.
        result = CalibrationResult(self.compression_report)
        if self._compression_samples is not None:
            self._compression_samples.close()
            try:
                self.residency_manager.sync_all_transfers()
                host_limit = self.controller.limits.host_peak_bytes
                result = calibrate_activations(self._compression_samples,
                    library=self._compression_library, device=torch.accelerator.current_accelerator(),
                    device_limit=self.controller.limits.device_peak_bytes,
                    host_limit=host_limit if host_limit is not None else self._host_limit_bytes,
                    training_host_bytes=self.residency_manager._host_pool.total_allocated)
            finally:
                self._compression_samples.release()
        started = time.perf_counter()
        local = dict(profiles=result.profiles, status=result.report['status'],
                     host_peak=result.report.get('combined_host_peak_bytes', 0))
        reports = [local]
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size(self.process_group) > 1
        if distributed:
            reports = [None] * dist.get_world_size(self.process_group)
            dist.all_gather_object(reports, local, group=self.process_group)
        self._codec_profiles = common_profiles([row['profiles'] for row in reports])
        self.calibration_host_peak_bytes = max(row['host_peak'] for row in reports)
        self.calibration_consensus_seconds += time.perf_counter()-started
        self.compression_report = dict(result.report,
            rank_statuses=[row['status'] for row in reports],
            common_profiles=[asdict(profile) for profile in self._codec_profiles])
        if self._codec_profiles:
            if result.codebook_sample is None:
                raise RuntimeError('common native support requires a local verified codebook sample')
            from mindspeed.core.memory.compression import SplitZipCodec
            self._codec_backend = SplitZipCodec(self._compression_library,
                sample=result.codebook_sample, max_inflight=1)
        self._calibration_pending = False

    def _synchronize_search_seconds(self, local_seconds):
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size(self.process_group) > 1
        if not distributed:
            return local_seconds
        started = time.perf_counter()
        values = [None] * dist.get_world_size(self.process_group)
        dist.all_gather_object(values, local_seconds, group=self.process_group)
        self.proposal_consensus_seconds += time.perf_counter() - started
        return max(values)

    def _propose(self, trials):
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size(self.process_group) > 1
        rank = dist.get_rank(self.process_group) if distributed else 0
        for trial in trials:
            key = trial.policy.structure_key
            if key in self._global_reports:
                continue
            started = time.perf_counter()
            observed = self._observations.get(key)
            report = compact_phase_report(observed[1]) if observed is not None else None
            reports = [report]
            if distributed:
                reports = [None] * dist.get_world_size(self.process_group)
                dist.all_gather_object(reports, report, group=self.process_group)
            self._global_reports[key] = merge_phase_reports(reports)
            self.global_phase_report = self._global_reports[key]
            profile = getattr(self.neighborhood, 'profile', None)
            if profile is not None and key == self.initial_policy.structure_key and (
                    self.global_phase_report is None
                    or self.global_phase_report['trace_signature'] != profile.signature
                    or self.global_phase_report['capacity_summary']['op_count'] != profile.op_count):
                raise ValueError('device-window anchor does not match this training trace on all ranks')
            self.proposal_consensus_seconds += time.perf_counter() - started
            if rank == 0 and key not in self._queue_models and trial.policy in self._plans:
                started = time.perf_counter()
                model = PeakWindowModel(self._plans[trial.policy].trace)
                selected = model.default_selection()
                forecast = model.forecast(selected)
                summary = dict(opportunities=len(model.opportunities), selected=len(selected),
                    baseline_peak_bytes=forecast.baseline_peak_bytes,
                    forecast_peak_bytes=forecast.peak_bytes,
                    max_d2h_pressure_lateness_ms=forecast.max_d2h_pressure_lateness_ms,
                    max_h2d_lateness_ms=forecast.max_h2d_lateness_ms,
                    zero_wait_schedule_found=forecast.zero_wait_schedule_found,
                    useful_selected=len(forecast.useful_keys),
                    scope='fixed-graph shared-queue estimate; not actual runtime scheduling')
                self._queue_models[key] = (model, summary)
                self.queue_analysis_seconds += time.perf_counter() - started
            if (rank == 0 and key not in self._codec_keys_by_structure and key in self._queue_models
                    and self._global_reports[key] is not None):
                host_limit = self.controller.limits.host_peak_bytes
                scopes = codec_scopes(self._queue_models[key][0], self._codec_profiles, key,
                    prior_origins=profile.codec_origins if profile is not None and key == self.initial_policy.structure_key else (),
                    device_headroom=max(0, self.controller.limits.device_peak_bytes-trial.device_peak_bytes),
                    host_headroom=max(0, (host_limit if host_limit is not None else self._host_limit_bytes)-trial.host_peak_bytes))
                self._auto_scopes.update((scope.key, scope) for scope in scopes)
                self._codec_keys_by_structure[key] = tuple(scope.key for scope in scopes)
        started = time.perf_counter()
        proposal = self.neighborhood(trials) if rank == 0 else None
        self.proposal_compute_seconds += time.perf_counter() - started
        new_scopes = [scope for key, scope in self._auto_scopes.items() if key not in self._announced_scopes]
        if distributed:
            started = time.perf_counter()
            values = [(proposal, new_scopes) if rank == 0 else None]
            source = 0 if self.process_group is None else dist.get_global_rank(self.process_group, 0)
            dist.broadcast_object_list(values, src=source, group=self.process_group,
                                       device=torch.accelerator.current_accelerator())
            proposal, new_scopes = values[0]
            self.proposal_consensus_seconds += time.perf_counter() - started
        for scope in new_scopes:
            if self._codec_backend is None:
                raise RuntimeError('native codec scope received without common calibration support')
            if scope.key in self._codecs:
                raise ValueError('duplicate immutable native codec key')
            self._auto_scopes[scope.key] = scope
            self._codecs[scope.key] = FrozenNodeCodec(self._codec_backend, scope)
            self._announced_scopes.add(scope.key)
        self.compression_report['scopes'] = [asdict(scope) for scope in self._auto_scopes.values()]
        return proposal
