"""Model-independent measured tuning with one-shot observation lifecycle.

Adapters provide legal regions; this session observes one full warmup, proposes
at most two candidates, and leaves ordinary fixed replay after the controller
freezes. Complete-step observations remain the training loop's responsibility.
"""
from __future__ import annotations

from dataclasses import replace
import time

import torch
import torch.distributed as dist

from .controller import JointPolicy, JointTuningController, TuningLimits
from .measured_regions import MeasuredRegionNeighborhood
from .phase_capacity import merge_phase_reports, compact_phase_report
from .phase_memory import PhaseMemoryObserver
from .profiling import RegionProfiler
from .session import JointOffloadSession


class MeasuredOffloadSession(JointOffloadSession):
    """Automatic raw-offload/recomputation tuning for an adapter-certified model.

    ``model`` must cover the complete forward whose backward runs in this
    session. Its first forward hook removes itself before recomputation.
    Distributed callers use one process group for proposals and must reduce
    full-step observations over that same group before calling ``observe``.
    A whole-device capacity is mandatory; unknown lifetime/structure evidence
    falls back to conservative proposals. No model identity is inspected.
    """
    def __init__(self, initial: JointPolicy, *, model, regions, limits: TuningLimits,
                 host_mb=65536, process_group=None, consider_replay_trim=True,
                 transfer_size_calibration=False, joint_refinement=False):
        if initial.recompute_regions is None or initial.codec_key != 'raw':
            raise ValueError('measured tuning requires named regions and an initial raw transport')
        if limits.device_peak_bytes is None or limits.max_steps > 12:
            raise ValueError('measured tuning requires a device capacity and at most 12 tuning steps')
        if not isinstance(consider_replay_trim, bool):
            raise ValueError('consider_replay_trim must be boolean')
        self.consider_replay_trim = consider_replay_trim
        self.initial_trim_samples_seconds = []
        self.initial_policy, self.regions = initial, regions
        self._host_limit_bytes = host_mb * 1024**2
        self.process_group = process_group
        self.region_report = None
        self.phase_report = None
        self.global_phase_report = None
        self._phase_consensus_done = False
        self.proposal_compute_seconds = 0.0
        self.proposal_consensus_seconds = 0.0
        self._root_handle = None
        self.region_profiler = None
        self.phase_observer = None
        limits = replace(limits,timing_noise_multiplier=1.0,validate_confirmation_timing=True)
        self.neighborhood = MeasuredRegionNeighborhood(self._evidence,
            minimum_relative_gain=limits.minimum_relative_gain,
            timing_noise_multiplier=limits.timing_noise_multiplier,
            joint_refinement=joint_refinement)
        controller = JointTuningController(initial,self._propose,limits)
        super().__init__(controller,apply_policy=regions.apply_policy,host_mb=host_mb,
                         transfer_size_calibration=transfer_size_calibration)
        try:
            self.region_profiler = RegionProfiler(regions).start()
            self.phase_observer = PhaseMemoryObserver(self,regions,on_finish=self._finish_memory)
            self._root_handle = model.register_forward_hook(self._finish_forward)
        except BaseException:
            self.close_tuning_observers()
            raise

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type,exc_val,exc_tb)
        finally:
            # Includes failures in forward, backward, trace finalization and
            # planning; hooks must never survive into subsequent user work.
            self.close_tuning_observers()

    def observe(self, observation):
        if (self.consider_replay_trim and self._awaiting_observation and not self.stable
                and self.controller.phase == 'measure'
                and self.executed_policy == self.initial_policy
                and len(self.initial_trim_samples_seconds) < self.controller.limits.samples_per_candidate):
            self.initial_trim_samples_seconds.append(self.last_trim_seconds)
        return super().observe(observation)

    def _evidence(self):
        plan = self._plans.get(self.initial_policy)
        if plan is None or self.region_report is None:
            return None
        host_limit = self.controller.limits.host_peak_bytes
        if host_limit is None:
            host_limit = self._host_limit_bytes
        return dict(trace=plan.trace,report=self.region_report,names=self.regions.names,
                    conflicts=self.regions._conflicts,
                    offloaded_checkpoint_inputs=self.regions.offloaded_checkpoint_inputs,
                    phase_report=self.global_phase_report,
                    replay_trim_samples_seconds=tuple(self.initial_trim_samples_seconds),
                    device_limit_bytes=self.controller.limits.device_peak_bytes,
                    host_limit_bytes=host_limit)

    def _propose(self, trials):
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size(self.process_group)>1
        rank = dist.get_rank(self.process_group) if distributed else 0
        if not self._phase_consensus_done:
            started = time.perf_counter()
            report = compact_phase_report(self.phase_report)
            reports = [report]
            if distributed:
                reports = [None] * dist.get_world_size(self.process_group)
                dist.all_gather_object(reports,report,group=self.process_group)
            self.global_phase_report = merge_phase_reports(reports)
            self._phase_consensus_done = True
            self.proposal_consensus_seconds += time.perf_counter()-started
        started = time.perf_counter()
        proposal = self.neighborhood(trials) if rank == 0 else None
        self.proposal_compute_seconds += time.perf_counter()-started
        if distributed:
            started = time.perf_counter()
            values = [proposal]
            source = 0 if self.process_group is None else dist.get_global_rank(self.process_group,0)
            dist.broadcast_object_list(values,src=source,group=self.process_group,
                                       device=torch.accelerator.current_accelerator())
            proposal = values[0]
            self.proposal_consensus_seconds += time.perf_counter()-started
        return proposal
