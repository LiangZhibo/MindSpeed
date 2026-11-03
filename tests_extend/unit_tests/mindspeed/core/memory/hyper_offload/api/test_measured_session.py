"""Integrated measured session: exact numerics, finite hooks and failure cleanup."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, StepObservation, TuningLimits
from mindspeed.core.memory.hyper_offload.tuning.measured_session import MeasuredOffloadSession
from mindspeed.core.memory.hyper_offload.tuning.regions import NamedRecomputeRegions


@contextmanager
def cpu_runtime():
    with ExitStack() as stack:
        timer=stack.enter_context(patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer'))
        timer.return_value.stop.return_value=1.0
        stack.enter_context(patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',return_value=(16.,16.)))
        stack.enter_context(patch('torch.Stream'))
        stack.enter_context(patch('torch.accelerator.is_available',return_value=False))
        stack.enter_context(patch('torch.accelerator.current_stream'))
        events=stack.enter_context(patch('torch.Event'))
        events.return_value.elapsed_time.return_value=1.0
        stats={k:0 for k in ('allocated_bytes.all.current','allocated_bytes.all.peak','reserved_bytes.all.current','reserved_bytes.all.peak')}
        stack.enter_context(patch('torch.get_device_module',return_value=SimpleNamespace(memory_stats=lambda:stats)))
        yield


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=torch.nn.Linear(16,16)
        self.dropout=torch.nn.Dropout(.2)
    def forward(self,x):
        return x+self.dropout(torch.sin(self.projection(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stages=torch.nn.ModuleList([Block() for _ in range(4)])
    def forward(self,x):
        for stage in self.stages:x=stage(x)
        return x.square().mean()


def setup(model):
    regions=NamedRecomputeRegions(model)
    for index in range(4):regions.register_module(f'stages.{index}')
    initial=JointPolicy(0,64,recompute_regions=('stages.0',))
    limits=TuningLimits(device_peak_bytes=1024**3,host_peak_bytes=64*1024**2)
    return regions,initial,limits


@pytest.mark.parametrize("joint_refinement",[False,True])
@pytest.mark.parametrize("measured_trim",[0.,.4])
@pytest.mark.parametrize("size_calibration",[False,True])
def test_integrated_session_preserves_full_updates_rng_and_stops_observing(measured_trim,size_calibration,joint_refinement):
    torch.manual_seed(21)
    reference=Model();model=deepcopy(reference)
    regions,initial,limits=setup(model)
    limits=replace(limits,max_steps=12)
    reference_opt=torch.optim.SGD(reference.parameters(),lr=.001)
    opt=torch.optim.SGD(model.parameters(),lr=.001)
    inputs=torch.randn(3,16)
    frozen_counters=None
    frozen_cadence=[]
    with cpu_runtime():
        session=MeasuredOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64,
            transfer_size_calibration=size_calibration,joint_refinement=joint_refinement)
        for step in range(16):
            reference_opt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
            a=inputs.clone().requires_grad_();b=inputs.clone().requires_grad_()
            torch.manual_seed(100+step)
            expected=reference(a);expected.backward();expected_rng=torch.get_rng_state().clone()
            torch.manual_seed(100+step)
            with session:
                actual=model(b);actual.backward()
            assert torch.equal(torch.get_rng_state(),expected_rng)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(b.grad,a.grad,rtol=0,atol=0)
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            reference_opt.step();opt.step()
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            del actual,expected,a,b
            assert session._root_handle is None and session.region_profiler.closed and session.phase_observer.closed
            assert regions.observer is None and session.phase_report['trace_signature']
            if not session.stable:
                session.last_trim_seconds=measured_trim  # Synthetic timing evidence; actual numerical replay above.
                session.observe(StepObservation(2. if session.executed_policy==initial else
                    (.5 if session.executed_policy.trim_replay_interval==2 else 1.),0,0))
            else:
                frozen_cadence.append(session.last_entry_trim_performed)
            counters=(session.controller.steps,session.proposal_compute_seconds,session.proposal_consensus_seconds,len(session.phase_report['samples']),len(session.timing_events),tuple(session.initial_trim_samples_seconds))
            if frozen_counters is not None:assert counters==frozen_counters
            if session.stable:
                frozen_counters=counters
                assert session.record_cache_reclaim is False
        assert session.controller.frozen_at_step<=12 and len(session.controller.trials)>=2
        assert session.trace_collections>=1  # A window-only candidate correctly reuses its trace.
        assert session.neighborhood.diagnostics['phase_capacity_used']
        assert session.initial_trim_samples_seconds==[measured_trim,measured_trim]
        assert session.controller.policy.trim_replay_interval==(2 if measured_trim else 1)
        if measured_trim:
            assert any(frozen_cadence) and not all(frozen_cadence)
            assert all(left!=right for left,right in zip(frozen_cadence,frozen_cadence[1:]))


def test_integrated_session_removes_all_observers_on_failed_forward():
    model=Model();regions,initial,limits=setup(model)
    with cpu_runtime():
        session=MeasuredOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64)
        with pytest.raises(RuntimeError,match='user forward failure'):
            with session:raise RuntimeError('user forward failure')
        assert session._root_handle is None and regions.observer is None
        assert session.phase_observer.closed and session.region_profiler.closed


def test_integrated_session_requires_explicit_capacity_and_hard_step_cap():
    model=Model();regions,initial,limits=setup(model)
    for invalid in (TuningLimits(),TuningLimits(max_steps=13,device_peak_bytes=1024**3)):
        with pytest.raises(ValueError,match='capacity and at most 12'):
            MeasuredOffloadSession(initial,model=model,regions=regions,limits=invalid)
    assert not regions.selected and not regions.resident and regions.observer is None
