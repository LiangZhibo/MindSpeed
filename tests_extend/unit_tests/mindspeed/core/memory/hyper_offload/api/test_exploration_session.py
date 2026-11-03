"""Full-step numerical/RNG and finite observation tests for new graph search."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, StepObservation, TuningLimits
from mindspeed.core.memory.hyper_offload.tuning.exploration_session import ExplorationOffloadSession
from mindspeed.core.memory.hyper_offload.tuning.regions import NamedRecomputeRegions

from .test_measured_session import Model, cpu_runtime, setup


@pytest.mark.parametrize('resident_start',[False,True])
def test_every_new_structure_is_observed_and_freeze_retires_hooks_and_search(resident_start):
    torch.manual_seed(21)
    reference=Model(); model=deepcopy(reference)
    regions,initial,limits=setup(model)
    if resident_start:
        initial=replace(initial,resident_regions=('stages.1',))
    # Guaranteed distinct legal cuts isolate lifecycle verification from
    # synthetic device cost estimates. All full steps execute real autograd.
    policies=[replace(initial,recompute_regions=('stages.1',),resident_regions=()),
              replace(initial,recompute_regions=(),resident_regions=()),
              replace(initial,recompute_regions=('stages.2','stages.3'),resident_regions=())]
    limits=replace(limits,max_steps=24,max_candidates=4,confirmation_steps=4,max_search_seconds=30.)
    aopt=torch.optim.SGD(reference.parameters(),lr=.001)
    bopt=torch.optim.SGD(model.parameters(),lr=.001)
    inputs=torch.randn(3,16)
    frozen=None
    def propose(trials):
        return policies[len(trials)-1] if len(trials)<=len(policies) else None
    with cpu_runtime(), patch('mindspeed.core.memory.hyper_offload.tuning.exploration_session.CapacityFirstNeighborhood.__call__',
                              side_effect=propose):
        session=ExplorationOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64)
        for step in range(25):
            aopt.zero_grad(set_to_none=True);bopt.zero_grad(set_to_none=True)
            a=inputs.clone().requires_grad_();b=inputs.clone().requires_grad_()
            torch.manual_seed(100+step)
            expected=reference(a);expected.backward();expected_rng=torch.get_rng_state().clone()
            torch.manual_seed(100+step)
            with session:
                actual=model(b);actual.backward()
            assert torch.equal(torch.get_rng_state(),expected_rng)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(b.grad,a.grad,rtol=0,atol=0)
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            aopt.step();bopt.step()
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            del expected,actual,a,b
            assert session._root_handle is None and regions.observer is None
            assert session.region_profiler.closed and session.phase_observer.closed
            if not session.stable:
                elapsed=10. if session.executed_policy==initial else 8.-policies.index(session.executed_policy)
                session.observe(StepObservation(elapsed,0,0))
            counters=(session.controller.steps,session.observer_collections,session.trace_collections,
                      session.proposal_compute_seconds,session.proposal_consensus_seconds,session.queue_analysis_seconds)
            if frozen is not None:assert frozen==counters
            if session.stable:frozen=counters
        assert session.observer_collections==session.trace_collections==4
        assert len(session._observations)==4
        assert session.controller.frozen_at_step<=24
        assert session.executed_policy==policies[-1]
        assert not session._queue_models and not session._plans


def test_nested_entry_and_missing_observation_do_not_replace_live_hooks():
    model=Model();regions,initial,limits=setup(model)
    with cpu_runtime():
        session=ExplorationOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64)
        with session:
            observer=session.region_profiler
            with pytest.raises(RuntimeError,match='nested'):session.__enter__()
            assert session.region_profiler is observer and not observer.closed
            loss=model(torch.randn(3,16,requires_grad=True));loss.backward()
        del loss
        with pytest.raises(RuntimeError,match='observe the completed'):session.__enter__()
        assert session.region_profiler.closed and session._root_handle is None


def test_failure_cleans_candidate_observers():
    model=Model();regions,initial,limits=setup(model)
    with cpu_runtime():
        session=ExplorationOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64)
        with pytest.raises(RuntimeError,match='failed forward'):
            with session:raise RuntimeError('failed forward')
        assert session.region_profiler.closed and session.phase_observer.closed
        assert session._root_handle is None and regions.observer is None
        assert session.observer_collections==0


@pytest.mark.parametrize('offloaded_inputs',[False,True])
def test_callable_lifetime_records_real_opaque_backward_not_previous_op(offloaded_inputs):
    class CallableModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(8,8)
        def forward(self,x):return self.call(self.linear(x)).square().sum()
    model=CallableModel();regions=NamedRecomputeRegions(model)
    model.call=regions.register_callable('arbitrary_group',lambda x:torch.sin(x)*x,
                                        offload_checkpoint_inputs=offloaded_inputs)
    initial=JointPolicy(0,64,recompute_regions=('arbitrary_group',))
    limits=TuningLimits(max_steps=20,device_peak_bytes=1024**3,max_search_seconds=30.)
    with cpu_runtime():
        session=ExplorationOffloadSession(initial,model=model,regions=regions,limits=limits,host_mb=64)
        with session:
            loss=model(torch.randn(3,8,requires_grad=True));loss.backward()
        del loss
        plan=session._plans[initial]
        row=session.phase_report['regions']['arbitrary_group'][0]
        assert plan.trace.ops[row['forward_op']].name.endswith('_fwd')
        assert len(row['backward_ops'])==1
        assert plan.trace.ops[row['backward_ops'][0]].name.endswith('_bwd')
        assert row['backward_ops'][0]>row['forward_op']
