from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.measured_regions import MeasuredRegionNeighborhood
from mindspeed.core.memory.hyper_offload.tuning.trace_regions import RegionCost, SavedVersion

MIB=1024**2
TARGET='mindspeed.core.memory.hyper_offload.tuning.measured_regions.region_costs'


def cost(name, ms, size=100*MIB, boundary=10*MIB, structure=('custom',), first=10, last=20):
    return RegionCost(name,structure,((1,8),),first,last,ms,(SavedVersion(first,first,100,size),),boundary,boundary,size)


def data(costs, capacity):
    return dict(trace=SimpleNamespace(d2h_bandwidth_gbps=10,h2d_bandwidth_gbps=10),
                report={'regions':[]}, names=[c.name for c in costs],device_limit_bytes=capacity,
                host_limit_bytes=10000*MIB,offloaded_checkpoint_inputs=(),conflicts=())


def initial(names=()):
    return Trial(JointPolicy(0,64,recompute_regions=names),samples=[10.])


def test_renamed_regions_choose_same_modes_and_never_read_family():
    results=[]
    for prefix in ('encoder.stage', 'unknown_branch'):
        costs=(cost(prefix+'.cheap',1),cost(prefix+'.expensive',100,first=30,last=40,structure=('other',)))
        evidence=data(costs,500*MIB)
        with patch(TARGET,return_value=costs):
            policy=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
        results.append((tuple(n.removeprefix(prefix) for n in policy.recompute_regions),
                        tuple(n.removeprefix(prefix) for n in policy.resident_regions)))
    assert results==[(('.cheap',),('.expensive',))]*2


def test_resident_admission_includes_input_and_output_boundary_storage():
    costs=(cost('opaque_id',100),)
    evidence=data(costs,110*MIB)
    with patch(TARGET,return_value=costs):
        policy=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
    assert not policy.resident_regions  # 100 internal + 20 boundary > 110


def test_conflicting_callable_groups_are_not_jointly_selected():
    costs=(cost('root.a',1),cost('root.b',2))
    evidence=data(costs,500*MIB);evidence['conflicts']=(('root.a','root.b'),)
    with patch(TARGET,return_value=costs):
        policy=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
    assert policy.recompute_regions==('root.a',)


def test_unknown_opaque_region_is_not_replaced_without_visible_equivalent():
    costs=(cost('visible.part',1),)
    evidence=data(costs,500*MIB)
    evidence['report']['regions']=[dict(name='opaque',complete=True,structure=('unknown',),input_shapes=())]
    evidence['names'].append('opaque')
    with patch(TARGET,return_value=costs):
        proposal=MeasuredRegionNeighborhood(lambda:evidence)
        policy=proposal((initial(('opaque',)),))
    assert 'opaque' in policy.recompute_regions and not proposal.diagnostics['expanded']


def test_provider_consumed_once_and_no_more_than_two_proposals():
    costs=(cost('region',1),);evidence=data(costs,500*MIB);calls=[]
    def provider():calls.append(1);return evidence
    proposal=MeasuredRegionNeighborhood(provider)
    with patch(TARGET,return_value=costs):
        start=initial();policy=proposal((start,))
        assert proposal((start,Trial(policy,samples=[9.]))) is None
    assert calls==[1]


def test_repeated_region_calls_require_aggregate_lifetime_evidence():
    evidence=data((),500*MIB)
    evidence['report']['regions']=[{'name':'shared'}, {'name':'shared'}]
    proposal=MeasuredRegionNeighborhood(lambda:evidence)
    assert proposal((initial(),)) is None
    assert 'shared' in proposal.diagnostics['skipped']


def test_zero_save_child_does_not_hide_useful_parent_checkpoint():
    costs=(cost('block',1),cost('block.activation',.1,size=0))
    evidence=data(costs,500*MIB)
    with patch(TARGET,return_value=costs):
        policy=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
    assert policy.recompute_regions==('block',)


def test_phase_capacity_admits_disjoint_peak_and_bounded_expansion_points():
    from mindspeed.core.memory.hyper_offload.ir.trace import TraceOp
    cheap=replace(cost('visible.cheap',1,first=3,last=4),saved_versions=(SavedVersion(3,3,12,100*MIB,12),))
    expensive=replace(cost('visible.expensive',1000,first=5,last=6,structure=('expensive',)),saved_versions=(SavedVersion(5,5,11,100*MIB,11),))
    whole=replace(cost('visible',200,first=2,last=6,structure=('whole',)),
                  saved_versions=cheap.saved_versions+expensive.saved_versions,
                  created_bytes=300*MIB,forward_save_window_ms=100,backward_save_window_ms=100,
                  working_set_bytes=200*MIB)
    costs=(whole,cheap,expensive);evidence=data(costs,1000*MIB)
    evidence['trace'].ops=[TraceOp('compute') for _ in range(20)]
    evidence['trace'].memory_limit_bytes=64*MIB
    regions={}
    for name,first,last in [('alpha',0,18),('beta',1,17)]:
        evidence['trace'].ops[first].name='checkpoint_fwd';evidence['trace'].ops[last].name='checkpoint_bwd'
        regions[name]=[dict(forward_op=first,backward_ops=[last])]
        evidence['names'].extend([name,name+'.cheap',name+'.expensive'])
        evidence['report']['regions'].append(dict(name=name,complete=True,structure=whole.structure,input_shapes=whole.input_shapes))
    evidence['phase_report']=dict(forward_end_op=6,truncated=False,regions=regions,
        samples=[(i,*([((990 if i==19 else 200)*MIB)]*4)) for i in range(20)])
    start=initial(('alpha','beta'));start.device_peak_bytes=990*MIB
    with patch(TARGET,return_value=costs):
        proposal=MeasuredRegionNeighborhood(lambda:evidence)
        first=proposal((start,));second=proposal((start,Trial(first,samples=[8.])))
    assert proposal.diagnostics['phase_capacity_used']
    assert len(proposal.diagnostics['expanded'])==2
    assert sum(n in first.recompute_regions for n in ('alpha','beta'))==1
    assert not set(('alpha','beta')).intersection(second.recompute_regions)
    assert proposal.diagnostics['phase_estimated_peak_bytes']<=1000*MIB


def test_queue_window_is_byte_weighted_deduplicated_and_capacity_derived():
    from mindspeed.core.memory.hyper_offload.tuning.measured_regions import _queue_window_scale
    large=cost('arbitrary.large',1,size=100*MIB)
    tiny=replace(cost('arbitrary.metadata',1),saved_versions=tuple(SavedVersion(i,50,100,MIB) for i in range(100,200)))
    # Tiny metadata dominate the packet count but not a strict byte majority.
    assert _queue_window_scale((large,tiny,large),800*MIB)==8.
    assert _queue_window_scale((large,tiny),400*MIB)==4.
    assert _queue_window_scale((large,),10*MIB)==1.
    assert _queue_window_scale((),800*MIB)==1.


def test_bounded_refinement_uses_measured_direction_and_noise_not_region_names():
    # Synthetic admitted paths isolate the bounded search decision from the
    # capacity model. A middle improvement explores further; noisy/slow or
    # infeasible trials retreat. No endpoint is automatically preferred.
    for samples, feasible, expected_count in (([7.9,8.1],True,6),
                                               ([9.,11.],True,2),
                                               ([11.,12.],True,2),
                                               ([],False,2)):
        start=initial(tuple('abcdefgh'))
        path=[]
        for count in range(1,9):
            remaining=tuple('abcdefgh'[count:])
            path.append((remaining,tuple('abcdefgh'[:count])))
        first=replace(start.policy,recompute_regions=path[3][0],resident_regions=path[3][1],transfer_window_scale=2.)
        search=MeasuredRegionNeighborhood(lambda:None)
        search._pending=[first]
        search._refinement=(start.policy,path,3,2.)
        middle=Trial(first,samples=samples,feasible=feasible)
        second=search((start,middle))
        assert len(second.resident_regions)==expected_count
        assert search((start,middle,Trial(second,samples=[6.]))) is None
        assert search.diagnostics['candidate_count']==2


def test_refinement_uses_configured_gain_margin():
    start=initial(('a','b','c','d'))
    path=[(tuple('abcd'[n:]),tuple('abcd'[:n])) for n in range(1,5)]
    first=replace(start.policy,recompute_regions=path[1][0],resident_regions=path[1][1])
    search=MeasuredRegionNeighborhood(lambda:None,minimum_relative_gain=.2)
    search._pending=[first];search._refinement=(start.policy,path,1,None)
    second=search((start,Trial(first,samples=[9.,9.])))
    assert len(second.resident_regions)==1  # 10% gain is below the requested 20%.


def test_trim_cadence_is_selected_from_cost_and_noise_without_model_identity():
    from mindspeed.core.memory.hyper_offload.tuning.measured_regions import measured_trim_interval
    start=initial()
    expensive=measured_trim_interval(start,[.4,.42],.005,1.)
    assert expensive['selected_interval']==2 and abs(expensive['estimated_saved_seconds']-.205)<1e-12
    assert measured_trim_interval(start,[.01,.02],.005,1.)['selected_interval']==1
    assert measured_trim_interval(start,[.01,.8],.005,1.)['selected_interval']==1
    for samples in ([],[.4],[.4,float('nan')],[.4,-1],[.4,100],[True,.4]):
        assert measured_trim_interval(start,samples,.005,1.)['selected_interval']==1
    for policy in (replace(start.policy,trim_before_replay=False),replace(start.policy,trim_replay_interval=3)):
        assert measured_trim_interval(replace(start,policy=policy),[.4,.4],.005,1.)['selected_interval']==policy.trim_replay_interval


def test_measured_trim_is_joint_with_renamed_fine_regions():
    for prefix in ('unknown_branch','encoder.stage'):
        costs=(cost(prefix+'.cheap',1),);evidence=data(costs,500*MIB)
        evidence['replay_trim_samples_seconds']=(.4,.4)
        with patch(TARGET,return_value=costs):
            search=MeasuredRegionNeighborhood(lambda:evidence);policy=search((initial(),))
        assert policy.recompute_regions==(prefix+'.cheap',) and policy.trim_replay_interval==1
        cadence=search((initial(),Trial(policy,samples=[8.,8.])))
        assert cadence==replace(policy,trim_replay_interval=2)
        assert search((initial(),Trial(policy,samples=[8.,8.]),Trial(cadence,samples=[9.,9.]))) is None
        assert search.diagnostics['replay_trim']['selected_interval']==2


def test_measured_size_latencies_change_cost_decision_without_region_identity():
    for prefix in ('unknown_branch','renamed.encoder'):
        costs=(cost(prefix+'.group',15.,size=100*MIB),)
        evidence=data(costs,500*MIB)
        with patch(TARGET,return_value=costs):
            raw=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
        assert raw.recompute_regions==(prefix+'.group',)
        evidence['trace'].transfer_cost_samples=((100*MIB,5.,5.),)
        with patch(TARGET,return_value=costs):
            calibrated=MeasuredRegionNeighborhood(lambda:evidence)((initial(),))
        assert calibrated.resident_regions==(prefix+'.group',)


def test_joint_refinement_combines_admitted_cut_and_cadence_with_two_candidates():
    for prefix in ('opaque_encoder.', 'other_branch.'):
        names=tuple(prefix+name for name in 'abcdefgh')
        start=initial(names)
        path=[(names[count:],names[:count]) for count in range(1,9)]
        first=replace(start.policy,recompute_regions=path[3][0],resident_regions=path[3][1])
        for enabled in (False,True):
            search=MeasuredRegionNeighborhood(lambda:None,joint_refinement=enabled)
            search._pending=[first];search._refinement=(start.policy,path,3,None);search._trim_interval=2
            measured=Trial(first,samples=[7.9,8.1])
            second=search((start,measured))
            assert second.trim_replay_interval==2
            assert len(second.resident_regions)==(6 if enabled else 4)
            assert second.recompute_regions==names[6 if enabled else 4:]
            assert search((start,measured,Trial(second,samples=[7.,7.]))) is None
            assert search.diagnostics['candidate_count']==2


def test_joint_refinement_does_not_add_cadence_when_first_candidate_is_not_better():
    names=tuple('abcdefgh');start=initial(names)
    path=[(names[count:],names[:count]) for count in range(1,9)]
    first=replace(start.policy,recompute_regions=path[3][0],resident_regions=path[3][1])
    for samples,feasible in (([9.,11.],True),([11.,12.],True),([],False)):
        search=MeasuredRegionNeighborhood(lambda:None,joint_refinement=True)
        search._pending=[first];search._refinement=(start.policy,path,3,None);search._trim_interval=2
        measured=Trial(first,samples=samples,feasible=feasible)
        second=search((start,measured))
        assert len(second.resident_regions)==2 and second.trim_replay_interval==1
        assert search((start,measured,Trial(second,samples=[8.,8.]))) is None


def test_joint_refinement_keeps_cadence_opportunity_when_no_further_cut_exists():
    start=initial(('opaque',));first=replace(start.policy,recompute_regions=(),resident_regions=('opaque',))
    search=MeasuredRegionNeighborhood(lambda:None,joint_refinement=True)
    search._pending=[first];search._refinement=(start.policy,[((),('opaque',))],0,None);search._trim_interval=2
    measured=Trial(first,samples=[8.,8.]);second=search((start,measured))
    assert second==replace(first,trim_replay_interval=2)
    assert search((start,measured,Trial(second,samples=[7.,7.]))) is None
    for value in (1,None,'yes'):
        with pytest.raises(ValueError,match='joint_refinement'):
            MeasuredRegionNeighborhood(lambda:None,joint_refinement=value)
