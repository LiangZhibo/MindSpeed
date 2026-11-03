from dataclasses import replace
from unittest.mock import patch

import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, TraceOp, StorageAccess, AccessKind as K
from mindspeed.core.memory.hyper_offload.tuning.capacity_exploration import CapacityFirstNeighborhood
from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.exploration_regions import Proposal
from mindspeed.core.memory.hyper_offload.tuning.peak_windows import PeakWindowModel
from mindspeed.core.memory.hyper_offload.tuning.trace_regions import RegionCost, SavedVersion
from .test_exploration_regions import fixture as opaque_fixture, TARGET

MIB = 2**20


def visible_fixture(prefix='generic'):
    ops = [TraceOp('compute',10.) for _ in range(11)]
    sizes = {i:100*MIB for i in range(5)}
    rows,costs = [],[]
    for sid in range(4):
        name = f'{prefix}.part{sid}'
        ops[sid].accesses.append(StorageAccess(sid,sid,K.WRITE))
        ops[10-sid].accesses.append(StorageAccess(10-sid,sid,K.READ))
        cost = RegionCost(name,('arbitrary',),((1,),),sid,sid,10.,
            (SavedVersion(sid,sid,10-sid,100*MIB,10-sid),),10*MIB,0,100*MIB,
            working_set_bytes=100*MIB)
        costs.append(cost)
        rows.append(dict(name=name,complete=True,structure=cost.structure,input_shapes=cost.input_shapes,
                         input_storage_ids=()))
    ops[6].accesses.append(StorageAccess(6,4,K.WRITE))
    trace = ActivationTrace(ops,sizes,memory_limit_bytes=100*MIB,d2h_bandwidth_gbps=10.,
        h2d_bandwidth_gbps=10.,transfer_cost_samples=((100*MIB,10.,10.),))
    trial = Trial(JointPolicy(0,100,recompute_regions=()),samples=[10.,10.],host_peak_bytes=400*MIB)
    data = dict(trace=trace,report=dict(regions=rows),names=tuple(c.name for c in costs),conflicts=(),
        offloaded_checkpoint_inputs=(),device_limit_bytes=4096*MIB,host_limit_bytes=4096*MIB,
        phase_report=dict(forward_end_op=6,truncated=False,regions={},
            samples=[(i,1024*MIB,1024*MIB,1024*MIB,1024*MIB) for i in range(len(ops))]))
    data['queue_model'] = PeakWindowModel(trace)
    return trial,data,tuple(costs)


def reference(trial,data,costs):
    search = CapacityFirstNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):search._ingest(data,trial.policy.structure_key)
    return search,search._reference(trial,data)


@pytest.mark.parametrize('prefix',['unknown.encoder','renamed_network'])
def test_first_move_opens_all_capacity_admitted_regions_without_legacy_seed(prefix):
    trial,data,costs = opaque_fixture(prefix)
    ops = data['trace'].ops
    for op in ops:op.duration_ms=100.
    data['trace'] = ActivationTrace(ops,{},memory_limit_bytes=64*MIB,
                                   d2h_bandwidth_gbps=1.,h2d_bandwidth_gbps=1.)
    search = CapacityFirstNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs),patch.object(search._seed,'__call__',side_effect=AssertionError):
        policy = search((trial,))
    assert policy.recompute_regions == ()
    assert search.diagnostics['proposals'][0]['family']=='global_capacity_seed'
    assert len(search.diagnostics['proposals'][0]['changed_regions'])==2
    assert search.diagnostics['global_budgets'][0]['capacity']['shared_server_ms']==1920.


def test_global_allocation_batches_regions_and_preserves_early_saves_on_equal_cost():
    trial,data,costs = visible_fixture()
    search,ref = reference(trial,data,costs)
    assert len(ref['packets'])==4
    high,high_details = search._allocate(ref,.75)
    low,low_details = search._allocate(ref,.5)
    assert high.policy.recompute_regions==('generic.part3',)
    assert low.policy.recompute_regions==('generic.part2','generic.part3')
    assert high_details['realized_load']==.75 and low_details['realized_load']==.5
    assert low_details['remaining_reference_packets']==2
    # Keeping early saves raw is an operator-position tie break, not a model
    # name or layer-number rule.
    renamed = replace(costs[-1],name='totally_different')
    data['report']['regions'][-1]['name']=renamed.name
    data['names']=(*data['names'][:-1],renamed.name)
    other,other_ref=reference(trial,data,(*costs[:-1],renamed))
    assert other._allocate(other_ref,.75)[0].policy.recompute_regions==(renamed.name,)


def test_alias_snapshot_matches_all_forward_copies_of_the_saved_version():
    trial,data,costs = visible_fixture()
    data['trace'].ops[4].name='view'
    data['trace'].ops[4].accesses=[StorageAccess(4,0,K.READ),StorageAccess(4,0,K.WRITE)]
    data['queue_model']=PeakWindowModel(data['trace'])
    search,ref=reference(trial,data,costs)
    row=next(c for c in ref['choices'] if c[0].name=='generic.part0')
    assert row[1]==frozenset({(0,0),(0,4)})
    # A snapshot after the first backward use cannot remove this forward save.
    packet=ref['packets'][(0,4)]
    assert packet.candidate.copy_start.op_id<costs[0].saved_versions[0].first_backward_read


def test_shared_saved_versions_are_removed_only_once():
    trial,data,costs=visible_fixture()
    duplicate=replace(costs[-1],name='extra_alias',compute_ms=.1)
    data['names']=(*data['names'],duplicate.name)
    data['report']['regions'].append(dict(data['report']['regions'][-1],name=duplicate.name))
    search,ref=reference(trial,data,(*costs,duplicate))
    proposal,details=search._allocate(ref,.5)
    assert details['removed_peak_versions']==2 and details['estimated_transfer_bytes']==200*MIB
    assert len(proposal.policy.recompute_regions)==3  # Two choices share one version.


def test_no_peak_crossing_storage_cannot_be_priced_as_useful_offload():
    trial,data,costs=visible_fixture()
    data['trace'].memory_limit_bytes=500*MIB
    data['queue_model']=PeakWindowModel(data['trace'])
    search,ref=reference(trial,data,costs)
    assert ref['packets']=={} and ref['choices']==[]


def test_missing_phase_or_full_device_headroom_rejects_synthetic_cut():
    trial,data,costs=visible_fixture()
    data['device_limit_bytes']=1024*MIB
    search,ref=reference(trial,data,costs)
    assert search._allocate(ref,.5) is None
    data['phase_report']=None
    assert CapacityFirstNeighborhood(lambda policy:data)._capacity(data) is None


def test_checkpoint_input_cost_is_charged_against_shared_budget_and_host():
    trial,data,costs=visible_fixture()
    # A previously untransferred checkpoint input must not be treated as free.
    data['trace'].storage_sizes[8]=200*MIB
    data['offloaded_checkpoint_inputs']=(costs[-1].name,)
    data['report']['regions'][-1]['input_storage_ids']=(8,)
    search,ref=reference(trial,data,costs)
    assert costs[-1].name not in {c[0].name for c in ref['choices']}  # Negative transfer saving.
    proposal,details=search._allocate(ref,.75)
    assert proposal.policy.recompute_regions==(costs[-2].name,)


def test_unplaceable_region_does_not_poison_later_legal_combinations():
    trial,data,costs=visible_fixture()
    costs=(*costs[:-1],replace(costs[-1],input_boundary_bytes=8192*MIB))
    search,ref=reference(trial,data,costs)
    proposal,details=search._allocate(ref,.5)
    assert proposal.policy.recompute_regions==('generic.part1','generic.part2')
    assert details['removed_peak_versions']==2


def test_slow_widest_reference_survives_fastest_beam_pruning():
    trial,data,costs=visible_fixture()
    slow=replace(trial,samples=[20.,20.])
    fast=Trial(replace(trial.policy,recompute_regions=(costs[0].name,)),samples=[5.,5.])
    search=CapacityFirstNeighborhood(lambda policy:data)
    search._capacity_seed_done=True
    with patch(TARGET,return_value=costs):
        result=search((slow,fast))
    assert result is not None
    assert search.diagnostics['proposals'][0]['family']=='global_budget_cut'
    assert len(search._references)==2
    assert search._budget_policies[result][0]==slow.policy.structure_key


def test_timing_variants_do_not_displace_distinct_structures_from_refinement():
    initial,data,costs=visible_fixture('renamed_network')
    policy=replace(initial.policy,transfer_window_scale=1.)
    trials=[Trial(replace(policy,trim_replay_interval=cycle),samples=[5.+cycle/10]*cycle)
            for cycle in (1,2,3)]
    trials.extend(Trial(replace(policy,recompute_regions=(cost.name,)),samples=[6.+i]*2)
                  for i,cost in enumerate(costs[:3]))
    search=CapacityFirstNeighborhood(lambda _:data)
    search._capacity_seed_done=True
    visited=[]
    def record_parent(trial, _data):
        visited.append(trial.policy)
        return []
    with patch(TARGET,return_value=costs),patch.object(search,'_reference',return_value=None), \
            patch.object(search,'_refinements',side_effect=record_parent), \
            patch.object(search,'_residency_refinements',return_value=[]):
        assert search(trials) is None
    assert visited==[trials[0].policy,trials[3].policy,trials[4].policy]


def test_compression_refinement_visits_other_promising_structures_before_repeating_one():
    initial,data,costs=visible_fixture('unknown_model')
    data['codec_candidates']=('codec_a','codec_b')
    less_offload=replace(initial.policy,recompute_regions=tuple(c.name for c in costs[:2]),
                         transfer_window_scale=1.)
    more_offload=replace(less_offload,recompute_regions=(costs[0].name,))
    measured_codec=replace(less_offload,codec_key='codec_a')
    trials=[Trial(less_offload,samples=[5.,5.]),Trial(more_offload,samples=[5.2,5.2]),
            Trial(measured_codec,samples=[4.9,4.9])]
    search=CapacityFirstNeighborhood(lambda _:data)
    search._capacity_seed_done=True
    search._record(Proposal(measured_codec,less_offload,'compression',0.,()))
    # The next family-balanced opportunity is compression. One structure
    # has already received a codec trial; the other must get an opportunity
    # before another codec is tried on the same fastest structure.
    search._family_counts.update(trim=2,schedule=2,residency=2)
    with patch(TARGET,return_value=costs),patch.object(search,'_reference',return_value=None), \
            patch.object(search,'_residency_refinements',return_value=[]):
        proposed=search(trials)
    assert proposed==replace(more_offload,codec_key='codec_a')


def test_budget_refinement_brackets_both_sides_and_retires_graph_state():
    trial,data,costs=visible_fixture()
    search,ref=reference(trial,data,costs)
    proposal,details=search._allocate(ref,.75)
    key=trial.policy.structure_key
    search._attempted_fractions[key]={1.,.75,.5,.25}
    search._budget_policies[proposal.policy]=(key,details)
    measured=Trial(proposal.policy,samples=[8.,8.])
    assert set(search._fractions(ref,[trial,measured]))=={.625,.875}
    search.retire_evidence()
    assert not any((search._references,search._costs,search._templates,search._fronts,
                    search._budget_policies,search._attempted_fractions,
                    search._reopen_proposals,search._reopened_structures,search._observed_scopes))
    assert search.diagnostics['global_budgets']


def test_reference_itself_is_a_measured_budget_point_after_reopening():
    trial,data,costs=visible_fixture()
    search,ref=reference(trial,data,costs)
    proposal,details=search._allocate(ref,.5)
    key=trial.policy.structure_key
    search._attempted_fractions[key]={1.,.75,.5,.25}
    search._budget_policies[proposal.policy]=(key,details)
    slower=Trial(proposal.policy,samples=[12.,12.])
    # The reference at full budget is faster, so refine toward it. The sole
    # explicit budget-cut sample must not incorrectly become the incumbent.
    assert list(search._fractions(ref,[trial,slower]))==[.875]


def test_budget_changes_can_reopen_all_remaining_opaque_parents_in_one_move():
    initial,data,costs=opaque_fixture('unfamiliar_network')
    for op in data['trace'].ops:op.duration_ms=100.
    data['trace']=ActivationTrace(data['trace'].ops,{},memory_limit_bytes=64*MIB,
                                 d2h_bandwidth_gbps=1.,h2d_bandwidth_gbps=1.)
    # A measured fine cut still has two opaque parents retained from an
    # earlier capacity-limited seed. Those decisions are not permanent.
    cut=Trial(replace(initial.policy,recompute_regions=(*initial.policy.recompute_regions,costs[-1].name)),
              samples=[8.,8.])
    search=CapacityFirstNeighborhood(lambda policy:data)
    search._capacity_seed_done=True;search._round=3
    with patch(TARGET,return_value=costs):
        policy=search((initial,cut))
        assert policy.recompute_regions==(costs[-1].name,)
        proposal=search._metadata[policy]
        assert proposal.family=='global_capacity_reopen' and len(proposal.changed_regions)==2
        expanded=Trial(policy,samples=[7.,7.])
        search._ingest(data,policy.structure_key)
        refs=[search._reference(t,data) for t in (initial,cut,expanded)]
        front=search._reference_front(refs,(initial,cut,expanded))
        assert search._reference(expanded,data) in front
        assert search._reference(initial,data) in front  # Preserve prior wide evidence too.


def test_already_observed_parent_does_not_consume_another_discovery_move():
    trial,data,costs=opaque_fixture('generic')
    for op in data['trace'].ops:op.duration_ms=100.
    data['trace']=ActivationTrace(data['trace'].ops,{},memory_limit_bytes=64*MIB,
                                 d2h_bandwidth_gbps=1.,h2d_bandwidth_gbps=1.)
    search=CapacityFirstNeighborhood(lambda policy:data)
    observed=replace(costs[0],name=trial.policy.recompute_regions[0])
    with patch(TARGET,return_value=(*costs,observed)):search._ingest(data,('earlier_raw_graph',))
    with patch(TARGET,return_value=costs):search._ingest(data,trial.policy.structure_key)
    proposal=search._large_seed(trial,data,parents_only=True)
    assert proposal.changed_regions==(trial.policy.recompute_regions[1],)


def test_opaque_winner_keeps_its_budget_reference_instead_of_a_slower_visible_graph():
    wide,data,costs=visible_fixture()
    wide=replace(wide,samples=[20.,20.])
    initial=Trial(replace(wide.policy,recompute_regions=(costs[0].name,)),samples=[10.,10.])
    best=Trial(replace(wide.policy,recompute_regions=(costs[2].name,costs[3].name)),samples=[8.,8.])
    search,wide_ref=reference(wide,data,costs)
    with patch(TARGET,return_value=costs):search._ingest(data,initial.policy.structure_key)
    initial_ref=search._reference(initial,data)
    search._budget_policies[best.policy]=(wide.policy.structure_key,dict(fraction=.5))
    search._round=5
    assert search._reference_front([wide_ref,initial_ref],[initial,wide,best])==[wide_ref]


def test_residency_reserves_created_storage_instead_of_only_saved_bytes():
    trial,data,costs=visible_fixture()
    costly=tuple(replace(c,created_bytes=2048*MIB) for c in costs)
    data['device_limit_bytes']=2048*MIB
    search,_=reference(trial,data,costly)
    assert search._residency_refinements(trial,data)==[]
    data['device_limit_bytes']=8192*MIB
    proposals=search._residency_refinements(trial,data)
    assert proposals and proposals[0].policy.codec_key=='raw'
    assert len(proposals[0].policy.resident_regions)==3


def test_opaque_checkpoint_can_become_resident_using_prior_cost_and_current_lifetime():
    trial,data,costs=opaque_fixture('arbitrary_graph',capacity=700*MIB)
    search=CapacityFirstNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):search._ingest(data,('prior_visible_graph',))
    with patch(TARGET,return_value=()):search._ingest(data,trial.policy.structure_key)
    assert not search._costs[trial.policy.structure_key]
    proposals=search._opaque_keep_front(trial,data)
    assert proposals[0].policy.recompute_regions==()
    assert proposals[0].policy.resident_regions==trial.policy.recompute_regions
    assert proposals[0].estimated_saved_compute_ms==80.
    assert proposals[0].capacity_peak_bytes==640*MIB
    # Do not infer a current lifetime from a prior visible graph.
    data['phase_report']['regions'][trial.policy.recompute_regions[0]][0]['backward_ops']=[17,18]
    proposals=search._opaque_keep_front(trial,data)
    assert len(proposals[0].policy.resident_regions)==1
    data['device_limit_bytes']=390*MIB
    assert search._opaque_keep_front(trial,data)==[]
