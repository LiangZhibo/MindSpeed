from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import TraceOp
from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.exploration_regions import ExplorationNeighborhood
from mindspeed.core.memory.hyper_offload.tuning.trace_regions import RegionCost, SavedVersion

MIB = 1024**2
TARGET = 'mindspeed.core.memory.hyper_offload.tuning.exploration_regions.region_costs'


def fixture(prefix='arbitrary', capacity=2048*MIB):
    names = tuple(prefix + '.' + x for x in ('early', 'later'))
    visible = prefix + '.visible'
    saved = SavedVersion(1, 3, 14, 100*MIB, 14)
    child = RegionCost(visible+'.child', ('elementwise',), ((1, 8),), 3, 4,
                       1., (saved,), 10*MIB, 10*MIB, 100*MIB, 1., 1., 100*MIB)
    whole = replace(child, name=visible, structure=('block',), first_op=2,
                    last_op=6, compute_ms=40., created_bytes=200*MIB, working_set_bytes=120*MIB)
    ops = [TraceOp('compute', duration_ms=1.) for _ in range(20)]
    regions, rows = {}, []
    for name, first, last in ((names[0], 0, 18), (names[1], 1, 17)):
        ops[first].name='scope_fwd'; ops[last].name='scope_bwd'
        regions[name] = [dict(forward_op=first, backward_ops=[last])]
        rows.append(dict(name=name, complete=True, structure=whole.structure, input_shapes=whole.input_shapes))
    policy = JointPolicy(0, 64, recompute_regions=names)
    trial = Trial(policy, samples=[10., 10.], device_peak_bytes=200*MIB)
    trace = SimpleNamespace(ops=ops, memory_limit_bytes=64*MIB, d2h_bandwidth_gbps=.1,
                            h2d_bandwidth_gbps=.1, transfer_cost_samples=())
    data = dict(trace=trace, report=dict(regions=rows),
        names=(*names, *(n+'.child' for n in names), visible, child.name),
        device_limit_bytes=capacity, host_limit_bytes=16*1024**3, conflicts=(),
        offloaded_checkpoint_inputs=(), replay_trim_samples_seconds=(.2, .2),
        phase_report=dict(forward_end_op=6, truncated=False, regions=regions,
            samples=[(i, 200*MIB, 200*MIB, 200*MIB, 200*MIB) for i in range(20)]))
    return trial, data, (whole, child)


@pytest.mark.parametrize('prefix', ['alien.encoder', 'completely_renamed'])
def test_total_transfer_is_not_a_gate_on_raw_expansion_and_early_region_is_retained(prefix):
    trial, data, costs = fixture(prefix)
    search=ExplorationNeighborhood(lambda policy: data)
    search._seed_done=True
    with patch(TARGET, return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        proposal=search._expansion_front(trial,data,False,'early')[0]
    # Total DMA >2s per region, versus only40ms saved compute. It still gets
    # a real trial rather than being declared slower by an additive model.
    assert proposal.policy.recompute_regions == ()
    assert proposal.family=='expand_raw'
    assert proposal.changed_regions==trial.policy.recompute_regions
    assert proposal.parent.structure_key!=proposal.policy.structure_key


def test_raw_endpoint_and_cheap_bracket_both_receive_trials():
    trial, data, costs=fixture()
    search=ExplorationNeighborhood(lambda policy: data);search._seed_done=True
    trials=[trial]
    with patch(TARGET, return_value=costs):
        for _ in range(5):
            policy=search(tuple(trials))
            if policy is None:break
            trials.append(Trial(policy,samples=[20.,20.]))
    assert any(t.policy.recompute_regions==() for t in trials)
    assert any(t.policy.recompute_regions==tuple(n+'.child' for n in trial.policy.recompute_regions) for t in trials)
    assert {'expand_raw','expand_cheap','schedule'} <= {p['family'] for p in search.diagnostics['proposals']}


def test_same_capacity_rejects_native_cut_but_preserves_raw_candidate():
    trial,data,costs=fixture(capacity=390*MIB)
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        raw=search._expansion_front(trial,data,False,'early')
        cheap=search._expansion_front(trial,data,True,'early')
    assert raw and raw[0].policy.recompute_regions==()
    assert not cheap  # Shared pool184MiB + native10MiB exceeds390MiB.


def test_missing_equivalence_or_lifetime_never_removes_unknown_checkpoint():
    trial,data,costs=fixture()
    data['report']['regions'][0]['structure']=('unknown',)
    data['phase_report']['regions'][trial.policy.recompute_regions[1]][0]['backward_ops']=[16,17]
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=costs):
        proposal=search((trial,))
    assert proposal.recompute_regions==trial.policy.recompute_regions
    assert search.diagnostics['proposals'][0]['family']=='schedule'


def test_conflicting_inferred_children_are_not_applied():
    trial,data,costs=fixture()
    data['conflicts']=((trial.policy.recompute_regions[0]+'.child',trial.policy.recompute_regions[1]),)
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        front=search._expansion_front(trial,data,True,'early')
    assert all(not set(data['conflicts'][0])<=set(p.policy.recompute_regions) for p in front)


def test_repeated_observations_disable_inferred_expansion():
    trial,data,costs=fixture()
    data['report']['regions'].append(dict(data['report']['regions'][0]))
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=costs):
        proposal=search((trial,))
    assert proposal.recompute_regions==trial.policy.recompute_regions
    assert search.diagnostics['exclusions'][0]['reason']=='repeated_or_incomplete_regions'


def test_late_codec_profile_does_not_disappear_in_cached_structural_front():
    trial,data,costs=fixture()
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=costs):
        proposal=search((trial,))
        data['codec_candidates']=('calibrated',)
        search._family_counts.update(expand_raw=5,expand_cheap=5,schedule=5,refine_regions=5,trim=5)
        codec=search((trial,Trial(proposal,samples=[20.,20.])))
    assert codec.codec_key=='calibrated'
    assert codec.structure_key==trial.policy.structure_key


def test_freeze_drops_graph_references_without_erasing_audit():
    trial,data,costs=fixture()
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=costs):search((trial,))
    search.retire_evidence()
    assert not search._costs and not search._fronts and not search._templates
    assert search.diagnostics['proposals'] and search.diagnostics['convergence']=='not_established'


def test_one_inconsistent_child_does_not_erase_all_common_fine_alternatives():
    trial,data,costs=fixture()
    whole,child=costs
    other=replace(whole,name='another_visible')
    other_child=replace(child,name='another_visible.child',compute_ms=3.)
    optional=replace(child,name=whole.name+'.optional',structure=('optional',),compute_ms=.2)
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=(*costs,other,other_child,optional)):
        search._ingest(data,trial.policy.structure_key)
    template=search._template(data['report']['regions'][0])
    assert tuple(row[0] for row in template.cheap_cut)==('.child',)
    assert template.cheap_cut[0][1]==2.  # Median compute, local lifetimes unchanged.


def test_structural_move_drops_old_graph_codec_binding():
    trial,data,costs=fixture()
    trial=replace(trial,policy=replace(trial.policy,codec_key='old-graph-profile'))
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        front=search._expansion_front(trial,data,True,'early')
    assert front and all(p.policy.codec_key=='raw' for p in front)


@pytest.mark.parametrize('prefix', ['opaque_network', 'renamed.graph'])
def test_measured_peer_cut_preserves_a_child_above_isolated_transfer_threshold(prefix):
    trial,data,costs=fixture(prefix)
    whole,child=costs
    whole=replace(whole,compute_ms=100.)
    expensive=replace(child,name=whole.name+'.second',structure=('different_compute',),compute_ms=30.)
    data['trace'].d2h_bandwidth_gbps=data['trace'].h2d_bandwidth_gbps=100.
    original=trial.policy.recompute_regions
    trial=replace(trial,policy=replace(trial.policy,recompute_regions=(*original,child.name,expensive.name)))
    data['names']=(*data['names'],expensive.name,*(name+'.second' for name in original))
    data['report']['regions'].append(dict(name=whole.name,complete=True,
        structure=whole.structure,input_shapes=whole.input_shapes))
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    with patch(TARGET,return_value=(whole,child,expensive)):
        search._ingest(data,trial.policy.structure_key)
        cheap=search._expansion_front(trial,data,True,'early')[0]
        retained=search._expansion_front(trial,data,False,'early',retained=True)[0]
    assert retained.family=='expand_retained'
    assert all(name+'.second' in retained.policy.recompute_regions for name in original)
    assert all(name+'.second' not in cheap.policy.recompute_regions for name in original)
    assert retained.estimated_saved_compute_ms == 138.
    assert retained.parent.structure_key!=retained.policy.structure_key
    # A resident child without observed capacity evidence cannot be inferred
    # or silently translated into checkpoint inputs.
    mixed=replace(trial,policy=replace(trial.policy,resident_regions=(whole.name+'.keep',)))
    assert search._expansion_front(mixed,data,False,'early',retained=True)==[]


def test_unobserved_child_in_measured_peer_cut_is_not_inferred():
    trial,data,costs=fixture()
    whole,child=costs
    trial=replace(trial,policy=replace(trial.policy,
        recompute_regions=(*trial.policy.recompute_regions,whole.name+'.unknown')))
    data['report']['regions'].append(dict(name=whole.name,complete=True,
        structure=whole.structure,input_shapes=whole.input_shapes))
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        assert search._expansion_front(trial,data,False,'early',retained=True)==[]


def test_release_is_independent_and_new_trim_parent_does_not_starve_other_axes():
    trial,data,costs=fixture()
    search=ExplorationNeighborhood(lambda policy:data);search._seed_done=True
    search._family_counts.update(expand_retained=5,expand_value=5,expand_cheap=5,
        expand_raw=5,refine_regions=5,compression=5,trim=5)
    with patch(TARGET,return_value=costs):
        first=search((trial,))
        assert first.defer_h2d_release and not first.prefetch_on_copy_stream
        faster_trim=Trial(replace(trial.policy,trim_replay_interval=2),samples=[9.,9.])
        second=search((trial,Trial(first,samples=[12.,12.]),faster_trim))
    assert second.prefetch_on_copy_stream and not second.defer_h2d_release
    assert [row['refinement_axis'] for row in search.diagnostics['proposals']]==[
        'defer_release','prefetch_allocation']


def test_new_structure_can_reverse_inherited_release_and_allocation_flags():
    trial,data,costs=fixture()
    trial=replace(trial,policy=replace(trial.policy,prefetch_on_copy_stream=True,defer_h2d_release=True))
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):
        search._ingest(data,trial.policy.structure_key)
        policies=[p.policy for p in search._refinements(trial,data)]
    assert {(False,True),(True,False),(False,False)} <= {
        (p.prefetch_on_copy_stream,p.defer_h2d_release) for p in policies}


def test_opaque_only_incumbent_uses_prior_templates_but_requires_its_own_valid_phase():
    trial,data,costs=fixture()
    whole,child=costs
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):search._ingest(data,trial.policy.structure_key)
    measured=replace(trial,policy=replace(trial.policy,
        recompute_regions=(*trial.policy.recompute_regions,child.name),transfer_window_scale=7.))
    data['report']['regions'].append(dict(name=whole.name,complete=True,
        structure=whole.structure,input_shapes=whole.input_shapes))
    with patch(TARGET,return_value=()):search._ingest(data,measured.policy.structure_key)
    assert search._costs[measured.policy.structure_key]==()
    front=search._expansion_front(measured,data,False,'early',retained=True)
    assert front and front[0].policy.recompute_regions==tuple(sorted([
        *(n+'.child' for n in trial.policy.recompute_regions),child.name]))
    assert front[0].policy.transfer_window_scale==7.
    # Pooled compute evidence never substitutes for candidate-local capacity.
    assert search._expansion_front(measured,dict(data,phase_report=None),False,'early',retained=True)==[]
    unknown=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=()):unknown._ingest(data,measured.policy.structure_key)
    assert unknown._expansion_front(measured,data,False,'early',retained=True)==[]
    # Repeated scopes are invalid evidence even when a previous graph supplied
    # an otherwise matching template.
    invalid=replace(measured,policy=replace(measured.policy,recompute_regions=(trial.policy.recompute_regions[0],)))
    data['report']['regions'].append(dict(data['report']['regions'][0]))
    with patch(TARGET,return_value=()):search._ingest(data,invalid.policy.structure_key)
    assert search._expansion_front(invalid,data,False,'early',retained=True)==[]


def test_measured_mixed_cut_preserves_residency_and_charges_full_native_bounds():
    trial,data,costs=fixture()
    whole,child=costs
    kept=replace(child,name=whole.name+'.keep',structure=('kept_computation',),compute_ms=30.)
    originals=trial.policy.recompute_regions
    trial=replace(trial,policy=replace(trial.policy,recompute_regions=(*originals,child.name),
        resident_regions=(kept.name,)))
    data['names']=(*data['names'],kept.name,*(name+'.keep' for name in originals))
    data['report']['regions'].append(dict(name=whole.name,complete=True,
        structure=whole.structure,input_shapes=whole.input_shapes))
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=(whole,child,kept)):
        search._ingest(data,trial.policy.structure_key)
    front=search._expansion_front(trial,data,False,'early',retained=True)
    assert front
    expanded=front[0]
    assert expanded.policy.recompute_regions==tuple(sorted([child.name,*(n+'.child' for n in originals)]))
    assert expanded.policy.resident_regions==tuple(sorted([kept.name,*(n+'.keep' for n in originals)]))
    assert expanded.estimated_saved_compute_ms==78.  # Kept children need no recomputation.
    assert expanded.capacity_peak_bytes==644*MIB  # Base200 + shared184 + two*(input10+keep120).
    # Keeping is never charged as just a small checkpoint input. Even one
    # mixed expansion exceeds the lower limit; raw remains a separate option.
    constrained=dict(data,device_limit_bytes=500*MIB)
    assert search._expansion_front(trial,constrained,False,'early',retained=True)==[]
    assert search._expansion_front(trial,constrained,False,'early')
    checkpointed=search._expansion_front(trial,constrained,False,'early',retained=True,checkpoint_kept=True)[0]
    assert checkpointed.family=='expand_checkpoint'
    assert checkpointed.policy.resident_regions==(kept.name,)
    assert all(name+'.keep' in checkpointed.policy.recompute_regions for name in originals)
    assert checkpointed.capacity_peak_bytes==424*MIB  # Base200 + shared184 + two*(input10+input10).
    assert checkpointed.estimated_saved_compute_ms==18.  # Both inferred children now recompute.


@pytest.mark.parametrize('prefix',['unknown_architecture','renamed.network'])
def test_resident_decisions_can_be_reopened_without_assuming_reclaimed_native_memory(prefix):
    trial,data,costs=fixture(prefix)
    names=trial.policy.recompute_regions
    trial=replace(trial,policy=replace(trial.policy,recompute_regions=(),resident_regions=names))
    search=ExplorationNeighborhood(lambda policy:data)
    with patch(TARGET,return_value=costs):search._ingest(data,trial.policy.structure_key)
    checkpoints=search._resident_front(trial,data,recompute=True)
    raw=search._resident_front(trial,data,recompute=False)
    assert checkpoints[0].policy.recompute_regions==names
    assert checkpoints[0].policy.resident_regions==()
    assert checkpoints[0].estimated_saved_compute_ms==-80.
    assert checkpoints[0].capacity_peak_bytes==440*MIB  #200base +220nativeworkspace +20inputs.
    assert raw[0].policy.recompute_regions==raw[0].policy.resident_regions==()
    assert raw[0].capacity_peak_bytes==384*MIB  #200base +64managed +120working set.
    limited=dict(data,device_limit_bytes=430*MIB)
    single=search._resident_front(trial,limited,recompute=True)[0]
    assert len(single.changed_regions)==1 and single.capacity_peak_bytes==430*MIB
    assert len(search._resident_front(trial,limited,recompute=False)[0].changed_regions)==2
    assert search._resident_front(trial,dict(data,phase_report=None),recompute=True)==[]
