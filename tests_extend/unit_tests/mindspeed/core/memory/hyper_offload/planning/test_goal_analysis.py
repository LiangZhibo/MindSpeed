"""Statistical evidence must retain explicit steps, rank pairing and provenance."""
import importlib.util
from pathlib import Path

import pytest

root=next(p for p in Path(__file__).resolve().parents if (p/'tools/hyper_offload/audit_results.py').exists())
spec=importlib.util.spec_from_file_location('goal_analysis',root/'tools/hyper_offload/audit_results.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def result(offset=0):
    return dict(session_setup_seconds=2,extra_step_results=[dict(index=i,elapsed_seconds=float(i+offset),cumulative_step_wall_seconds=i*10.) for i in range(2,37)])


def test_explicit_training_steps_and_per_step_rank_max_before_median():
    a,b=result(),result()
    a['extra_step_results'].reverse()  # Serialization order is not the index.
    for x in b['extra_step_results']:
        if x['index']%2:x['elapsed_seconds']+=100
    summary=analysis.stable_summary([a,b])
    assert summary['samples']==24
    assert summary['maxrank_times'][0]==113 and summary['maxrank_times'][1]==14
    assert summary['median_seconds']==74.5
    b['extra_step_results'].append(b['extra_step_results'][0])
    with pytest.raises(ValueError,match='duplicate'):analysis.stable_summary([a,b])


def test_first12_matches_each_ranks_setup_and_cumulative_boundary():
    a,b=result(),result();b['session_setup_seconds']=5
    b['extra_step_results'][10]['cumulative_step_wall_seconds']=118
    assert analysis.first12_wall([a,b])==123  # max(2+120, 5+118), not 5+120.
    del b['session_setup_seconds']
    assert analysis.first12_wall([a,b]) is None
    b['extra_step_results']=[x for x in b['extra_step_results'] if x['index']!=12]
    with pytest.raises(ValueError,match='missing'):analysis.first12_wall([a,b])


def test_pairs_reject_copied_executions_and_mixed_workloads(monkeypatch):
    from copy import deepcopy
    order=['A1','B1','B2','A2','A3','B3']
    def run(label):
        return dict(path='/tmp/'+label,execution_identity=[label],started=order.index(label),source_sha256='same',
                    workload={'model':'same'},stable={'last_step':36,'median_seconds':10.},
                    ranks=[dict(allocator_capacity_pass=True,host_pool_capacity_pass=True,frozen_runtime_audit_pass=True)])
    data={label:run(label) for label in order}
    def compare(candidate,baseline):
        return dict(candidate=deepcopy(data[candidate]),reference=deepcopy(data[baseline]),final_state_exact=True,step_probes_exact=True,stable_ratio=1.)
    monkeypatch.setattr(analysis,'compare_runs',compare)
    manifest=dict(order=order,pairs=[dict(candidate='B'+str(i),baseline='A'+str(i)) for i in range(1,4)],
                  protocol=dict(statistics=dict(final_pairs=3,order=order,steps_per_run=36),
                                target=dict(stable_seconds=11.),generalization=dict(regression_target=.02)))
    assert analysis.paired_summary(manifest)['candidate_geomean_seconds']==pytest.approx(10.)
    data['B2']['execution_identity']=data['B1']['execution_identity']
    with pytest.raises(ValueError,match='copied'):analysis.paired_summary(manifest)
    data['B2']=run('B2');data['B2']['workload']={'model':'different'}
    with pytest.raises(ValueError,match='workload'):analysis.paired_summary(manifest)


def test_overhead_rejects_fixed_control_that_still_observes(monkeypatch):
    from copy import deepcopy
    common=dict(source_sha256='same',workload={'model':'same'},stable={'last_step':36,'median_seconds':11.},
                ranks=[dict(policy={'opaque':'same'})],trace_capture=True,direct_fixed_policy=False)
    rows={'auto':dict(deepcopy(common),first12_wall_seconds=120.),
          'fixed':dict(deepcopy(common),first12_wall_seconds=100.)}
    monkeypatch.setattr(analysis,'inspect_run',lambda path:deepcopy(rows[path]))
    monkeypatch.setattr(analysis,'compare_final_state',lambda a,b:True)
    monkeypatch.setattr(analysis,'compare_step_probes',lambda a,b:True)
    with pytest.raises(ValueError,match='bypass tuning observations'):
        analysis.compare_runs('auto','fixed',fixed_final=True)
    rows['fixed']['direct_fixed_policy']=True
    compared=analysis.compare_runs('auto','fixed',fixed_final=True)
    assert compared['first12_extra_fraction']==pytest.approx(.2)
    assert not compared['first12_budget_pass']



def test_every_step_probe_check_rejects_mismatch_missing_and_nonfinite(tmp_path):
    import json
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    initial=dict(loss=2.,probe_grad_max=.2,parameter_update_max=.01)
    result=dict(initial,steps=3,extra_step_results=[dict(index=i,**initial) for i in (2,3)])
    def write(folder,rank,data):
        (folder/f'result_rank{rank}.json').write_text(json.dumps(data))
    for folder in (a,b):
        for rank in (0,1):write(folder,rank,result)
    assert analysis.compare_step_probes(a,b)
    result['extra_step_results'][0]['loss']=3.
    write(b,1,result)
    assert not analysis.compare_step_probes(a,b)
    result['extra_step_results'][0]['loss']=float('nan');write(b,1,result)
    with pytest.raises(ValueError,match='nonfinite'):analysis.compare_step_probes(a,b)
    result['extra_step_results'][0]['loss']=2.
    result['extra_step_results'].pop();write(b,1,result)
    with pytest.raises(ValueError,match='missing'):analysis.compare_step_probes(a,b)


def test_overhead_control_requires_matching_size_calibration(monkeypatch):
    common=dict(source_sha256='same',workload={},stable={'last_step':36,'median_seconds':11.},
                ranks=[dict(policy={})],trace_capture=True,direct_fixed_policy=True,first12_wall_seconds=120.)
    def inspect(path):return dict(common,transfer_size_calibration=path=='auto')
    monkeypatch.setattr(analysis,'inspect_run',inspect)
    with pytest.raises(ValueError,match='transfer calibration differs'):
        analysis.compare_runs('auto','fixed',fixed_final=True)
