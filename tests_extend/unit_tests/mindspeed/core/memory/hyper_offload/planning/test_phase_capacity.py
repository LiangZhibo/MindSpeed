import pytest

from mindspeed.core.memory.hyper_offload.tuning.phase_capacity import PhaseCapacity


def report(peaks):
    return dict(forward_end_op=1,truncated=False,samples=[(i,p,p,p,p) for i,p in enumerate(peaks)])


def test_early_activation_does_not_add_to_later_optimizer_peak():
    capacity=PhaseCapacity(report([20,30,80,90]),op_count=4,limit_bytes=100)
    assert capacity.try_reserve([(0,1,50,None)])
    assert capacity.estimated_peak_bytes==90
    assert not capacity.try_reserve([(1,3,20,None)])
    assert capacity.estimated_peak_bytes==90  # Rejection is atomic.


def test_shared_planner_pool_is_an_envelope_native_allocations_add():
    capacity=PhaseCapacity(report([20]*5),op_count=5,limit_bytes=100)
    assert capacity.try_reserve([(0,3,40,'raw_pool'),(1,4,50,'raw_pool')])
    assert capacity.estimated_peak_bytes==70
    assert capacity.try_reserve([(0,4,20,None)])
    assert not capacity.try_reserve([(0,4,20,None)])
    assert capacity.estimated_peak_bytes==90


def test_coarsening_rounds_lifetimes_outward_and_uses_bucket_peak():
    capacity=PhaseCapacity(report([10,90,20,90]),op_count=4,limit_bytes=100,max_buckets=2)
    assert not capacity.try_reserve([(0,0,20,None)])
    assert capacity.estimated_peak_bytes==90


def test_missing_positions_and_truncated_evidence_cannot_admit_policy():
    r=report([10,20]);r['samples'][1]=(9,20,20,20,20)
    with pytest.raises(ValueError,match='cover every'):
        PhaseCapacity(r,op_count=2,limit_bytes=100)
    r=report([10,20]);r['truncated']=True
    with pytest.raises(ValueError,match='incomplete'):
        PhaseCapacity(r,op_count=2,limit_bytes=100)


def test_rank_merge_requires_trace_identity_and_takes_phase_maximum():
    from mindspeed.core.memory.hyper_offload.tuning.phase_capacity import merge_phase_reports
    a=report([10,80]);a.update(trace_signature='one',columns=['op','allocated','peak','reserved','peak_reserved'],regions={})
    b=report([30,70]);b.update(trace_signature='one',columns=a['columns'],regions={})
    merged=merge_phase_reports([a,b])
    assert [row[2] for row in merged['samples']]==[30,80]
    b['trace_signature']='another'
    assert merge_phase_reports([a,b]) is None
    b['trace_signature']='one';b['regions']={'scope':[1]}
    assert merge_phase_reports([a,b]) is None


def test_baseline_peak_over_limit_cannot_be_hidden_by_empty_bounds():
    with pytest.raises(ValueError,match='already exceeds'):
        PhaseCapacity(report([101]),op_count=1,limit_bytes=100)


def test_compact_capacity_matches_raw_and_merges_conservatively():
    from mindspeed.core.memory.hyper_offload.tuning.phase_capacity import summarize_phase_capacity, compact_phase_report, merge_phase_reports
    reports=[]
    for peaks in ([10,20,70,80],[20,40,60,90]):
        r=report(peaks);r.update(trace_signature='same',columns=['op','allocated','peak','reserved','peak_reserved'],regions={})
        r['capacity_summary']=summarize_phase_capacity(r,op_count=4,max_buckets=2)
        reports.append(r)
    merged=merge_phase_reports([compact_phase_report(r) for r in reports])
    assert 'samples' not in merged and merged['capacity_summary']['peak_bytes']==[40,90]
    compact=PhaseCapacity(merged,op_count=4,limit_bytes=100)
    assert compact.try_reserve([(0,1,60,None)])
    assert not compact.try_reserve([(2,3,11,None)])
    raw=PhaseCapacity(reports[0],op_count=4,limit_bytes=100)
    assert raw.width==2 and raw.peak==[20,80]
