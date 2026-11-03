from dataclasses import replace

import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, AccessKind as K, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.tuning.peak_windows import PeakWindowModel, _ready_deadline_queue


def trace(*, packets=1, milliseconds=1.0, latency=1.0, pressure_at=3,
          consumer=8, limit=None):
    """Long-lived saved tensors and an ephemeral allocation at one peak."""
    ops = [TraceOp('arbitrary', milliseconds) for _ in range(consumer + 1)]
    sizes = {sid: 100 for sid in range(packets)}
    for sid in range(packets):
        ops[0].accesses.append(StorageAccess(0, sid, K.WRITE))
        ops[consumer].accesses.append(StorageAccess(consumer, sid, K.READ))
    sizes[99] = 100
    ops[pressure_at].accesses.append(StorageAccess(pressure_at, 99, K.WRITE))
    return ActivationTrace(ops, sizes, memory_limit_bytes=100*packets if limit is None else limit,
                           transfer_cost_samples=((100, latency, latency),))


def test_reuses_storage_peak_model_and_ignores_non_crossing_temporaries():
    model = PeakWindowModel(trace())
    assert model.profile.resident_bytes[3] == 200
    assert {o.candidate.storage_id for o in model.opportunities} == {0}
    opportunity, = model.opportunities
    result = model.forecast((opportunity,))
    assert result.baseline_peak_bytes == 200 and result.peak_bytes == 100
    assert result.zero_wait_schedule_found and result.memory_fits
    assert result.useful_keys == (opportunity.key,)
    assert result.d2h[0].start_ms == 1 and result.d2h[0].finish_ms == 2
    assert result.h2d[0].start_ms == 7 and result.h2d[0].finish_ms == 8


def test_no_pressure_means_no_pointless_offload_and_names_are_irrelevant():
    graph = trace(limit=200)
    assert PeakWindowModel(graph).opportunities == ()
    original = PeakWindowModel(trace())
    renamed = trace()
    for op in renamed.ops:
        op.name = 'entirely_different_model'
    other = PeakWindowModel(renamed)
    assert other.opportunities == original.opportunities
    assert other.forecast(other.opportunities) == original.forecast(original.opportunities)


def test_late_d2h_cannot_claim_it_removed_the_peak():
    model = PeakWindowModel(trace(pressure_at=2, latency=2))
    result = model.forecast(model.opportunities)
    assert result.max_d2h_pressure_lateness_ms == 1
    assert result.peak_bytes == 200 and not result.memory_fits
    assert not result.zero_wait_schedule_found and not result.useful_keys


def test_two_packets_share_outbound_window_instead_of_counting_slack_twice():
    graph = trace(packets=2, pressure_at=3, latency=2, limit=200)
    graph.storage_sizes[99] = 200
    model = PeakWindowModel(graph)
    assert len(model.opportunities) == 2
    result = model.forecast(model.opportunities)
    assert [(s.start_ms, s.finish_ms) for s in result.d2h] == [(1, 3), (3, 5)]
    assert result.max_d2h_pressure_lateness_ms == 2
    assert result.resident_bytes[3] == 300
    assert not result.memory_fits
    assert len(result.useful_keys) < len(model.opportunities)


def test_prefetch_allocates_at_start_and_can_reintroduce_peak():
    model = PeakWindowModel(trace(pressure_at=6, consumer=8, latency=3))
    result = model.forecast(model.opportunities)
    assert result.h2d[0].start_ms == 5 and result.h2d[0].finish_ms == 8
    assert result.max_h2d_lateness_ms == 0
    assert result.resident_bytes[6] == 200  # The buffer is back before its read.
    assert not result.memory_fits and not result.useful_keys


def test_long_lifetime_without_a_gap_across_pressure_is_not_an_opportunity():
    graph = trace()
    graph.ops[3].accesses.append(StorageAccess(3, 0, K.READ))
    model = PeakWindowModel(graph)
    assert model.opportunities == ()


def test_h2d_release_constraint_and_same_direction_serialization():
    model = PeakWindowModel(trace(packets=2, latency=3, pressure_at=3, consumer=7, limit=100))
    result = model.forecast(model.opportunities)
    assert result.max_h2d_lateness_ms > 0
    outbound = {slot.key: slot for slot in result.d2h}
    for slot in result.h2d:
        assert slot.start_ms >= outbound[slot.key].finish_ms
    for left, right in zip(result.h2d, result.h2d[1:]):
        assert left.finish_ms <= right.start_ms
    combined=sorted((*result.d2h,*result.h2d),key=lambda slot:slot.start_ms)
    assert all(left.finish_ms<=right.start_ms for left,right in zip(combined,combined[1:]))
    assert not result.zero_wait_schedule_found


def test_opposite_directions_cannot_reuse_the_same_raw_copy_stream_window():
    model=PeakWindowModel(trace(packets=2,latency=2.,pressure_at=3,consumer=7,limit=100))
    forecast=model.forecast(model.opportunities)
    # Four2ms operations cannot fit into the6ms interval[1,7]. Independent
    # directional queues would incorrectly claim a zero-late pipeline.
    assert forecast.max_h2d_lateness_ms>=2.
    combined=sorted((*forecast.d2h,*forecast.h2d),key=lambda slot:slot.start_ms)
    assert all(a.finish_ms<=b.start_ms for a,b in zip(combined,combined[1:]))


def test_ready_queue_does_not_idle_for_an_unreleased_urgent_packet():
    slots = _ready_deadline_queue([(0, 20, 2, (0,)), (1, 3, 1, (1,))])
    assert [(slot.key, slot.start_ms, slot.finish_ms) for slot in slots] == [((0,), 0, 2), ((1,), 2, 3)]


def test_same_version_alternatives_cannot_spend_memory_credit_twice():
    model = PeakWindowModel(trace())
    opportunity, = model.opportunities
    with pytest.raises(ValueError, match='one gap per storage version'):
        model.forecast((opportunity, opportunity))
    with pytest.raises(ValueError, match='does not belong'):
        model.forecast((replace(opportunity, d2h_ms=0),))


def test_versions_and_aliases_come_from_existing_candidate_generation():
    graph = trace()
    graph.ops[2].accesses.extend([StorageAccess(2, 0, K.READ), StorageAccess(2, 0, K.WRITE)])
    graph.ops[2].name = 'view'
    model = PeakWindowModel(graph)
    opportunity, = model.opportunities
    # Current planner conservatively starts the snapshot after the alias write.
    assert opportunity.candidate.copy_start.op_id == 2
    graph.retained_sids.add(0)
    assert PeakWindowModel(graph).opportunities == ()


def test_new_candidate_graph_requires_rebuilding_the_peak_model():
    graph = trace(limit=200)
    assert PeakWindowModel(graph).opportunities == ()
    # Removing a recomputation may materialize an additional large saved value.
    graph.storage_sizes[4] = 150
    graph.ops[1].accesses.append(StorageAccess(1, 4, K.WRITE))
    graph.ops[8].accesses.append(StorageAccess(8, 4, K.READ))
    updated = PeakWindowModel(graph)
    assert updated.opportunities
    assert {o.candidate.storage_id for o in updated.opportunities} == {0, 4}


def test_empty_trace_and_missing_budget_are_explicit():
    model = PeakWindowModel(ActivationTrace(memory_limit_bytes=0))
    result = model.forecast(())
    assert result.peak_bytes == 0 and result.memory_fits
    with pytest.raises(ValueError, match='finite managed-memory budget'):
        PeakWindowModel(ActivationTrace())
