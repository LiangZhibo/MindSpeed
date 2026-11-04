from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind, StorageAccess
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.masking import apply_communication_masking
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.transfer_calendar import place_transfers
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import MaskingContext, PeakCandidate


def candidate(sid, produced, released, consumer):
    def access(op, kind):
        return StorageAccess(op, sid, kind)

    return PeakCandidate(
        sid,
        1048576,
        access(produced, AccessKind.WRITE),
        access(released, AccessKind.READ),
        access(consumer, AccessKind.READ),
        produced,
        consumer,
    )


def context(candidates):
    return MaskingContext(
        candidates,
        [0] * 13,
        4 * 1048576,
        {},
        set(),
        time_prefix_ms=list(range(14)),
        transfer_cost_samples=((1048576, 2.0, 2.0),),
        shared_transfer_calendar=True,
    )


def test_opposite_directions_share_one_timeline():
    first = candidate(1, 0, 1, 8)
    second = candidate(2, 3, 5, 12)
    rows = [first, second]
    copy, release, read = place_transfers(
        context(rows),
        {first.selection_key: 3, second.selection_key: 7},
        {first.selection_key: 4, second.selection_key: 8},
    )
    intervals = []
    for c in rows:
        key = c.selection_key
        assert c.copy_start.op_id <= copy[key] <= release[key] < read[key] <= c.end.op_id
        intervals += [(copy[key] + 1, copy[key] + 3), (read[key], read[key] + 2)]
    intervals.sort()
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert any(copy[c.selection_key] != c.copy_start.op_id for c in rows)


def test_infeasible_deadline_keeps_legal_actions_and_original_codec_node():
    small = candidate(8, 2, 2, 3)
    wide = candidate(9, 0, 2, 11)
    schedule, profile = apply_communication_masking(context([small, wide]))

    def actions(kind):
        return {
            a.storage_id: a.op_id
            for table in (schedule.pre, schedule.post)
            for row in table.values()
            for a in row
            if a.kind.name == kind
        }

    outbound, inbound, release = [actions(k) for k in ("COPY_D2H", "COPY_H2D", "RELEASE_DEVICE")]
    for c in (small, wide):
        sid = c.storage_id
        assert c.copy_start.op_id <= outbound[sid] <= release[sid] < inbound[sid] <= c.end.op_id
        assert schedule.copy_origins.get((outbound[sid], sid), outbound[sid]) == c.copy_start.op_id
    assert max(profile) <= 4 * 1048576
