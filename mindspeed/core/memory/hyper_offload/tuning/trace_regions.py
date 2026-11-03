"""Read-only region costs from one complete warmup trace.

A boundary tensor is not necessarily an autograd save. Only a version produced
inside a visible region and read in backward contributes internal saved bytes.
Opaque calls, external/retained storage and ambiguous mutation are excluded.
The result is proposal evidence; a new execution structure still needs a trace
and full-step memory/performance admission.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math

from ..ir.trace import AccessKind


@dataclass(frozen=True)
class SavedVersion:
    storage_id: int
    produced_at: int
    first_backward_read: int
    nbytes: int
    last_backward_read: int | None = None


@dataclass(frozen=True)
class RegionCost:
    name: str
    structure: tuple
    input_shapes: tuple
    first_op: int
    last_op: int
    compute_ms: float
    saved_versions: tuple[SavedVersion, ...]
    input_boundary_bytes: int
    output_boundary_bytes: int
    created_bytes: int
    forward_save_window_ms: float = 0.0
    backward_save_window_ms: float = 0.0
    working_set_bytes: int = 0

    @property
    def saved_bytes(self):
        return sum(v.nbytes for v in self.saved_versions)


def region_costs(trace, report, *, max_regions=2048, max_ops=100000):
    """O(ops + versions log versions + observed nesting depth * versions).

    Operator event durations, not inclusive module intervals, are summed.
    Unknown/incomplete/opaque scopes have no RegionCost entry, never zero cost.
    Non-alias writes to existing storage are conservatively not attributed to a
    checkpoint region because a changed checkpoint may alter those lifetimes.
    """
    if report.get('truncated'):
        return ()
    rows = report['regions']
    if len(rows) > max_regions or len(trace.ops) > max_ops:
        raise ValueError('region evidence exceeds bounded analysis capacity')
    end_forward = report.get('forward_end_op')
    if not isinstance(end_forward, int) or not 0 <= end_forward < len(trace.ops):
        return ()
    # Dense executor IDs are validated during the single access traversal.
    aliases = frozenset(('view', '_unsafe_view', 'reshape', 'detach', 'transpose', 't',
                         'slice', 'select', 'as_strided', 'squeeze', 'unsqueeze', 'expand',
                         'alias', 'view_as_real', 'view_as_complex'))
    name_flags = {}
    for name in dict.fromkeys(op.name for op in trace.ops):
        lower = name.lower()
        name_flags[name] = (name.endswith(('_fwd', '_bwd')) or any(
            key in lower for key in ('all_gather', 'reduce_scatter', 'all_reduce', 'broadcast')),
            name.split('.')[0] in aliases)
    read, write = AccessKind.READ, AccessKind.WRITE
    prefix = [0.0]
    unsafe_prefix = [0]
    compute_prefix = [0.0]
    current, versions, ambiguous = {}, [], set()
    working_sets = []
    storage_sizes,retained = trace.storage_sizes,trace.retained_sids
    for index, op in enumerate(trace.ops):
        duration = float(op.duration_ms)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid measured operator duration')
        prefix.append(prefix[-1] + duration)
        # An opaque inner scope hides its own recomputation and storage costs.
        # Collective time must not be counted as removable model recomputation.
        unsafe, alias = name_flags[op.name]
        unsafe_prefix.append(unsafe_prefix[-1] + int(unsafe))
        compute_prefix.append(compute_prefix[-1] + (0.0 if unsafe else duration))
        grouped = {}
        for access in op.accesses:
            if access.op_id != index:
                raise ValueError('region observations require matching dense trace positions')
            if access.kind is read:
                mask = 1
            elif access.kind is write:
                mask = 2
            else:
                raise ValueError('unknown storage access kind')
            sid = access.storage_id
            grouped[sid] = grouped.get(sid, 0) | mask
        working_set = 0
        for sid, kinds in grouped.items():
            if sid in retained:
                continue
            size = storage_sizes[sid]
            working_set += size
            if kinds & 2:
                if kinds & 1:
                    if not (alias and len(grouped) == 1):
                        ambiguous.add(sid)
                else:
                    current[sid] = [sid, index, None, size, None]
                    versions.append(current[sid])
            if kinds & 1 and index > end_forward and sid in current:
                version = current[sid]
                if version[2] is None:
                    version[2] = index
                version[4] = index
        working_sets.append(working_set)
    # Iterative range maxima keep nested/large scopes from rescanning ops.
    count = len(working_sets)
    maximum_tree = [0] * count + working_sets
    for index in range(count-1, 0, -1):
        maximum_tree[index] = max(maximum_tree[2*index], maximum_tree[2*index+1])
    def maximum_working_set(first, last):
        first, last, value = first+count, last+count+1, 0
        while first < last:
            if first & 1:
                value = max(value, maximum_tree[first]); first += 1
            if last & 1:
                last -= 1; value = max(value, maximum_tree[last])
            first //= 2; last //= 2
        return value
    # Versions are appended in operator order; filtering preserves that order.
    all_versions = versions
    all_positions = [v[1] for v in all_versions]
    saved = [SavedVersion(*v) for v in versions
             if v[0] not in ambiguous and v[2] is not None and v[1] <= end_forward and v[3] > 0]
    positions = [v.produced_at for v in saved]
    result = []
    for row in rows:
        first, last = row['first_op'], row['last_op']
        if (not row['complete'] or row['opaque'] or not 0 <= first <= last <= end_forward
                or unsafe_prefix[last+1] != unsafe_prefix[first]):
            continue
        boundary = set(row.get('input_storage_ids', ())) | set(row.get('output_storage_ids', ()))
        internal = tuple(v for v in saved[bisect_left(positions, first):bisect_right(positions, last)]
                         if v.storage_id not in boundary)
        created = {v[0]: v[3] for v in all_versions[bisect_left(all_positions, first):bisect_right(all_positions, last)] if v[0] not in boundary}
        # These are disjoint region-local compute budgets for ordering
        # experiments, not a claim that every transfer meets its deadline.
        # A complete newly planned step remains the admission authority.
        forward_window = (compute_prefix[last+1] - compute_prefix[min(v.produced_at for v in internal)+1]) if internal else 0.0
        backward_window = (compute_prefix[max(v.last_backward_read for v in internal)]
                           - compute_prefix[min(v.first_backward_read for v in internal)]) if internal else 0.0
        working_set = maximum_working_set(first, last)
        if internal:
            working_set = max(working_set, maximum_working_set(min(v.first_backward_read for v in internal),
                                                              max(v.last_backward_read for v in internal)))
        result.append(RegionCost(row['name'], tuple(row['structure']), tuple(row['input_shapes']), first, last,
                                 prefix[last+1]-prefix[first], internal,
                                 row['input_boundary_bytes'], row['output_boundary_bytes'], sum(created.values()),
                                 forward_window, backward_window, working_set))
    return tuple(result)
