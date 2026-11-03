"""Peak-linked transfer opportunities on a captured, fixed execution graph.

Reuse the production planner's storage versions, access gaps and op/edge memory
model. A transfer is useful only across pressure in this candidate graph. The
forecast shares the runtime's raw copy stream across both directions and charges device residency from H2D
submission, not completion. It never subtracts a global DMA sum from compute.

This is a proposal estimate, not a new runtime or a proof of optimality. Queue
orders are heuristics. The fixed compute timeline does not model copy/kernel or
collective contention, codec kernels, allocator fragmentation, or a changed
recomputation graph. Changed structures require fresh evidence and full-step
validation; actual stream events remain the authority for release and reuse.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import accumulate
import math

from ..ir.transfer import transfer_duration_ms
from ..planning.peak_strategy.modeling import build_profile_and_candidates, group_accesses_by_storage
from ..planning.peak_strategy.types import PeakCandidate, SelectionKey


@dataclass(frozen=True)
class GapOpportunity:
    candidate: PeakCandidate
    ready_ms: float
    release_after_ms: float
    needed_ms: float
    first_pressure_ms: float
    last_pressure_ms: float
    pressure_slots: int
    d2h_ms: float
    h2d_ms: float

    @property
    def key(self):
        return self.candidate.selection_key


@dataclass(frozen=True)
class TransferSlot:
    key: SelectionKey
    start_ms: float
    finish_ms: float


@dataclass(frozen=True)
class QueueForecast:
    d2h: tuple[TransferSlot, ...]
    h2d: tuple[TransferSlot, ...]
    resident_bytes: tuple[int, ...]
    edge_bytes: tuple[int, ...]
    baseline_peak_bytes: int
    peak_bytes: int
    max_d2h_pressure_lateness_ms: float
    max_h2d_lateness_ms: float
    memory_fits: bool
    zero_wait_schedule_found: bool
    useful_keys: tuple[SelectionKey, ...]


def _ready_deadline_queue(jobs):
    """Non-preemptive earliest-deadline dispatch among ready jobs.

    Inputs are (ready, deadline, duration, key). A future urgent job must not
    prevent an already-ready job from using otherwise idle queue time.
    """
    pending = sorted(jobs)
    ready = []
    cursor = 0
    now = 0.0
    result = []
    while cursor < len(pending) or ready:
        if not ready:
            now = max(now, pending[cursor][0])
        while cursor < len(pending) and pending[cursor][0] <= now:
            release, deadline, duration, key = pending[cursor]
            heappush(ready, (deadline, release, key, duration))
            cursor += 1
        _, _, key, duration = heappop(ready)
        result.append(TransferSlot(key, now, now + duration))
        now += duration
    return tuple(result)


def _shared_copy_queue(opportunities, costs):
    """Ready EDF with D2H -> H2D dependencies on a single runtime stream."""
    pending, ready, outbound, inbound = [], [], [], []
    by_key = {o.key:o for o in opportunities}
    for o in opportunities:
        d2h, _ = costs.get(o.key,(o.d2h_ms,o.h2d_ms))
        heappush(pending,(o.ready_ms,o.first_pressure_ms,o.key,'d2h',d2h))
    now = 0.
    while pending or ready:
        if not ready:
            now = max(now,pending[0][0])
        while pending and pending[0][0]<=now:
            release, deadline, key, direction, duration = heappop(pending)
            heappush(ready,(deadline,release,key,direction,duration))
        _, _, key, direction, duration = heappop(ready)
        slot = TransferSlot(key,now,now+duration)
        now = slot.finish_ms
        if direction == 'd2h':
            outbound.append(slot)
            o = by_key[key]
            _, h2d = costs.get(key,(o.d2h_ms,o.h2d_ms))
            heappush(pending,(max(now,o.release_after_ms),o.needed_ms,key,'h2d',h2d))
        else:
            inbound.append(slot)
    return tuple(outbound),tuple(inbound)


class PeakWindowModel:
    """Fixed-graph managed activation pressure, separate from device high-water.

    Rebuild after changing the graph or managed-memory budget. An opportunity
    may have several gaps for the same immutable version; forecast permits at
    most one of these alternatives, matching the existing residency planner.
    """
    def __init__(self, trace):
        self.trace = trace.validate()
        if self.trace.memory_limit_bytes is None:
            raise ValueError('peak opportunities require a finite managed-memory budget')
        self.limit = self.trace.memory_limit_bytes
        self.profile = build_profile_and_candidates(self.trace, group_accesses_by_storage(self.trace))
        self.prefix = [0.0, *accumulate(float(op.duration_ms) for op in self.trace.ops)]
        self._pressure_slots = []
        for op_id, size in enumerate(self.profile.resident_bytes):
            if size > self.limit:
                self._pressure_slots.append((2 * op_id, self.prefix[op_id]))
            if op_id < len(self.profile.edge_bytes) and self.profile.edge_bytes[op_id] > self.limit:
                self._pressure_slots.append((2 * op_id + 1, self.prefix[op_id + 1]))
        self._pressure_ids = [slot for slot, _ in self._pressure_slots]
        self._pressure_ops = [slot // 2 for slot in self._pressure_ids if slot % 2 == 0]
        self._pressure_edges = [slot // 2 for slot in self._pressure_ids if slot % 2 == 1]
        self.opportunities = tuple(
            opportunity
            for candidate in self.profile.candidates + self.profile.boundary_candidates
            if (opportunity := self._opportunity(candidate)) is not None
        )
        self._known = {self._identity(o): o for o in self.opportunities}

    @staticmethod
    def _identity(opportunity):
        c = opportunity.candidate
        return c.selection_key, c.release_start.op_id, c.end.op_id

    def _opportunity(self, candidate):
        # Interior ops and the edges after release and before next use both
        # count. A lifetime spanning a peak is insufficient if its idle gap
        # does not cover that peak.
        first = bisect_left(self._pressure_ids, 2 * candidate.release_start.op_id + 1)
        last = bisect_left(self._pressure_ids, 2 * candidate.end.op_id)
        if first == last:
            return None
        times = [transfer_duration_ms(candidate.size_bytes, bandwidth,
                                     self.trace.transfer_cost_samples, direction)
                 for bandwidth, direction in ((self.trace.d2h_bandwidth_gbps, 'd2h'),
                                               (self.trace.h2d_bandwidth_gbps, 'h2d'))]
        if any(not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError('positive finite transfer durations required')
        return GapOpportunity(candidate, self.prefix[candidate.copy_start.op_id + 1],
            self.prefix[candidate.release_start.op_id + 1], self.prefix[candidate.end.op_id],
            self._pressure_slots[first][1], self._pressure_slots[last - 1][1], last - first, *times)

    def default_selection(self):
        """Reuse the production peak selector without mutating this profile."""
        from ..planning.peak_strategy.selection import select_peak_candidates, select_boundary_candidates
        from ..planning.peak_strategy.types import SelectionState, ScoreWeights
        profile = replace(self.profile, resident_bytes=list(self.profile.resident_bytes),
                          edge_bytes=list(self.profile.edge_bytes))
        state = SelectionState(set(), [])
        selected = select_peak_candidates(profile, state, self.limit, ScoreWeights(), lambda *args: None)
        selected += select_boundary_candidates(profile.boundary_candidates, state.selected_candidate_keys,
                                               profile.edge_bytes, self.limit)
        identities = {(c.selection_key, c.release_start.op_id, c.end.op_id) for c in selected}
        return tuple(o for o in self.opportunities if self._identity(o) in identities)

    def forecast(self, opportunities, *, transfer_costs=None):
        """Estimate one joint raw transfer schedule, including prefetch memory.

        First try latest-deadline backwards packing for H2D, minimizing early
        residency for that order. If releases make it infeasible, report a
        forward dispatch estimate and its lateness; never call the missed
        window an unavoidable hardware stall or silently assume it is hidden.
        """
        opportunities = tuple(opportunities)
        if len({o.key for o in opportunities}) != len(opportunities):
            raise ValueError('only one gap per storage version may be selected')
        for opportunity in opportunities:
            if self._known.get(self._identity(opportunity)) != opportunity:
                raise ValueError('opportunity does not belong to this graph and cost model')
        costs = dict(transfer_costs or {})
        if set(costs).difference(o.key for o in opportunities):
            raise ValueError('transfer cost overrides require selected storage versions')
        for pair in costs.values():
            if (not isinstance(pair, tuple) or len(pair) != 2 or any(isinstance(x, bool)
                    or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in pair)):
                raise ValueError('two positive finite directional transfer costs required')
        outbound = _ready_deadline_queue([
            (o.ready_ms, o.first_pressure_ms, costs.get(o.key, (o.d2h_ms, o.h2d_ms))[0], o.key)
            for o in opportunities
        ])
        out_by_key = {slot.key: slot for slot in outbound}
        jobs = [(max(out_by_key[o.key].finish_ms, o.release_after_ms),
                 o.needed_ms, costs.get(o.key, (o.d2h_ms, o.h2d_ms))[1], o.key) for o in opportunities]
        next_start = math.inf
        inbound = []
        latest_feasible = True
        outbound_starts = [slot.start_ms for slot in outbound]
        for release, deadline, duration, key in sorted(jobs, key=lambda row: (row[1], row[3]), reverse=True):
            finish = min(deadline, next_start)
            # Raw D2H and H2D use the SAME ResidencyManager copy stream. Pack
            # backwards around already-placed outbound work, rather than
            # claiming a second independent link window at the same time.
            cursor = bisect_left(outbound_starts, finish)-1
            while cursor >= 0:
                occupied = outbound[cursor]
                if occupied.finish_ms <= finish-duration:
                    break
                finish = occupied.start_ms
                cursor -= 1
            start = finish - duration
            latest_feasible &= start >= release
            inbound.append(TransferSlot(key, start, finish))
            next_start = start
        if latest_feasible:
            inbound = tuple(reversed(inbound))
        else:
            outbound,inbound = _shared_copy_queue(opportunities,costs)
            out_by_key = {slot.key:slot for slot in outbound}
        in_by_key = {slot.key: slot for slot in inbound}
        op_delta = [0] * (len(self.profile.resident_bytes) + 1)
        edge_delta = [0] * (len(self.profile.edge_bytes) + 1)
        useful = []
        for o in opportunities:
            c = o.candidate
            released = max(o.release_after_ms, out_by_key[o.key].finish_ms)
            allocated = in_by_key[o.key].start_ms
            if allocated <= released:
                continue
            # Conservative op peak: credit only if absent for the entire op.
            # Prefetch allocates the complete raw buffer at START, not finish.
            lo = max(c.start_interior, bisect_left(self.prefix, released, hi=len(self.trace.ops)))
            hi = min(c.end.op_id, bisect_right(self.prefix, allocated, lo=1) - 1,
                     bisect_left(self.prefix, allocated))
            op_relief = lo < hi
            if op_relief:
                op_delta[lo] -= c.size_bytes
                op_delta[hi] += c.size_bytes
            edge_lo = max(c.release_start.op_id, bisect_left(self.prefix, released, lo=1) - 1)
            edge_hi = min(c.end.op_id, bisect_left(self.prefix, allocated, lo=1) - 1,
                          len(self.profile.edge_bytes))
            edge_relief = edge_lo < edge_hi
            if edge_relief:
                edge_delta[edge_lo] -= c.size_bytes
                edge_delta[edge_hi] += c.size_bytes
            lowers_pressure = (
                op_relief and bisect_left(self._pressure_ops, hi) > bisect_left(self._pressure_ops, lo)
                or edge_relief and bisect_left(self._pressure_edges, edge_hi) > bisect_left(self._pressure_edges, edge_lo)
            )
            if lowers_pressure:
                useful.append(o.key)
        resident = tuple(value + delta for value, delta in zip(self.profile.resident_bytes, accumulate(op_delta)))
        edges = tuple(value + delta for value, delta in zip(self.profile.edge_bytes, accumulate(edge_delta)))
        if min((*resident, *edges), default=0) < 0:
            raise ValueError('overlapping storage residency credits')
        peak = max((*resident, *edges), default=0)
        late_out = max((out_by_key[o.key].finish_ms - o.first_pressure_ms for o in opportunities), default=0.0)
        late_in = max((in_by_key[o.key].finish_ms - o.needed_ms for o in opportunities), default=0.0)
        return QueueForecast(outbound, inbound, resident, edges,
            max((*self.profile.resident_bytes, *self.profile.edge_bytes), default=0), peak,
            max(0.0, late_out), max(0.0, late_in), peak <= self.limit,
            peak <= self.limit and late_in <= 0, tuple(useful))
