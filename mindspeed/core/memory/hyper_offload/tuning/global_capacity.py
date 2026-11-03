"""Whole-step copy capacity before choosing any activation nodes.

The estimate is a fractional scheduling relaxation on one shared D2H/H2D
server. It bounds a round trip's original bytes, not the sum of both DMA
directions. It contains no storage identities or per-node slack sums. Node
availability, deadlines, pressure relief, packet latency and device capacity
must subsequently make this global budget executable on a concrete graph.
"""
from dataclasses import dataclass
import math

from ..ir.transfer import transfer_duration_ms


@dataclass(frozen=True)
class CapacitySegment:
    start_ms: float
    finish_ms: float
    d2h_available: bool
    h2d_available: bool


@dataclass(frozen=True)
class GlobalCopyCapacity:
    segments: tuple[CapacitySegment, ...]
    d2h_available_ms: float
    h2d_available_ms: float
    shared_server_ms: float
    d2h_gibps: float
    h2d_gibps: float
    roundtrip_original_bytes: int
    limiting_constraints: tuple[str, ...]
    scope: str = 'global fractional window capacity; node placement and full-step validation still required'


def _union(windows):
    ordered = []
    for window in windows:
        if len(window) != 2:
            raise ValueError('copy windows must contain start and finish')
        start, finish = window
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x < 0 for x in window) or finish < start:
            raise ValueError('copy windows must be finite ordered nonnegative intervals')
        if start < finish:
            ordered.append((float(start), float(finish)))
    merged = []
    for start, finish in sorted(ordered):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(finish, merged[-1][1]))
        else:
            merged.append((start, finish))
    return tuple(merged)


def estimate_global_copy_capacity(d2h_windows, h2d_windows, *, d2h_gibps, h2d_gibps,
                                  host_limit_bytes):
    """Return a single global byte budget and its nonduplicated time segments.

    For a raw round trip of S bytes, D2H requires S/Bd and H2D S/Bh.
    Each direction must fit its allowed windows, and their SUM must fit the
    union because the server is shared. These are global constraints, before
    any storage is selected. Intersecting windows never create extra time.
    """
    for rate in (d2h_gibps, h2d_gibps):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('global capacity requires positive finite measured rates')
    if isinstance(host_limit_bytes, bool) or not isinstance(host_limit_bytes, int) or host_limit_bytes < 0:
        raise ValueError('host capacity must be a nonnegative integer')
    directions = (_union(d2h_windows), _union(h2d_windows))
    boundaries = sorted({point for windows in directions for window in windows for point in window})
    cursors = [0, 0]
    segments = []
    for start, finish in zip(boundaries, boundaries[1:]):
        active = []
        for index, windows in enumerate(directions):
            while cursors[index] < len(windows) and windows[cursors[index]][1] <= start:
                cursors[index] += 1
            active.append(cursors[index] < len(windows) and windows[cursors[index]][0] <= start)
        if any(active):
            segments.append(CapacitySegment(start, finish, *active))
    dtime = sum(s.finish_ms-s.start_ms for s in segments if s.d2h_available)
    htime = sum(s.finish_ms-s.start_ms for s in segments if s.h2d_available)
    total = sum(s.finish_ms-s.start_ms for s in segments)
    drate, hrate = d2h_gibps * 2**30 / 1000., h2d_gibps * 2**30 / 1000.
    bounds = dict(d2h_windows=dtime*drate, h2d_windows=htime*hrate,
                  shared_copy_server=total/(1./drate+1./hrate), host_capacity=host_limit_bytes)
    smallest = min(bounds.values())
    limiting = tuple(name for name, value in bounds.items() if math.isclose(value, smallest, rel_tol=1e-12, abs_tol=1e-6))
    return GlobalCopyCapacity(tuple(segments), dtime, htime, total, d2h_gibps, h2d_gibps,
                              math.floor(smallest), limiting)


def estimate_forward_backward_capacity(trace, forward_end_op, *, host_limit_bytes,
                                       removed_backward_compute_ms=0., size_histogram=None):
    """A whole-step starting envelope for forward-produced training saves.

    Uses forward egress and backward ingress as an initial planning envelope;
    it does not claim that copies cannot cross the forward/backward boundary.
    Later node placement can account for such crossings and device headroom.
    Existing recomputation that a raw seed would remove is subtracted BEFORE
    computing capacity. Captured operator durations and isolated calibration
    remain estimates, not measured available runtime slack.
    """
    trace.validate()
    if isinstance(forward_end_op, bool) or not isinstance(forward_end_op, int) or not 0 <= forward_end_op < len(trace.ops):
        raise ValueError('a complete forward boundary is required')
    forward = sum(op.duration_ms for op in trace.ops[:forward_end_op+1])
    backward = sum(op.duration_ms for op in trace.ops[forward_end_op+1:])
    removed = removed_backward_compute_ms
    if (isinstance(removed, bool) or not isinstance(removed, (int, float)) or not math.isfinite(removed)
            or not 0 <= removed <= backward):
        raise ValueError('removed recomputation must fit the observed backward duration')
    backward -= removed
    # A global size histogram may refine the measured link rates before node
    # selection. Do not apply the slowest small-packet rate to every byte of
    # the workload. Exact packet latency still matters during node placement.
    d2h, h2d = trace.d2h_bandwidth_gbps, trace.h2d_bandwidth_gbps
    if size_histogram is not None:
        byte_count, dms, hms = 0, 0., 0.
        for size, count in size_histogram:
            if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (size, count)):
                raise ValueError('global packet histogram requires positive integer sizes and counts')
            byte_count += size * count
            dms += count * transfer_duration_ms(size, d2h, trace.transfer_cost_samples, 'd2h')
            hms += count * transfer_duration_ms(size, h2d, trace.transfer_cost_samples, 'h2d')
        if byte_count:
            d2h, h2d = byte_count/2**30/(dms/1000.), byte_count/2**30/(hms/1000.)
    return estimate_global_copy_capacity(((0., forward),), ((forward, forward+backward),),
        d2h_gibps=d2h, h2d_gibps=h2d, host_limit_bytes=host_limit_bytes)
