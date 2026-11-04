"""Place both transfer directions in one captured graph's device timeline."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ...ir.transfer import transfer_duration_ms


@dataclass(frozen=True)
class _Job:
    key: tuple
    direction: str
    earliest: int
    deadline: int
    duration: float


class _FreeTime:
    """Remaining compute intervals in milliseconds on one shared copy queue."""

    def __init__(self, end):
        self.spans = [(0.0, end)]

    def take(self, start, finish):
        remaining = []
        for left, right in self.spans:
            if finish <= left or start >= right:
                remaining.append((left, right))
                continue
            if left < start:
                remaining.append((left, start))
            if finish < right:
                remaining.append((finish, right))
        self.spans = remaining

    def place(self, job, prefix):
        # Delay until required to preserve earlier compute for urgent jobs.
        # Snap to a real submit boundary before spending any interval.
        index = job.earliest
        for left, right in reversed(self.spans):
            latest = min(right, prefix[job.deadline]) - job.duration
            bound = bisect_right(prefix, latest, job.earliest, job.deadline + 1) - 1
            if bound >= job.earliest and prefix[bound] >= left:
                index = bound
                break
        # An oversubscribed job issues at its first legal boundary. Charge the
        # occupied part of the timeline too. Runtime events cover any spill.
        self.take(prefix[index], prefix[index] + job.duration)
        return index


def place_transfers(context, last_release, first_prefetch):
    """Return concrete D2H, release and H2D positions within a safe envelope.

    A boundary b is after op b-1 and before op b. D2H cannot start until
    production. H2D cannot start before device release and must issue by use.
    Durations are directional packet calibration, not a guarantee of overlap.
    """
    prefix = context.time_prefix_ms
    if not prefix or len(prefix) != len(context.resident_bytes) + 1:
        raise ValueError("shared transfers require a complete device timeline")
    jobs = []
    minimum_release = {}
    for c in context.selected_candidates:
        key = c.selection_key
        minimum_release[key] = c.release_start.op_id
        for direction, begin, end, rate in (
            ("d2h", c.copy_start.op_id + 1, last_release[key] + 1, context.d2h_bandwidth_gbps),
            ("h2d", first_prefetch[key], c.end.op_id, context.h2d_bandwidth_gbps),
        ):
            duration = transfer_duration_ms(c.size_bytes, rate, context.transfer_cost_samples, direction)
            jobs.append(_Job(key, direction, begin, end, duration))
    # Place the least flexible jobs first. Both directions consume the same
    # intervals, including when different versions' forward/backward overlap.
    jobs.sort(
        key=lambda j: (prefix[j.deadline] - prefix[j.earliest] - j.duration, j.deadline, j.key, j.direction)
    )
    free = _FreeTime(prefix[-1])
    copies, releases, prefetches = {}, {}, {}
    for job in jobs:
        boundary = free.place(job, prefix)
        if job.direction == "h2d":
            prefetches[job.key] = boundary
        else:
            copies[job.key] = boundary - 1
            finish = bisect_left(prefix, prefix[boundary] + job.duration, lo=boundary)
            releases[job.key] = min(last_release[job.key], max(minimum_release[job.key], finish - 1))
    return copies, releases, prefetches
