"""Whole-configuration proposals from profiled device compute / DMA intervals.

The input is a compact, graph-bound adapter contract. Times are seconds on
each sample's own device timeline. Reference packet keys are for accounting
only. No storage identifier from this profile is installed on a new graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from bisect import bisect_left
from statistics import median

from .phase_capacity import PhaseCapacity


def _number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("expected a finite nonnegative duration or timestamp")
    return float(value)


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative integer")
    return value


def _interval(row):
    if len(row) != 2:
        raise ValueError("interval must have two boundaries")
    lo, hi = map(_number, row)
    if hi < lo:
        raise ValueError("inverted interval")
    return lo, hi


def _names(names):
    if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("region names must be nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("duplicate region names")
    return tuple(sorted(names))


@dataclass(frozen=True)
class Packet:
    key: str
    size: int
    ready: float
    first_pressure: float
    peak_pressure: float
    prefetch: float
    use: float
    d2h: float
    h2d: float

    @classmethod
    def parse(cls, row):
        key = row["key"]
        if not isinstance(key, str) or not key:
            raise ValueError("missing version key")
        p = cls(
            key,
            _integer(row["size"]),
            *(
                _number(row[k])
                for k in ("ready", "first_pressure", "peak_pressure", "prefetch", "use", "d2h", "h2d")
            ),
        )
        if (
            p.size == 0
            or min(p.d2h, p.h2d) <= 0
            or p.prefetch > p.use
            or p.first_pressure > p.peak_pressure
        ):
            raise ValueError("invalid packet profile")
        return p


@dataclass(frozen=True)
class RegionSample:
    removed_compute: tuple[float, float]
    packets: tuple[Packet, ...]


@dataclass(frozen=True)
class Region:
    name: str
    saved: float
    workspace: int
    native: int
    first_op: int
    last_op: int
    native_until: int
    samples: tuple[RegionSample, ...]
    replaces: tuple[str, ...] = ()


@dataclass(frozen=True)
class Structure:
    offload: tuple[str, ...]
    resident: tuple[str, ...]
    saved_seconds: float
    gross_bytes: int
    mode: str
    replaced_checkpoints: tuple[str, ...] = ()


class _Calendar:
    """Shared, nonpreemptive D2H/H2D slots. Never spend an interval twice."""

    def __init__(self, free, removed):
        merged = []
        for lo, hi in sorted(removed):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        self.slots = []
        i = 0
        for lo, hi in free:
            while i < len(merged) and merged[i][1] <= lo:
                i += 1
            j = i
            while j < len(merged) and merged[j][0] < hi:
                start, stop = merged[j]
                if lo < start:
                    self.slots.append((lo, start))
                lo = max(lo, stop)
                if lo >= hi:
                    break
                j += 1
            if lo < hi:
                self.slots.append((lo, hi))
        self.ends = [hi for _, hi in self.slots]

    def reserve(self, ready, deadline, duration):
        i = bisect_left(self.ends, ready)
        while i < len(self.slots):
            lo, hi = self.slots[i]
            start = max(ready, lo)
            if start + duration > deadline:
                return None
            if start + duration <= hi:
                finish = start + duration
                fragments = [(a, b) for a, b in ((lo, start), (finish, hi)) if a < b]
                self.slots[i : i + 1] = fragments
                self.ends[i : i + 1] = [b for _, b in fragments]
                return finish
            i += 1
        return None


class DeviceWindowProfile:
    """A reusable profile of one workload and one qualified anchor graph."""

    def __init__(self, data):
        if data.get("schema") != 1:
            raise ValueError("unsupported device-window profile schema")
        self.signature = data["anchor_signature"]
        if (
            not isinstance(self.signature, str)
            or len(self.signature) != 64
            or any(c not in "0123456789abcdef" for c in self.signature)
        ):
            raise ValueError("invalid anchor graph signature")
        self.recompute = _names(data["anchor_recompute"])
        self.resident = _names(data["anchor_resident"])
        if set(self.recompute) & set(self.resident):
            raise ValueError("inconsistent anchor")
        scale = data.get("anchor_window_scale")
        self.window_scale = None if scale is None else _number(scale)
        self.op_count = _integer(data["anchor_op_count"])
        self.device_mb = _integer(data["resident_mb"])
        self.codec_origins = tuple(
            (_integer(op), _integer(size)) for op, size in data.get("codec_origins", ())
        )
        if len(set(self.codec_origins)) != len(self.codec_origins) or any(
            op >= self.op_count or size == 0 for op, size in self.codec_origins
        ):
            raise ValueError("invalid profiled codec origins")
        self.prefer_shared_calendar = False
        pair = data.get("calendar_comparison")
        if pair is not None:
            if pair["trace_signature"] != self.signature:
                raise ValueError("schedule observations belong to a different anchor")
            old, new = (tuple(_number(v) for v in pair[name]) for name in ("original", "shared"))
            if not old or len(old) != len(new) or min(*old, *new) <= 0:
                raise ValueError("schedule comparison requires equal nonempty positive samples")
            self.prefer_shared_calendar = median(new) < median(old)
        self.windows = []
        for sample in data["samples"]:
            spans = tuple(_interval(v) for v in sample)
            if (
                not spans
                or any(lo == hi for lo, hi in spans)
                or any(b[0] < a[1] for a, b in zip(spans, spans[1:]))
            ):
                raise ValueError("profile requires ordered disjoint device compute windows")
            self.windows.append(spans)
        if not self.windows:
            raise ValueError("profile requires at least one device sample")
        self.quiet_windows = None
        if "collective_free_samples" in data:
            quiet = [tuple(_interval(v) for v in sample) for sample in data["collective_free_samples"]]
            if len(quiet) != len(self.windows):
                raise ValueError("collective and compute sample inventories differ")
            for subset, original in zip(quiet, self.windows):
                i = 0
                previous = 0.0
                for left, right in subset:
                    while i < len(original) and original[i][1] <= left:
                        i += 1
                    if not (
                        previous <= left < right
                        and i < len(original)
                        and original[i][0] <= left
                        and right <= original[i][1]
                    ):
                        raise ValueError(
                            "collective-free windows must be an ordered subset of compute windows"
                        )
                    previous = right
            self.quiet_windows = quiet
        self.regions = []
        self.groups = []
        seen = set()
        for row in (*data["regions"], *data.get("native_groups", ())):
            name = row["name"]
            replaces = _names(row.get("replaces", ()))
            if name in seen or not isinstance(name, str) or not name:
                raise ValueError("duplicate or invalid region")
            if replaces:
                children = {n for n in self.recompute if n.startswith(name + ".")}
                if set(replaces) != children or name in self.recompute or not row["native_bytes"]:
                    raise ValueError("native group must replace its complete checkpoint subtree")
            elif name not in self.recompute:
                raise ValueError("region must be an anchor checkpoint")
            seen.add(name)
            samples = []
            for sample in row["samples"]:
                packets = tuple(Packet.parse(p) for p in sample["packets"])
                if len({p.key for p in packets}) != len(packets):
                    raise ValueError("duplicate version inside one region sample")
                samples.append(RegionSample(_interval(sample["removed_compute"]), packets))
            if len(samples) != len(self.windows):
                raise ValueError("sample inventory does not match device windows")
            first, last, until = (_integer(row[k]) for k in ("first_op", "last_op", "native_until"))
            if not first <= last <= until < self.op_count:
                raise ValueError("invalid phase allocation bounds")
            inventory = self.groups if replaces else self.regions
            inventory.append(
                Region(
                    name,
                    _number(row["saved_seconds"]),
                    _integer(row["workspace"]),
                    _integer(row["native_bytes"]),
                    first,
                    last,
                    until,
                    tuple(samples),
                    replaces,
                )
            )

    def _admit(self, opened, held, pressure, memory, host_headroom, *, windows=None):
        bounds = [(r.first_op, r.last_op, r.workspace, None) for r in (*opened, *held)]
        bounds += [(r.first_op, r.native_until, r.native, None) for r in held]
        # PhaseCapacity.try_reserve is transactional but mutates on success.
        # Each proposed *complete* allocation starts with the observed report.
        phase = PhaseCapacity(memory[0], op_count=self.op_count, limit_bytes=memory[1])
        if not phase.try_reserve(bounds):
            return None
        largest = 0
        for i, free in enumerate(self.windows if windows is None else windows):
            cal = _Calendar(free, [r.samples[i].removed_compute for r in (*opened, *held)])
            versions = {}
            for r in opened:
                if not r.samples[i].packets:
                    return None
                for p in r.samples[i].packets:
                    if versions.setdefault(p.key, p) != p:
                        return None
            volume = sum(p.size for p in versions.values())
            if volume > host_headroom:
                return None
            largest = max(largest, volume)

            def deadline(p):
                return p.first_pressure + (p.peak_pressure - p.first_pressure) * pressure

            written = {}
            for p in sorted(versions.values(), key=lambda p: (deadline(p), p.ready, p.key)):
                finish = cal.reserve(p.ready, deadline(p), p.d2h)
                if finish is None:
                    return None
                written[p.key] = finish
            for p in sorted(versions.values(), key=lambda p: (p.use, p.prefetch, p.key)):
                if cal.reserve(max(p.prefetch, written[p.key]), p.use, p.h2d) is None:
                    return None
        return largest

    def portfolio(self, phase_report, *, device_limit, host_headroom):
        if phase_report is None or phase_report.get("trace_signature") != self.signature:
            raise ValueError("device windows do not match the captured anchor graph")
        memory = (phase_report, _integer(device_limit))
        host_headroom = _integer(host_headroom)
        if not self.regions:
            return ()
        ordered = sorted(self.regions, key=lambda r: -r.saved)
        out = []
        known = set()

        def emit(opened, held, volume, mode):
            off = tuple(sorted(r.name for r in opened))
            keep = tuple(sorted(r.name for r in held))
            if (off, keep) not in known and (off or keep):
                known.add((off, keep))
                replaced = tuple(sorted(n for r in held for n in r.replaces))
                out.append(
                    Structure(off, keep, sum(r.saved for r in (*opened, *held)), volume, mode, replaced)
                )

        # Change the execution boundary as well as the memory destination.
        # An opaque native parent replaces ALL its checkpointed descendants.
        # Account for its whole native lifetime without crediting any former
        # checkpoint bytes. Unrelated regions can spend separate DMA windows.
        held = []
        covered = set()
        groups = sorted(self.groups, key=lambda r: (-r.saved / max(1, r.native), -r.saved, r.name))
        for r in groups:
            if covered.intersection(r.replaces):
                continue
            if self._admit([], held + [r], 1.0, memory, host_headroom) is not None:
                held.append(r)
                covered.update(r.replaces)
        if held:
            emit([], held, 0, "native_groups")
            for mode, windows in (
                ("native_groups_with_copy", self.windows),
                ("native_groups_uncontended_copy", self.quiet_windows),
            ):
                opened = []
                volume = 0
                for r in ordered:
                    if r.name in covered:
                        continue
                    value = self._admit(opened + [r], held, 1.0, memory, host_headroom, windows=windows)
                    if value is not None:
                        opened.append(r)
                        volume = value
                emit(opened, held, volume, mode)
            emit([], held[: max(1, len(held) // 2)], 0, "native_groups")

        if self.quiet_windows is not None:
            opened = []
            volume = 0
            for r in ordered:
                value = self._admit(
                    opened + [r], [], 1.0, memory, host_headroom, windows=self.quiet_windows
                )
                if value is not None:
                    opened.append(r)
                    volume = value
            emit(opened, [], volume, "collective_free_copy")

        # Sweep *whole* allocation objectives, including partial frontier
        # snapshots. Rank by benefit, and separately by benefit per packet.
        for pressure, by_packet in ((1.0, False), (0.45, True), (0.0, False)):
            candidates = (
                ordered
                if not by_packet
                else sorted(ordered, key=lambda r: -r.saved / max(1, len(r.samples[0].packets)))
            )
            opened = []
            history = []
            for r in candidates:
                vol = self._admit(opened + [r], [], pressure, memory, host_headroom)
                if vol is not None:
                    opened.append(r)
                    history.append((tuple(opened), vol))
            if history:
                for idx in (len(history) - 1, (len(history) - 1) // 2, (len(history) - 1) // 4):
                    batch, vol = history[idx]
                    emit(batch, [], vol, f"copy_pressure_{pressure}")
        # A new saved activation can also stay native when its lifetime fits
        # the phase envelope. Let raw offload and native keep compete for
        # expensive work; never subtract the old checkpoint from the reserve.
        opened = []
        held = []
        history = []
        for r in ordered:
            if r.native and self._admit(opened, held + [r], 1.0, memory, host_headroom) is not None:
                held.append(r)
            elif self._admit(opened + [r], held, 1.0, memory, host_headroom) is not None:
                opened.append(r)
            else:
                continue
            volume = self._admit(opened, held, 1.0, memory, host_headroom)
            history.append((tuple(opened), tuple(held), volume))
        for idx in sorted({len(history) - 1, (len(history) - 1) // 2}, reverse=True):
            if idx >= 0:
                emit(*history[idx], "joint_keep_copy")
        return tuple(out)
