"""Bounded proposals from measured region and transport costs.

Names are opaque adapter-provided identities. This module does not inspect a
model family, layer number, tensor-size magic constant or codec codebook. The
estimates only order experiments; full-step measurements and memory admission
remain the controller's authority. Missing evidence keeps the incumbent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .controller import JointPolicy, Trial

_MIB = 1024 ** 2


def _finite_nonnegative(value, label):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative')


@dataclass(frozen=True)
class TransferDemand:
    """One storage roundtrip; slack excludes the transfer itself."""
    nbytes: int
    dtype: str = "unknown"
    d2h_slack_ms: float = 0.0
    h2d_slack_ms: float = 0.0

    def __post_init__(self):
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int) or self.nbytes < 1:
            raise ValueError('transfer bytes must be a positive integer')
        for field in ('d2h_slack_ms', 'h2d_slack_ms'):
            _finite_nonnegative(getattr(self, field), field)


@dataclass(frozen=True)
class TransportMeasurement:
    """A calibrated profile, including measured host submission costs.

    Sizes outside the calibrated support fall back to raw transfers. Temporary
    device and Host reservation bounds include worst-case raw codec packets;
    compression ratio is NEVER used to discount those capacity requirements.
    The executable codec selector must implement the same support predicate.
    """
    key: str
    d2h_ms_per_mib: float
    h2d_ms_per_mib: float
    packet_ratio: float = 1.0
    encode_ms_per_mib: float = 0.0
    decode_ms_per_mib: float = 0.0
    submission_ms: float = 0.0
    min_bytes: int = 1
    max_bytes: int = 2**63 - 1
    temporary_device_bytes: int = 0
    additional_host_reservation_bytes: int = 0
    lossless_verified: bool = False
    dtypes: tuple[str, ...] = ("bfloat16",)

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError('transport profile key is required')
        for field in ('d2h_ms_per_mib', 'h2d_ms_per_mib', 'packet_ratio',
                      'encode_ms_per_mib', 'decode_ms_per_mib', 'submission_ms'):
            _finite_nonnegative(getattr(self, field), field)
        if self.packet_ratio <= 0:
            raise ValueError('packet ratio must be positive')
        for field in ('min_bytes', 'max_bytes', 'temporary_device_bytes', 'additional_host_reservation_bytes'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{field} must be nonnegative integer bytes')
        if not 0 < self.min_bytes <= self.max_bytes:
            raise ValueError('invalid measured size support')
        if self.key == "raw":
            object.__setattr__(self, "lossless_verified", True)
        if self.key == 'raw' and (self.packet_ratio != 1 or self.encode_ms_per_mib
                                  or self.decode_ms_per_mib or self.submission_ms):
            raise ValueError('raw profile cannot claim codec behavior')

    def supports(self, demand):
        return self.key == "raw" or (self.lossless_verified and demand.dtype in self.dtypes
                                    and self.min_bytes <= demand.nbytes <= self.max_bytes)

    def exposed_ms(self, demand: TransferDemand, raw: TransportMeasurement):
        profile = self if self.supports(demand) else raw
        mib = demand.nbytes / _MIB
        codec = mib * (profile.encode_ms_per_mib + profile.decode_ms_per_mib) + profile.submission_ms
        outbound = mib * profile.packet_ratio * profile.d2h_ms_per_mib
        inbound = mib * profile.packet_ratio * profile.h2d_ms_per_mib
        return codec + max(0.0, outbound - demand.d2h_slack_ms) + max(0.0, inbound - demand.h2d_slack_ms)


def _traffic_totals(profile, raw, demands):
    codec = outbound = inbound = late_outbound = late_inbound = 0.0
    for demand in demands:
        chosen = profile if profile.supports(demand) else raw
        mib = demand.nbytes / _MIB
        codec += mib * (chosen.encode_ms_per_mib + chosen.decode_ms_per_mib) + chosen.submission_ms
        out = mib * chosen.packet_ratio * chosen.d2h_ms_per_mib
        back = mib * chosen.packet_ratio * chosen.h2d_ms_per_mib
        outbound += out
        inbound += back
        late_outbound += max(0., out - demand.d2h_slack_ms)
        late_inbound += max(0., back - demand.h2d_slack_ms)
    return codec, outbound, inbound, late_outbound, late_inbound


def _exposed_traffic(totals, windows):
    codec, outbound, inbound, late_outbound, late_inbound = totals
    # Multiple packets cannot independently claim the same DMA window.
    return codec + max(late_outbound, outbound - windows[0]) + max(late_inbound, inbound - windows[1])


def _add_traffic(left, right):
    return tuple(a + b for a, b in zip(left, right))


@dataclass(frozen=True)
class RegionAlternative:
    """One adapter-certified replacement of a non-overlapping checkpoint cut.

    An alternative may replace a coarse checkpoint with fine operator groups,
    or remove an expensive group and retain its cheap subgroups. Added storage
    demands and device deltas must come from measurements of that structure or
    an explicitly recorded equivalent repeated region. No raw op ID survives a
    new trace. Extrapolated evidence is a proposal estimate, never qualification.
    """
    remove_recompute: tuple[str, ...]
    add_recompute: tuple[str, ...]
    avoided_recompute_ms: float
    additional_transfers: tuple[TransferDemand, ...]
    additional_device_bytes: int = 0
    additional_host_bytes: int = 0
    remove_resident: tuple[str, ...] = ()
    add_resident: tuple[str, ...] = ()

    def __post_init__(self):
        _finite_nonnegative(self.avoided_recompute_ms, 'avoided recomputation')
        for field in ('additional_device_bytes', 'additional_host_bytes'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('memory deltas must be conservative nonnegative integer bounds')
        for field in ('remove_recompute', 'add_recompute', 'remove_resident', 'add_resident'):
            values = getattr(self, field)
            if not isinstance(values, tuple) or len(values) != len(set(values)) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError('region identities must be unique nonempty tuples')
        if not self.remove_recompute:
            raise ValueError('an alternative must replace at least one recomputed region')


class MeasuredJointNeighborhood:
    """O(profiles * regions log regions), without a combinatorial search.

    The provider is called once, after the initial trial. It returns frozen
    evidence from the bounded warmup. Unique total DMA overlap budgets must
    exclude removable recomputation; overlapping per-storage slack is not summed.
    All ranks must supply identical reduced
    evidence. ``current_transfers`` is mandatory: compression of already hidden
    transfers can cost more than it saves and must be charged to the proposal.
    """
    def __init__(self, evidence_provider, *, max_profiles=4, max_regions=4096, minimum_gain_ms=0.0):
        if max_profiles < 1 or max_regions < 1:
            raise ValueError('positive bounded profile/region limits required')
        _finite_nonnegative(minimum_gain_ms, 'minimum gain')
        self.provider = evidence_provider
        self.max_profiles, self.max_regions = max_profiles, max_regions
        self.minimum_gain_ms = minimum_gain_ms
        self.estimates = []
        self._pending = None

    def __call__(self, trials: tuple[Trial, ...]) -> JointPolicy | None:
        if not trials or not trials[0].feasible:
            return None
        if self._pending is None:
            self._pending = self._build(trials[0])
        tested = {trial.policy for trial in trials}
        return next((policy for policy in self._pending if policy not in tested), None)

    def _build(self, initial):
        data = self.provider()
        if data is None:
            return []
        base = initial.policy
        if base.recompute_regions is None or base.codec_key != 'raw':
            raise ValueError('measured neighborhood requires an explicit raw checkpoint cut')
        raw = data['raw']
        if raw.key != 'raw':
            raise ValueError('raw transport measurement required')
        profiles = (raw, *tuple(data.get('codecs', ())))
        alternatives = tuple(data['regions'])
        if len(profiles) > self.max_profiles or len(alternatives) > self.max_regions:
            raise ValueError('calibration evidence exceeds bounded search capacity')
        if len({p.key for p in profiles}) != len(profiles):
            raise ValueError('transport profile keys must be unique')
        current = tuple(data['current_transfers'])
        windows = (data['d2h_overlap_ms'], data['h2d_overlap_ms'])
        for window in windows:
            _finite_nonnegative(window, 'total unique DMA overlap window')
        if len(current) + sum(len(a.additional_transfers) for a in alternatives) > 32768:
            raise ValueError('transfer evidence exceeds bounded search capacity')
        base_cost = _exposed_traffic(_traffic_totals(raw, raw, current), windows)
        conflicts = tuple(frozenset(pair) for pair in data.get('conflicts', ()))
        device_headroom = data['device_headroom_bytes']
        host_headroom = data['host_headroom_bytes']
        for name, value in [('device headroom', device_headroom), ('host headroom', host_headroom)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be nonnegative integer bytes')
        ranked = []
        for profile in profiles:
            if not profile.lossless_verified:
                continue
            device = profile.temporary_device_bytes
            host = profile.additional_host_reservation_bytes
            if device > device_headroom or host > host_headroom:
                continue
            # Windows count only unavoidable compute, excluding recomputation
            # which an alternative removes. The provider must measure their
            # union, not sum overlapping per-storage opportunities.
            traffic = _traffic_totals(profile, raw, current)
            gain = base_cost - _exposed_traffic(traffic, windows)
            scored = []
            for i, alternative in enumerate(alternatives):
                added = _traffic_totals(profile, raw, alternative.additional_transfers)
                cost = _exposed_traffic(_add_traffic(traffic, added), windows) - _exposed_traffic(traffic, windows)
                scored.append((alternative.avoided_recompute_ms - cost, i, alternative, added))
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected, resident = set(base.recompute_regions), set(base.resident_regions)
            accepted = 0
            for benefit, _, alternative, added in scored:
                next_traffic = _add_traffic(traffic, added)
                benefit = alternative.avoided_recompute_ms - (
                    _exposed_traffic(next_traffic, windows) - _exposed_traffic(traffic, windows))
                if benefit <= 0:
                    continue
                if not set(alternative.remove_recompute) <= selected or not set(alternative.remove_resident) <= resident:
                    continue
                next_device = device + alternative.additional_device_bytes
                next_host = host + alternative.additional_host_bytes
                if next_device > device_headroom or next_host > host_headroom:
                    continue
                next_selected = selected.difference(alternative.remove_recompute).union(alternative.add_recompute)
                next_resident = resident.difference(alternative.remove_resident).union(alternative.add_resident)
                if any(pair <= (next_selected | next_resident) for pair in conflicts):
                    continue
                try:
                    candidate = replace(base, recompute_regions=tuple(sorted(next_selected)),
                                        resident_regions=tuple(sorted(next_resident)), codec_key=profile.key)
                except ValueError:
                    continue  # Parent/child conflicts are never resolved by dropping arbitrary groups.
                selected, resident, device, host = next_selected, next_resident, next_device, next_host
                gain += benefit
                traffic = next_traffic
                accepted += 1
            candidate = replace(base, recompute_regions=tuple(sorted(selected)), resident_regions=tuple(sorted(resident)),
                                codec_key=profile.key)
            self.estimates.append(dict(codec_key=profile.key, gain_ms=gain, alternatives=accepted,
                                       additional_device_bound=device, additional_host_bound=host))
            if gain > self.minimum_gain_ms and candidate != base:
                ranked.append((gain, candidate))
        ranked.sort(key=lambda item: -item[0])
        return [policy for _, policy in ranked]
