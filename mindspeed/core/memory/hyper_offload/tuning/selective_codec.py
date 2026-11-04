"""Immutable graph-specific native-codec candidates with raw fallback.

Actual BF16 calibration supplies common size support. Node IDs are taken only
from the corresponding captured graph's peak-covering gaps. A shorter packet
may justify a full-step traffic-budget experiment even if isolated inclusive
codec time is worse; this is never reported as a measured training speedup.
"""
from dataclasses import dataclass
import hashlib
import json

from ..runtime.pinned_memory import _align_to_bucket


@dataclass(frozen=True)
class CodecScope:
    key: str
    structure_key: tuple
    nodes: tuple[tuple[int, str, int, int], ...]
    purpose: str
    extra_device_bytes: int
    extra_host_bytes: int
    estimated_link_bytes_saved: float
    raw_h2d_lateness_ms: float
    codec_h2d_lateness_ms: float
    raw_peak_bytes: int
    codec_peak_bytes: int


class FrozenNodeCodec:
    """A fixed selection recipe sharing one session-owned native backend."""
    def __init__(self, backend, scope):
        self._backend = backend
        self.scope = scope
        self._nodes = frozenset(scope.nodes)

    def accepts(self, context):
        node = context.node
        return (node is not None and (node.op_id, node.phase, context.storage_id, context.original_bytes)
                in self._nodes and self._backend.accepts(context))

    def encode(self, source, stream):
        return self._backend.encode(source, stream)

    def decode(self, packet, metadata, stream):
        return self._backend.decode(packet, metadata, stream)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def codec_scopes(model, profiles, structure_key, *, device_headroom, host_headroom, prior_origins=()):
    """Offer selective exposed-queue and reduced-traffic experiments.

    Capacity reserves worst-case packet memory for the entire selected scope,
    not just an estimated overlap: all decoder packets may be pending, plus
    one bounded encoder packet and padding. Existing cached raw Host buffers
    are not credited away; encoded Host reservations are charged in full.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (device_headroom, host_headroom)):
        raise ValueError('codec scopes require nonnegative integer headroom')
    indexed = {p.nbytes:p for p in profiles}
    if not indexed:
        return ()
    selected = model.default_selection()
    raw = model.forecast(selected)
    outbound, inbound = {s.key:s for s in raw.d2h}, {s.key:s for s in raw.h2d}
    eligible = [o for o in selected if o.candidate.size_bytes in indexed
                and indexed[o.candidate.size_bytes].packet_ratio < 1.]
    def exposure(o):
        return (max(0., outbound[o.key].finish_ms-o.first_pressure_ms)
                + max(0., inbound[o.key].finish_ms-o.needed_ms))
    scopes, known = [], set()
    purposes = ('profiled_anchor', 'exposed_queue', 'transfer_capacity') if prior_origins else ('exposed_queue', 'transfer_capacity')
    for purpose in purposes:
        if purpose == 'profiled_anchor':
            # Storage identities and supported sizes come from this fresh
            # capture. A stale or ambiguous prior never creates a codec node.
            required = set(prior_origins)
            candidates = [o for o in eligible if (o.candidate.copy_start.op_id, o.candidate.size_bytes) in required]
            origins = {(o.candidate.copy_start.op_id, o.candidate.size_bytes) for o in candidates}
            if origins != required or len(candidates) != len(required):
                continue
        else:
            candidates = [o for o in eligible if purpose != 'exposed_queue' or exposure(o) > 0]
            candidates.sort(key=lambda o: (-exposure(o) if purpose == 'exposed_queue' else
                -o.candidate.size_bytes*(1.-indexed[o.candidate.size_bytes].packet_ratio), o.ready_ms))
        chosen, decoder_bytes, largest_encode, host_bytes = [], 0, 0, 4*1024**2
        for o in candidates:
            size = o.candidate.size_bytes
            padded = ((size+16383)//16384)*16384
            # Packet plus a conservative duplicate/padding output per decode.
            proposed_decode = decoder_bytes + 2*padded + 2048
            proposed_encode = max(largest_encode, padded+2048)
            proposed_host = host_bytes + _align_to_bucket(padded+2048) + 2048
            device_bytes = proposed_decode + proposed_encode + 4*1024**2
            if device_bytes > device_headroom or proposed_host > host_headroom:
                continue
            chosen.append(o)
            decoder_bytes, largest_encode, host_bytes = proposed_decode, proposed_encode, proposed_host
        nodes = tuple(sorted((o.candidate.copy_start.op_id, 'post', o.candidate.storage_id,
                              o.candidate.size_bytes) for o in chosen))
        if purpose == 'profiled_anchor' and len(chosen) != len(candidates):
            continue
        if not nodes or nodes in known:
            continue
        known.add(nodes)
        overrides = {o.key:(indexed[o.candidate.size_bytes].codec_d2h_ms,
                            indexed[o.candidate.size_bytes].codec_h2d_ms) for o in chosen}
        forecast = model.forecast(selected, transfer_costs=overrides)
        digest = hashlib.sha256(json.dumps((structure_key, nodes), sort_keys=True).encode()).hexdigest()[:20]
        scopes.append(CodecScope('splitzip-'+digest, structure_key, nodes, purpose,
            decoder_bytes+largest_encode+4*1024**2, host_bytes,
            2*sum(o.candidate.size_bytes*(1.-indexed[o.candidate.size_bytes].packet_ratio) for o in chosen),
            raw.max_h2d_lateness_ms, forecast.max_h2d_lateness_ms, raw.peak_bytes, forecast.peak_bytes))
    return tuple(scopes)
