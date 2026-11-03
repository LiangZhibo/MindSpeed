"""Conservative packet costs from bounded, losslessly checked calibration.

These estimates rank experiments, not training speed. Size support is exact;
unknown sizes/dtypes must retain raw transport. Codec residual cost is charged
outside the overlap window, so a good ratio cannot justify already hidden DMA.
"""
from dataclasses import dataclass
import math
from statistics import median


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _bytes(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be nonnegative integer bytes')
    return value


@dataclass(frozen=True)
class CalibratedPacketCost:
    nbytes: int
    raw_d2h_ms: float
    raw_h2d_ms: float
    codec_d2h_ms: float
    codec_h2d_ms: float
    packet_ratio: float
    extra_device_bytes: int
    extra_host_bytes: int
    sample_hashes: tuple[str, ...] = ()

    def __post_init__(self):
        if _bytes(self.nbytes, 'packet size') == 0:
            raise ValueError('packet size must be positive')
        for name in ('raw_d2h_ms','raw_h2d_ms','codec_d2h_ms','codec_h2d_ms','packet_ratio'):
            _positive(getattr(self,name),name)
        _bytes(self.extra_device_bytes,'device bound')
        _bytes(self.extra_host_bytes,'Host bound')

    def supports(self, nbytes, dtype):
        return nbytes == self.nbytes and dtype == 'bfloat16'

    def capacity_fits(self, device_headroom, host_headroom):
        # Packet ratio never discounts worst-case working or reservation bytes.
        return (self.extra_device_bytes <= _bytes(device_headroom,'device headroom')
                and self.extra_host_bytes <= _bytes(host_headroom,'Host headroom'))

    def exposed_costs(self, d2h_slack_ms, h2d_slack_ms):
        """Return (raw, codec) estimated exposed milliseconds for this packet.

        Isolated inclusive codec timing does not separately identify encode,
        decode and submission. Charge the non-DMA residual conservatively even
        when DMA overlaps. This is a scheduling estimate, never a measured stall.
        Overlapping packets still require a shared queue/window budget upstream.
        """
        raw = codec = 0.0
        for raw_ms, codec_ms, slack in ((self.raw_d2h_ms,self.codec_d2h_ms,d2h_slack_ms),
                                       (self.raw_h2d_ms,self.codec_h2d_ms,h2d_slack_ms)):
            if isinstance(slack,bool) or not isinstance(slack,(int,float)) or not math.isfinite(slack) or slack < 0:
                raise ValueError('overlap slack must be finite and nonnegative')
            packet_dma = raw_ms * self.packet_ratio
            residual = max(0.0, codec_ms - packet_dma)
            raw += max(0.0,raw_ms-slack)
            codec += residual + max(0.0,packet_dma-slack)
        return raw, codec


def profiles_from_calibration(report, *, max_profiles=3):
    """Parse real BF16 calibration rows; missing/failed rows supply no profile.

    Use per-snapshot medians, the fastest observed raw snapshot and slowest
    codec snapshot for each direction. This retains adverse observations rather
    than declaring one favorable activation representative of a whole model.
    The report comes from the BF16-only actual-activation calibration boundary.
    No warmup sample participates in the measured cost estimate.
    """
    if isinstance(max_profiles,bool) or not isinstance(max_profiles,int) or not 1 <= max_profiles <= 3:
        raise ValueError('one to three packet profiles required')
    if report.get('status') != 'complete':
        return ()
    groups = {}
    for row in report.get('rows',()):
        if row.get('status') != 'complete':
            continue
        size = _bytes(row['bytes'],'packet size')
        if not size or size % 2:
            raise ValueError('BF16 calibration requires positive even bytes')
        measurements = row.get('measurements',())
        if len(measurements) < 2:
            continue
        times = {name:[] for name in ('raw_d2h','raw_h2d','codec_d2h','codec_h2d')}
        ratios = []
        for sample in measurements:
            results = sample['results']
            for name in ('raw','codec'):
                result = results[name]
                if result.get('lossless_exact') is not True:
                    raise ValueError('calibration roundtrip is not losslessly verified')
                for direction in ('d2h','h2d'):
                    times[f'{name}_{direction}'].append(1000*_positive(result[f'{direction}_inclusive_seconds'],'transfer duration'))
                packet = _bytes(result['packet_bytes'],'encoded packet size')
                if not packet:
                    raise ValueError('encoded packet size must be positive')
                if name == 'codec':ratios.append(packet/size)
        groups.setdefault(size,[]).append(dict(
            **{name:median(values) for name,values in times.items()},ratio=max(ratios),
            device=_bytes(row['required_extra_device_bytes'],'device bound'),
            host=_bytes(row['required_extra_host_bytes'],'Host bound'),sample_hash=row['source_sha256']))
    # Bound support by measured byte mass, without fixed model/tensor thresholds.
    sizes = sorted(groups,key=lambda size:(-size*len(groups[size]),size))[:max_profiles]
    return tuple(CalibratedPacketCost(size,
        min(x['raw_d2h'] for x in groups[size]),min(x['raw_h2d'] for x in groups[size]),
        max(x['codec_d2h'] for x in groups[size]),max(x['codec_h2d'] for x in groups[size]),
        max(x['ratio'] for x in groups[size]),max(x['device'] for x in groups[size]),
        max(x['host'] for x in groups[size]),tuple(sorted({x['sample_hash'] for x in groups[size]})))
        for size in sorted(sizes))


def common_profiles(per_rank):
    """Conservative common size support after all rank reports are gathered.

    A missing or unsupported rank excludes that size everywhere. Rank-local
    codebooks may differ, but no rank may choose a profile from another rank's
    favorable timing or capacity evidence. Empty support means raw transport.
    """
    if not per_rank:
        return ()
    indexed = []
    for profiles in per_rank:
        if len(profiles) > 3 or any(not isinstance(p, CalibratedPacketCost) for p in profiles):
            raise ValueError('each rank must provide at most three calibrated packet profiles')
        values = {p.nbytes:p for p in profiles}
        if len(values) != len(profiles):
            raise ValueError('duplicate calibrated size on one rank')
        indexed.append(values)
    sizes = set(indexed[0]).intersection(*(set(values) for values in indexed[1:]))
    result = []
    for size in sorted(sizes):
        values = [rank[size] for rank in indexed]
        result.append(CalibratedPacketCost(size,
            min(p.raw_d2h_ms for p in values), min(p.raw_h2d_ms for p in values),
            max(p.codec_d2h_ms for p in values), max(p.codec_h2d_ms for p in values),
            max(p.packet_ratio for p in values), max(p.extra_device_bytes for p in values),
            max(p.extra_host_bytes for p in values),
            tuple(sorted({h for p in values for h in p.sample_hashes}))))
    return tuple(result)
