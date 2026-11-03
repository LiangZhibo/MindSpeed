"""Bounded conservative capacity estimates over observed warmup phases.

Each bucket uses the largest observed cumulative allocator peak and rounds
candidate lifetimes outward. Independent native allocations add; a specified
shared planner pool contributes its maximum envelope, not one pool per region.
This orders experiments only: changed graphs still require fresh traces and
complete-step capacity admission.
"""
from __future__ import annotations

import math


class PhaseCapacity:
    def __init__(self, report, *, op_count, limit_bytes, max_buckets=256):
        if report.get('truncated') or report.get('forward_end_op') is None:
            raise ValueError('incomplete phase memory evidence')
        if not isinstance(op_count,int) or op_count < 1 or not isinstance(max_buckets,int) or max_buckets < 1:
            raise ValueError('positive bounded phase dimensions required')
        if not isinstance(limit_bytes,int) or limit_bytes < 0:
            raise ValueError('nonnegative byte limit required')
        summary = report.get('capacity_summary')
        if summary is None:
            summary = summarize_phase_capacity(report,op_count=op_count,max_buckets=max_buckets)
        self.width = summary['width']
        self.op_count, self.limit = op_count, limit_bytes
        self.peak = list(summary['peak_bytes'])
        if (summary['op_count'] != op_count or isinstance(self.width,bool) or not isinstance(self.width,int)
                or self.width < 1 or len(self.peak) != math.ceil(op_count/self.width)
                or not self.peak or len(self.peak)>max_buckets
                or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in self.peak)
                or any(a>b for a,b in zip(self.peak,self.peak[1:]))):
            raise ValueError('invalid compact phase capacity summary')
        if max(self.peak) > self.limit:
            raise ValueError('observed phase peak already exceeds capacity')
        self._total = self.peak.copy()
        self._pools = {}
        self.accepted = 0

    def try_reserve(self, bounds):
        """Atomically admit (first_op,last_op,bytes,shared_pool_or_None) bounds."""
        total = self._total.copy()
        pools = {key:values.copy() for key,values in self._pools.items()}
        for first,last,size,pool in bounds:
            if any(isinstance(x,bool) or not isinstance(x,int) for x in (first,last,size)):
                raise ValueError('phase bounds require integer positions and bytes')
            if not 0 <= first <= last < self.op_count or size < 0:
                raise ValueError('invalid phase allocation bound')
            if pool is not None and (not isinstance(pool,str) or not pool):
                raise ValueError('shared pool requires an explicit identity')
            if size == 0:
                continue
            lo,hi = first//self.width,last//self.width+1
            if pool is None:
                values = [value+size for value in total[lo:hi]]
                if max(values)>self.limit:
                    return False
                total[lo:hi]=values
            else:
                envelope = pools.setdefault(pool,[0]*len(total))
                for bucket in range(lo,hi):
                    if envelope[bucket]>=size:
                        continue
                    total[bucket] += size-envelope[bucket]
                    if total[bucket]>self.limit:
                        return False
                    envelope[bucket]=size
        self._total, self._pools = total,pools
        self.accepted += 1
        return True

    @property
    def estimated_peak_bytes(self):
        return max(self._total)



def summarize_phase_capacity(report, *, op_count, max_buckets=256):
    """Validate dense raw samples once at collection, before strategy search."""
    if report.get('truncated') or report.get('forward_end_op') is None:
        raise ValueError('incomplete phase memory evidence')
    samples=report['samples']
    if len(samples)!=op_count or any(row[0]!=i for i,row in enumerate(samples)):
        raise ValueError('phase samples must cover every matching trace operator')
    if op_count<1 or max_buckets<1:
        raise ValueError('positive phase dimensions required')
    width=math.ceil(op_count/max_buckets)
    peaks=[0]*math.ceil(op_count/width)
    for row in samples:
        if len(row)!=5 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in row):
            raise ValueError('invalid phase allocator counters')
        index,allocated,peak,reserved,peak_reserved=row
        if allocated>peak or reserved>peak_reserved or allocated>reserved:
            raise ValueError('inconsistent phase allocator counters')
        bucket=index//width
        peaks[bucket]=max(peaks[bucket],peak)
    return dict(op_count=op_count,width=width,peak_bytes=peaks)


def compact_phase_report(report):
    if report is None or report.get('capacity_summary') is None:
        return None
    keys=('trace_signature','columns','regions','forward_end_op','truncated','capacity_summary')
    return {key:report[key] for key in keys}


def merge_phase_reports(reports):
    """Pointwise max across ranks only when execution identities/lifetimes agree."""
    if not reports or any(r is None for r in reports):
        return None
    first = reports[0]
    keys = ('trace_signature', 'columns', 'regions', 'forward_end_op')
    if (not first.get('trace_signature') or any(r.get('truncated') for r in reports)
            or any(any(r.get(k) != first.get(k) for k in keys) for r in reports[1:])):
        return None
    summaries=[r.get('capacity_summary') for r in reports]
    if all(s is not None for s in summaries):
        baseline=summaries[0]
        if any(s['op_count']!=baseline['op_count'] or s['width']!=baseline['width']
               or len(s['peak_bytes'])!=len(baseline['peak_bytes']) for s in summaries[1:]):
            return None
        merged=dict(baseline,peak_bytes=[max(values) for values in zip(*(s['peak_bytes'] for s in summaries))])
        result={k:v for k,v in first.items() if k!='samples'}
        return dict(result,capacity_summary=merged,rank_count=len(reports),reduction='per-bucket max rank capacity')
    if any('samples' not in r or len(r['samples'])!=len(first.get('samples',())) for r in reports):
        return None
    samples = []
    for index, rows in enumerate(zip(*(r['samples'] for r in reports))):
        if any(len(row) != 5 or row[0] != index for row in rows):
            return None
        samples.append((index, *(max(row[column] for row in rows) for column in range(1, 5))))
    return dict(first, samples=samples, rank_count=len(reports), reduction='per-operator max rank counters')
