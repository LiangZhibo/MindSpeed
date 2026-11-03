"""Bounded measured packet-latency estimates, without a model identity."""
from bisect import bisect_left
import math


def validate_transfer_samples(samples):
    if not isinstance(samples,(tuple,list)) or len(samples)>4:
        raise ValueError('transfer calibration requires at most four size points')
    previous=0
    for point in samples:
        if not isinstance(point,(tuple,list)) or len(point)!=3:
            raise ValueError('transfer calibration points are bytes, D2H ms, H2D ms')
        size,*times=point
        if isinstance(size,bool) or not isinstance(size,int) or size<=previous:
            raise ValueError('transfer calibration sizes must be positive and strictly increasing')
        if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0 for t in times):
            raise ValueError('transfer calibration latencies must be positive and finite')
        previous=size


def transfer_duration_ms(nbytes, bandwidth_gibps, samples=(), direction='d2h'):
    """Interpolate latency inside measured support; use legacy rate outside it.

    A cumulative maximum preserves slow observations when a noisy larger copy
    appears cheaper than a smaller one. These are scheduling/candidate estimates,
    not completion fences or a claim of exposed training latency.
    """
    if direction not in ('d2h','h2d'):
        raise ValueError('unknown transfer direction')
    if not math.isfinite(bandwidth_gibps) or bandwidth_gibps<=0:
        raise ValueError('positive finite transfer bandwidth required')
    if not samples or nbytes<samples[0][0] or nbytes>samples[-1][0]:
        return nbytes/(bandwidth_gibps*2**30)*1000
    index=1 if direction=='d2h' else 2
    sizes=[p[0] for p in samples];times=[];largest=0.
    for point in samples:
        largest=max(largest,point[index]);times.append(largest)
    right=bisect_left(sizes,nbytes)
    if sizes[right]==nbytes:return times[right]
    fraction=(nbytes-sizes[right-1])/(sizes[right]-sizes[right-1])
    return times[right-1]+fraction*(times[right]-times[right-1])
