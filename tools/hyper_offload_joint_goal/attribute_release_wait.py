"""Locate H2D release host waits relative to recorded hardware intervals."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import json
from pathlib import Path


def union(intervals):
    out = []
    for left, right in sorted(intervals):
        if out and left <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], right))
        else:
            out.append((left, right))
    return out


def duration(intervals):
    return sum(right - left for left, right in union(intervals)) / 1e6


def intersect(first, second):
    first, second = union(first), union(second)
    i = j = 0
    out = []
    while i < len(first) and j < len(second):
        left, right = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if left < right:
            out.append((left, right))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return out


def interval(event):
    return float(event['ts']), float(event['ts']) + float(event['dur'])


parser = argparse.ArgumentParser()
parser.add_argument('profile', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
reports = []
for folder in sorted(args.profile.glob('*/ASCEND_PROFILER_OUTPUT')):
    data = json.loads((folder / 'trace_view.json').read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    hardware = next(event['pid'] for event in events if event.get('ph') == 'M'
                    and event.get('name') == 'process_name' and event['args']['name'] == 'Ascend Hardware')
    spans = [event for event in events if event.get('ph') == 'X' and 'dur' in event]
    syncs = defaultdict(list)
    for event in spans:
        if event['name'].split('@')[-1] == 'aclrtSynchronizeEvent':
            syncs[event['tid']].append(interval(event))
    for values in syncs.values():
        values.sort()
    starts = {thread: [left for left, _ in values] for thread, values in syncs.items()}
    release_waits = []
    for event in spans:
        if event['name'] != 'HyperOffload::release_device':
            continue
        left, right = interval(event)
        times, values = starts.get(event['tid'], []), syncs.get(event['tid'], [])
        release_waits.extend((a, b) for a, b in values[bisect_left(times, left):bisect_left(times, right)]
                             if b <= right)
    kernels = list(csv.DictReader((folder / 'kernel_details.csv').open()))
    compute, hccl = [], []
    for kernel in kernels:
        start = float(kernel['Start Time(us)'])
        target = hccl if kernel['Accelerator Core'] == 'COMMUNICATION' else compute
        target.append((start, start + float(kernel['Duration(us)'])))
    dma = [interval(event) for event in spans if event['pid'] == hardware
           and event.get('args', {}).get('Task Type') == 'PCIE_DMA_SQE']
    rows = []
    for event in spans:
        if not event['name'].startswith('ProfilerStep#'):
            continue
        step = [interval(event)]
        waits = intersect(release_waits, step)
        seconds = duration(waits)
        rows.append(dict(step=event['name'], release_event_sync_seconds=seconds,
                         overlapping_compute_seconds=duration(intersect(waits, compute)),
                         outside_compute_seconds=seconds - duration(intersect(waits, compute)),
                         overlapping_hccl_spans_seconds=duration(intersect(waits, hccl)),
                         overlapping_dma_seconds=duration(intersect(waits, dma)),
                         outside_compute_hccl_dma_spans_seconds=seconds - duration(intersect(waits, compute + hccl + dma))))
    reports.append(dict(profile=folder.parent.name, steps=rows))
    del data, events, spans, kernels
report = dict(caveat='Instrumented temporal association only, not a causal speedup bound. HCCL spans may '
              'contain waits; overlap with a span is not proof that hardware is doing useful work. '
              'Only aclrtSynchronizeEvent calls contained in same-thread HyperOffload::release_device '
              'scopes are counted.', ranks=reports)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(reports, indent=2))
