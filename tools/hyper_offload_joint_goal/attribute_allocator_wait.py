"""Associate allocator API intervals with FSDP scopes and hardware timelines."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


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
names = {'aclrtSynchronizeDeviceWithTimeout', 'aclrtMallocPhysical', 'aclrtFreePhysical',
         'aclrtMapMem', 'aclrtUnmapMem', 'aclrtGetMemInfo'}
reports = []
for folder in sorted(args.profile.glob('*/ASCEND_PROFILER_OUTPUT')):
    data = json.loads((folder / 'trace_view.json').read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    hardware = next(event['pid'] for event in events if event.get('ph') == 'M'
                    and event.get('name') == 'process_name' and event['args']['name'] == 'Ascend Hardware')
    spans = [event for event in events if event.get('ph') == 'X' and 'dur' in event]
    scopes = defaultdict(list)
    for event in spans:
        if event['name'].startswith(('FSDP::', 'HyperOffload::')):
            left, right = interval(event)
            scopes[event['tid']].append((left, right, event['name']))
    indices = {}
    for thread, values in scopes.items():
        values.sort()
        maximum = float('-inf')
        prefix = []
        for left, right, _ in values:
            maximum = max(maximum, right)
            prefix.append(maximum)
        indices[thread] = ([row[0] for row in values], prefix, values)
    associated = []
    for event in spans:
        api = event['name'].split('@')[-1]
        if api not in names:
            continue
        left, right = interval(event)
        starts, prefix, values = indices.get(event['tid'], ([], [], []))
        index = bisect_right(starts, left) - 1
        container = None
        while index >= 0 and prefix[index] >= right:
            row = values[index]
            if row[1] >= right and (container is None or row[1] - row[0] < container[1] - container[0]):
                container = row
            index -= 1
        name = container[2] if container else 'outside_named_FSDP_or_HyperOffload'
        layer = re.search(r'model\.layers\.(\d+)', name)
        region = ('checkpoint_prefix_0_32' if int(layer[1]) < 33 else 'fine_suffix_33_47') if layer else name
        associated.append((api, region, name, (left, right)))
    kernels = list(csv.DictReader((folder / 'kernel_details.csv').open()))
    compute, hccl = [], []
    for kernel in kernels:
        left = float(kernel['Start Time(us)'])
        (hccl if kernel['Accelerator Core'] == 'COMMUNICATION' else compute).append(
            (left, left + float(kernel['Duration(us)'])))
    dma = [interval(event) for event in spans if event['pid'] == hardware
           and event.get('args', {}).get('Task Type') == 'PCIE_DMA_SQE']
    rows = []
    for step in (event for event in spans if event['name'].startswith('ProfilerStep#')):
        limits = [interval(step)]
        grouped = defaultdict(list)
        for api, region, name, span in associated:
            clipped = intersect([span], limits)
            if clipped:
                grouped[(api, region)].extend(clipped)
        groups = []
        for (api, region), times in sorted(grouped.items()):
            seconds = duration(times)
            groups.append(dict(api=api, region=region, calls=len(times), union_seconds=seconds,
                               outside_compute_seconds=seconds - duration(intersect(times, compute)),
                               outside_compute_hccl_dma_spans_seconds=seconds - duration(intersect(times, compute + hccl + dma))))
        rows.append(dict(step=step['name'], groups=groups))
    reports.append(dict(profile=folder.parent.name, steps=rows))
    del data, events, spans, kernels, scopes, indices, associated
report = dict(caveat='Instrumented temporal associations only. Region names describe this fine33 policy, '
              'not other policies. The smallest containing same-thread FSDP/HyperOffload scope is selected. '
              'Intervals are unioned within groups. Groups may overlap across threads, and HCCL spans may '
              'contain waits. Neither API time nor outside-hardware-span time proves removable critical-path cost.',
              ranks=reports)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(dict(output=str(args.output), ranks=len(reports))))
