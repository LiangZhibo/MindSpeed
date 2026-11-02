"""Separate hardware compute, HCCL and PCIe DMA intervals in an Ascend profile.

Times are instrumented diagnostics, never standalone performance samples.
Interval overlap is not a critical-path attribution or a causal speedup bound.
"""
import argparse
import collections
import csv
import json
from pathlib import Path


def union(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def duration(intervals):
    return sum(b - a for a, b in union(intervals)) / 1e6


def intersect(a, b):
    a, b = union(a), union(b)
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        left, right = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if left < right:
            out.append((left, right))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


parser = argparse.ArgumentParser()
parser.add_argument('profile', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
results = []
for folder in sorted(args.profile.glob('*/ASCEND_PROFILER_OUTPUT')):
    data = json.loads((folder / 'trace_view.json').read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    processes = {e['args']['name']: e['pid'] for e in events
                 if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    hardware = processes['Ascend Hardware']
    dma = [(float(e['ts']), float(e['ts']) + e['dur']) for e in events
           if e.get('ph') == 'X' and e['pid'] == hardware
           and e.get('args', {}).get('Task Type') == 'PCIE_DMA_SQE']
    steps = [e for e in events if e.get('ph') == 'X' and e['name'].startswith('ProfilerStep#')]
    kernels = list(csv.DictReader((folder / 'kernel_details.csv').open()))
    compute = [(float(k['Start Time(us)']), float(k['Start Time(us)']) + float(k['Duration(us)']))
               for k in kernels if k['Accelerator Core'] != 'COMMUNICATION']
    comm = [(float(k['Start Time(us)']), float(k['Start Time(us)']) + float(k['Duration(us)']))
            for k in kernels if k['Accelerator Core'] == 'COMMUNICATION']
    rows = []
    for step in steps:
        span = [(float(step['ts']), float(step['ts']) + step['dur'])]
        c, h, d = [intersect(v, span) for v in (compute, comm, dma)]
        names, counts = collections.Counter(), collections.Counter()
        for e in events:
            if e.get('cat') == 'cpu_op' and e.get('ph') == 'X' and not e['name'].startswith('ProfilerStep#'):
                start = float(e['ts'])
                if span[0][0] <= start < span[0][1]:
                    names[e['name']] += e['dur'] / 1e6
                    counts[e['name']] += 1
        rows.append(dict(step=step['name'], span_seconds=duration(span), compute_seconds=duration(c),
                         hccl_seconds=duration(h), pcie_dma_seconds=duration(d),
                         hccl_without_compute_seconds=duration(h) - duration(intersect(h, c)),
                         dma_without_compute_seconds=duration(d) - duration(intersect(d, c)),
                         dma_without_compute_or_hccl_seconds=duration(d) - duration(intersect(d, union(c + h))),
                         no_compute_hccl_or_dma_seconds=duration(span) - duration(union(c + h + d)),
                         cpu_inclusive_top=[dict(name=k, seconds=v, count=counts[k]) for k, v in names.most_common(12)]))
    results.append(dict(folder=folder.parent.name, steps=rows))
    del data, events
report = dict(qualification='Instrumented diagnostic only. CPU durations are inclusive and cannot be summed. '
              'HCCL intervals are distinct from PCIe DMA; uncovered intervals do not prove the causal critical path.',
              ranks=results)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps([{k: v for k, v in row.items() if k != 'cpu_inclusive_top'}
                  for result in results for row in result['steps']], indent=2))
