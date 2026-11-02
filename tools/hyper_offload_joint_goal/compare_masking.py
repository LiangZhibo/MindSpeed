"""Compare planned overlap on one immutable trace; this is not a benchmark."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys
import types

root = Path(__file__).resolve().parents[2]
for package in ('mindspeed', 'mindspeed.core', 'mindspeed.core.memory', 'mindspeed.core.memory.hyper_offload'):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split('.')))]
    sys.modules[package] = module
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, AccessKind, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType as Action
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.planner import PeakResidencyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('trace', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
value = json.loads(args.trace.read_text())
value.pop('diagnostic')
value['storage_sizes'] = {int(k): v for k, v in value['storage_sizes'].items()}
value['retained_sids'] = set(value['retained_sids'])
value['ops'] = [TraceOp(op['name'], op['duration_ms'],
                       [StorageAccess(a['op_id'], a['storage_id'], AccessKind[a['kind']]) for a in op['accesses']])
                for op in value['ops']]
trace = ActivationTrace(**value)
uses = defaultdict(list)
prefix = [0.0]
for i, op in enumerate(trace.ops):
    prefix.append(prefix[-1] + op.duration_ms)
    for sid in {a.storage_id for a in op.accesses}:
        uses[sid].append(i)
rows = []
for name, scale, limit in [('legacy', None, True), ('timed_both', 1, True), ('timed_release', 1, False),
                           ('timed_both4', 4, True), ('timed_both8', 8, True), ('timed_both16', 16, True)]:
    schedule = PeakResidencyPlanner(enable_communication_masking=True, transfer_window_scale=scale,
                                    limit_prefetch_window=limit).build(trace)
    windows, calls, traffic = [], 0, 0
    weighted_ms = 0
    for pos, actions in sorted(schedule.pre.items()):
        for action in actions:
            if action.kind != Action.COPY_H2D:
                continue
            future = uses[action.storage_id]
            consumer = future[bisect_left(future, pos)]
            window = prefix[consumer] - prefix[pos]
            size = trace.storage_sizes[action.storage_id]
            windows.append(window)
            traffic += size
            calls += 1
            weighted_ms += window * size
    rows.append(dict(mode=name, calls=calls, traffic_gib=traffic / 2**30,
                     median_planned_h2d_window_ms=median(windows),
                     byte_weighted_h2d_window_ms=weighted_ms/traffic,
                     zero_window_count=sum(w == 0 for w in windows)))
result = dict(trace=str(args.trace), results=rows,
              caveat='Lookahead from captured warmup compute durations; not actual DMA overlap or steady performance.')
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
