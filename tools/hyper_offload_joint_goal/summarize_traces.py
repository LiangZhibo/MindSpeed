"""Summarize warmup traces and planned transfer volumes; not steady timing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[2]
for package in ('mindspeed', 'mindspeed.core', 'mindspeed.core.memory', 'mindspeed.core.memory.hyper_offload'):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split('.')))]
    sys.modules[package] = module
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, AccessKind, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.planner import PeakResidencyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('traces', nargs='+', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
rows = []
for path in args.traces:
    data = json.loads(path.read_text())
    diagnostic = data.pop('diagnostic')
    data['storage_sizes'] = {int(k): v for k, v in data['storage_sizes'].items()}
    data['retained_sids'] = set(data['retained_sids'])
    data['ops'] = [TraceOp(op['name'], op['duration_ms'],
                           [StorageAccess(a['op_id'], a['storage_id'], AccessKind[a['kind']]) for a in op['accesses']])
                   for op in data['ops']]
    trace = ActivationTrace(**data)
    schedule = PeakResidencyPlanner(enable_communication_masking=diagnostic['masking'],
                                     transfer_window_scale=diagnostic['window_scale'],
                                     limit_prefetch_window=diagnostic.get('limit_prefetch_window', True)).build(trace)
    sizes, calls = Counter(), Counter()
    compute = Counter()
    for op in trace.ops:
        compute[op.name] += op.duration_ms
    for op, actions in schedule.post.items():
        for action in actions:
            if action.kind == ResidencyActionType.COPY_D2H:
                sizes[trace.ops[op].name] += trace.storage_sizes[action.storage_id] / 2**30
                calls[trace.ops[op].name] += 1
    row = dict(trace=str(path), ops=len(trace.ops), storages=len(trace.storage_sizes),
               diagnostic=diagnostic, d2h_gib=sum(sizes.values()), d2h_calls=sum(calls.values()),
               profiled_op_seconds=sum(compute.values())/1000,
               transfer_bandwidth_gibps=[trace.d2h_bandwidth_gbps, trace.h2d_bandwidth_gbps],
               copy_sources=[dict(op=k, gib=v, calls=calls[k]) for k, v in sizes.most_common()],
               profiled_ops=[dict(op=k, ms=v) for k, v in compute.most_common()],
               caveat='Warmup timers and planned traffic explain candidates; they are not stable-step profiling or measured transfer overlap.')
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ('copy_sources', 'profiled_ops', 'caveat')}))
args.output.write_text(json.dumps(rows, indent=2))
