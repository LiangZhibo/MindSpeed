"""Compare exact schedules and CPU planning time against an immutable archive."""
import argparse
import json
import logging
from pathlib import Path
import sys
import tarfile
import time
import types

root = Path(__file__).resolve().parents[2]
for package in ('mindspeed', 'mindspeed.core', 'mindspeed.core.memory', 'mindspeed.core.memory.hyper_offload'):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split('.')))]
    sys.modules[package] = module

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, AccessKind, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.planner import PeakResidencyPlanner

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('traces', type=Path, nargs='+')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
legacy = {}
with tarfile.open(args.archive) as tf:
    for name in ('modeling', 'selection'):
        path = f'mindspeed/core/memory/hyper_offload/planning/peak_strategy/{name}.py'
        module = types.ModuleType('legacy_' + name)
        exec(compile(tf.extractfile(path).read(), path, 'exec'), module.__dict__)
        legacy[name] = module


class LegacyPlanner(PeakResidencyPlanner):
    _build_profile_and_candidates = staticmethod(legacy['modeling'].build_profile_and_candidates)
    _compute_unavoidable_peak = staticmethod(legacy['selection'].compute_unavoidable_peak)

    def _select_peak_candidates(self, profile, state, limit):
        return legacy['selection'].select_peak_candidates(profile, state, limit, self._weights, self._log_selection_exhaustion)


logging.disable(logging.WARNING)
results = []
for path in args.traces:
    value = json.loads(path.read_text())
    diagnostic = value.pop('diagnostic')
    value['storage_sizes'] = {int(k): v for k, v in value['storage_sizes'].items()}
    value['retained_sids'] = set(value['retained_sids'])
    value['ops'] = [TraceOp(op['name'], op['duration_ms'],
                           [StorageAccess(a['op_id'], a['storage_id'], AccessKind[a['kind']]) for a in op['accesses']])
                    for op in value['ops']]
    trace = ActivationTrace(**value)
    schedules, timings = [], []
    for planner_type in (LegacyPlanner, PeakResidencyPlanner):
        planner = planner_type(enable_communication_masking=diagnostic['masking'],
                               transfer_window_scale=diagnostic['window_scale'],
                               limit_prefetch_window=diagnostic.get('limit_prefetch_window', True))
        start = time.perf_counter()
        schedules.append(planner.build(trace))
        timings.append(time.perf_counter() - start)
    assert schedules[0] == schedules[1], f'schedule differs: {path}'
    row = dict(trace=str(path), ops=len(trace.ops), storages=len(trace.storage_sizes),
               old_seconds=timings[0], new_seconds=timings[1], speedup=timings[0]/timings[1], exact_schedule=True)
    results.append(row)
    print(json.dumps(row), flush=True)
args.output.write_text(json.dumps(results, indent=2))
