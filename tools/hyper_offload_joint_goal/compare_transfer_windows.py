"""Compare final plans against matching fine-region warmup traces.

Trace timing is an input to planning, not a measurement of stable overlap.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()


def distribution(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None


reports = []
plans = []
for folder in args.runs:
    result = json.loads((folder / 'result_rank0.json').read_text())
    audit = json.loads((folder / 'full_audit_rank0.json').read_text())
    trace = json.loads((folder / 'trace2_rank0.json').read_text())
    assert result['status'] == 'steps_complete' and audit['status'] == 'complete'
    policy = result['joint_tuning']['policy']
    assert len(policy['resident_regions']) == 15
    assert len(policy['recompute_regions']) == 63
    assert len(trace['ops']) == len(audit['guide'])
    names = [op['name'] for op in trace['ops']]
    structural = dict(names=names, accesses=[op['accesses'] for op in trace['ops']],
                      sizes=trace['storage_sizes'])
    structure_hash = hashlib.sha256(json.dumps(structural, sort_keys=True).encode()).hexdigest()
    plan = audit['schedule']
    plans.append(plan)
    elapsed = [0.0]
    reads = defaultdict(list)
    for index, op in enumerate(trace['ops']):
        elapsed.append(elapsed[-1] + op['duration_ms'])
        for access in op['accesses']:
            if access['kind'] in ('READ', 'READ_WRITE'):
                reads[access['storage_id']].append(index)
    by_storage = defaultdict(list)
    for action in plan:
        by_storage[action['storage_id']].append(action)
    transfers = []
    for sid, actions in by_storage.items():
        copies = sorted(a['op'] for a in actions if a['kind'] == 'COPY_D2H')
        prefetches = sorted(a['op'] for a in actions if a['kind'] == 'COPY_H2D')
        releases = sorted(a['op'] for a in actions if a['kind'] == 'RELEASE_DEVICE')
        assert len(copies) == len(prefetches) <= 1, 'this analysis expects at most one transfer gap per storage'
        if not copies:
            continue
        copy, prefetch = copies[0], prefetches[0]
        release = min(op for op in releases if op >= copy)
        # D2H may start while read-only forward users still consume the device
        # copy. The prefetch deadline is the next read after device release.
        demand = min(op for op in reads[sid] if op > release)
        assert copy <= release < prefetch <= demand
        transfers.append(dict(storage_id=sid, bytes=trace['storage_sizes'][str(sid)],
                              d2h=copy, release=release, h2d=prefetch, demand=demand,
                              release_delay_ops=release-copy, prefetch_lead_ops=demand-prefetch,
                              warmup_release_delay_ms=elapsed[release+1]-elapsed[copy+1],
                              warmup_prefetch_lead_ms=elapsed[demand]-elapsed[prefetch]))
    reports.append(dict(run=str(folder), structure_sha256=structure_hash,
                        plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                        policy=policy, action_counts=dict(Counter(a['kind'] for a in plan)),
                        copied_gib=sum(row['bytes'] for row in transfers) / 2**30,
                        distributions={key: distribution([row[key] for row in transfers]) for key in
                                       ('release_delay_ops', 'prefetch_lead_ops',
                                        'warmup_release_delay_ms', 'warmup_prefetch_lead_ms')},
                        transfers=transfers))
comparisons = []
for i, left in enumerate(reports):
    for j in range(i+1, len(reports)):
        right = reports[j]
        comparable = left['structure_sha256'] == right['structure_sha256']
        changes = None
        if comparable:
            a = {r['storage_id']: r for r in left['transfers']}
            b = {r['storage_id']: r for r in right['transfers']}
            changes = dict(shared_storages=len(a.keys() & b.keys()),
                           different_offload_set=set(a) != set(b),
                           **{key: sum(a[s][key] != b[s][key] for s in a.keys() & b.keys())
                              for key in ('d2h', 'release', 'h2d', 'demand')})
        comparisons.append(dict(left=left['run'], right=right['run'], same_structure=comparable,
                                identical_action_sequence=plans[i] == plans[j], changed_positions=changes))
report = dict(caveat='Final action locations are exact. Millisecond lead/delay values use instrumented warmup '
              'op durations and do not establish stable DMA overlap or critical-path savings. '
              'Independent runs have different measured trace timings; they are not a single-trace ablation.',
              runs=reports, comparisons=comparisons)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(dict(runs=[{k: v for k, v in row.items() if k not in ('policy', 'transfers')}
                            for row in reports], comparisons=comparisons), indent=2))
