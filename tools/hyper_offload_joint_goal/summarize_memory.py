"""Report full-step allocator and sampled physical/Host-pool memory separately."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('runs', nargs='+', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
reports = []
for folder in args.runs:
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'complete' and status['exit_code'] == 0 and not status.get('interference')
    assert status['cards'] == [6, 7]
    before = json.loads((folder / 'source-before.json').read_text())
    assert before == json.loads((folder / 'source-after.json').read_text())
    assert before['status'] == 'verified'
    audits = [json.loads((folder / f'full_audit_rank{rank}.json').read_text()) for rank in (0, 1)]
    assert all(audit['status'] == 'complete' for audit in audits)
    cutoff = min(audit['started'] for audit in audits)
    samples = [json.loads(line) for line in (folder / 'device-memory.jsonl').read_text().splitlines()]
    samples = [row for row in samples if row['time'] < cutoff]
    assert all(row['code'] == 0 for row in samples)
    physical = []
    for card in (6, 7):
        rows = sorted((row for row in samples if row['card'] == card), key=lambda row: row['monotonic'])
        assert len(rows) > 1
        physical.append(dict(card=card, samples=len(rows), peak_mib=max(row['memory_usage'] for row in rows),
                             capacity_mib=min(row['memory_size'] for row in rows),
                             maximum_sample_gap_seconds=max(b['monotonic'] - a['monotonic']
                                                            for a, b in zip(rows, rows[1:]))))
    allocators = []
    for rank in (0, 1):
        result = json.loads((folder / f'result_rank{rank}.json').read_text())
        config = json.loads((folder / f'launch_rank{rank}.json').read_text())['config']
        assert result['status'] == 'steps_complete'
        allocators.append(dict(rank=rank, allocator_fraction=config['allocator_fraction'],
                               managed_budget_mib=result['offload_device_mb'],
                               host_cap_mib=result['offload_host_mb'],
                               host_pool_gib=result['host_pool_allocated_gib'],
                               pageable_host_gib=result['host_pool_pageable_gib'],
                               peak_allocated_gib=max([result['first_step_npu_peak_allocated_gib']] +
                                   [row['npu_peak_allocated_gib'] for row in result['extra_step_results']]),
                               peak_reserved_gib=max([result['first_step_npu_peak_reserved_gib']] +
                                   [row['npu_peak_reserved_gib'] for row in result['extra_step_results']])))
    reports.append(dict(run=str(folder), source_verification=before,
                        allocator_and_host_pool=allocators, sampled_training_physical=physical))
report = dict(caveat='All training steps, including tuning, are included. Physical samples stop before either '
              'rank begins final numerical audit. Nominal sampling interval50ms; brief peaks may be missed. '
              'Allocator, physical device and Host pool measurements have different scopes. This is a memory '
              'report, not independent numerical or performance qualification.', runs=reports)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(reports, indent=2))
