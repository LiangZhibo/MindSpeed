"""Audit a completed candidate against a matching baseline, without filtering steps."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--first-step', type=int, default=13)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--memory-mode', choices=('baseline-peak', 'fixed-capacity'), default='baseline-peak')
args = parser.parse_args()


def load_run(folder):
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'complete' and status['exit_code'] == 0
    assert not status.get('interference', False)
    assert status['cards'] == [6, 7]
    before = json.loads((folder / 'source-before.json').read_text())
    after = json.loads((folder / 'source-after.json').read_text())
    assert before == after and before['status'] == 'verified'
    results = [json.loads((folder / f'result_rank{rank}.json').read_text()) for rank in (0, 1)]
    audits = [json.loads((folder / f'full_audit_rank{rank}.json').read_text()) for rank in (0, 1)]
    launches = [json.loads((folder / f'launch_rank{rank}.json').read_text()) for rank in (0, 1)]
    expected_source = None
    if 'archive_sha256' in before:
        manifest = json.loads((folder.parent / 'source-manifest.json').read_text())
        assert manifest['archive_sha256'] == before['archive_sha256']
        expected_source = '/tmp/liangzhibo-joint-tuning-20261101/' + manifest['tag'] + '/source'
    for launch in launches:
        assert launch['source'] == launch['config']['source']
        assert launch['argv'][0] == launch['source'] + (
            '/examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_qwen25_native_mindspeed_fsdp2.py')
        if expected_source is not None:
            assert launch['source'] == expected_source, 'entry source differs from the verified archive'
        assert launch['performance_usable'] and not launch['config'].get('profile_stable', False)
        assert launch['config']['mode'] == 'performance'
    for result, audit in zip(results, audits):
        assert result['status'] == 'steps_complete' and audit['status'] == 'complete'
        assert len(audit['tensors']) == 579 and audit['deterministic']
        assert [step['index'] for step in result['extra_step_results']] == list(range(2, result['steps'] + 1))
        assert all(math.isfinite(step['elapsed_seconds']) and step['elapsed_seconds'] > 0
                   for step in result['extra_step_results'])
    return results, audits, launches


baseline, baseline_audits, baseline_launches = load_run(args.baseline)
candidate, candidate_audits, candidate_launches = load_run(args.candidate)
memory = []
for rank in (0, 1):
    a, b = baseline[rank], candidate[rank]
    for key in ('steps', 'layers', 'seq_length', 'parameter_count', 'optimizer', 'learning_rate',
                'world_size', 'fsdp_size', 'cpu_offload', 'activation_swap',
                'offload_device_mb', 'offload_host_mb'):
        assert a[key] == b[key], (rank, key)
    for key in ('model', 'steps', 'allocator_fraction', 'deterministic'):
        assert baseline_launches[rank]['config'][key] == candidate_launches[rank]['config'][key], (rank, key)
    for key in ('torch', 'torch_npu', 'deterministic'):
        assert baseline_launches[rank][key] == candidate_launches[rank][key], (rank, key)
    assert baseline_launches[rank]['cpu_affinity'] == candidate_launches[rank]['cpu_affinity']
    for side in ('parameter', 'gradient'):
        left = {row['name']: row[side] for row in baseline_audits[rank]['tensors']}
        right = {row['name']: row[side] for row in candidate_audits[rank]['tensors']}
        assert left == right, f'full {side} hashes differ on rank{rank}'
    for key in ('loss', 'probe_grad_max', 'parameter_update_max'):
        assert a[key] == b[key], (rank, 'first step', key)
    for left, right in zip(a['extra_step_results'], b['extra_step_results']):
        for key in ('index', 'loss', 'probe_grad_max', 'parameter_update_max'):
            assert left[key] == right[key], (rank, left['index'], key)
    tuning = b['joint_tuning']
    assert tuning['phase'] == 'stable' and tuning['frozen_at_step'] <= 12
    assert tuning['frozen_at_step'] < args.first_step
    stable_rows = [row for row in b['extra_step_results'] if row['index'] >= args.first_step]
    assert all(row['tuning_phase'] == 'stable' and row['joint_policy'] == tuning['policy'] for row in stable_rows)
    assert candidate_audits[rank]['schedule']
    assert any(action['kind'] == 'COPY_D2H' for action in candidate_audits[rank]['schedule'])
    assert b['d2h_calls'] > 0 and b['d2h_gib'] > 0
    # Whole-step allocation protects against hiding opaque-region saves from
    # the storage planner's budget. Only rounding tolerance is allowed here.
    a_peak = max([a['first_step_npu_peak_allocated_gib']] +
                 [row['npu_peak_allocated_gib'] for row in a['extra_step_results']])
    b_peak = max([b['first_step_npu_peak_allocated_gib']] +
                 [row['npu_peak_allocated_gib'] for row in b['extra_step_results']])
    peak_matched = b_peak <= a_peak + 1 / 1024
    argv = candidate_launches[rank]['argv']
    cap_mb = (int(argv[argv.index('--offload-tuning-device-peak-mb') + 1])
              if '--offload-tuning-device-peak-mb' in argv else None)
    capacity_matched = peak_matched or cap_mb is not None and b_peak <= cap_mb / 1024
    memory.append(dict(rank=rank, baseline_peak_gib=a_peak, candidate_peak_gib=b_peak,
                       growth_gib=b_peak-a_peak, peak_matched=peak_matched,
                       allocated_cap_mb=cap_mb, capacity_matched=capacity_matched))
    assert b['host_pool_allocated_gib'] <= b['offload_host_mb'] / 1024
    assert b['host_pool_pageable_gib'] == 0
assert candidate[0]['joint_tuning']['policy'] == candidate[1]['joint_tuning']['policy']


def stable_times(run):
    left, right = [r['extra_step_results'] for r in run]
    return [max(a['elapsed_seconds'], b['elapsed_seconds']) for a, b in zip(left, right)
            if a['index'] >= args.first_step]


a, b = stable_times(baseline), stable_times(candidate)
assert len(a) == len(b) and len(a) > 0
eligible = all(row['peak_matched' if args.memory_mode == 'baseline-peak' else 'capacity_matched'] for row in memory)
summary = dict(status='screen_audit_passed' if eligible else 'screen_memory_ineligible',
               memory_mode=args.memory_mode, memory=memory, eligible=eligible,
               first_step=args.first_step, samples=len(a),
               baseline_seconds=median(a), candidate_seconds=median(b),
               reduction_percent=100 * (1 - median(b) / median(a)),
               target_seconds=.9 * median(a), target_point_estimate=median(b) <= .9 * median(a),
               qualification='all final579 local parameter and gradient hashes per rank exact; all step probes exact',
               tuning=[r['joint_tuning'] for r in candidate], baseline_times=a, candidate_times=b,
               warning='one screening run is not independent repeated performance confirmation')
args.output.write_text(json.dumps(summary, indent=2))
print(json.dumps({key: value for key, value in summary.items() if key not in ('tuning', 'baseline_times', 'candidate_times')}, indent=2))
