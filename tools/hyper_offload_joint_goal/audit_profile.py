"""Qualify a diagnostic profile numerically; never produce a speedup claim."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()


def load(folder):
    status = json.loads((folder / 'status.json').read_text())
    assert status['status'] == 'complete' and status['exit_code'] == 0
    assert status['cards'] == [6, 7] and not status.get('interference')
    before = json.loads((folder / 'source-before.json').read_text())
    after = json.loads((folder / 'source-after.json').read_text())
    assert before == after and before['status'] == 'verified'
    manifest = json.loads((folder.parent / 'source-manifest.json').read_text())
    assert before['archive_sha256'] == manifest['archive_sha256']
    expected_source = '/tmp/liangzhibo-joint-tuning-20261101/' + manifest['tag'] + '/source'
    records = []
    for rank in (0, 1):
        result = json.loads((folder / f'result_rank{rank}.json').read_text())
        audit = json.loads((folder / f'full_audit_rank{rank}.json').read_text())
        launch = json.loads((folder / f'launch_rank{rank}.json').read_text())
        assert result['status'] == 'steps_complete' and result['steps'] == 16
        assert audit['status'] == 'complete' and len(audit['tensors']) == 579 and audit['deterministic']
        assert not launch['performance_usable'] and launch['config']['profile_stable']
        assert launch['source'] == launch['config']['source'] == expected_source
        assert launch['argv'][0] == expected_source + (
            '/examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_qwen25_native_mindspeed_fsdp2.py')
        assert [row['index'] for row in result['extra_step_results']] == list(range(2, 17))
        records.append((result, audit, launch))
    return records


reference, candidate = load(args.reference), load(args.candidate)
for rank, ((left, a, la), (right, b, lb)) in enumerate(zip(reference, candidate)):
    for key in ('steps', 'layers', 'seq_length', 'parameter_count', 'optimizer', 'learning_rate',
                'world_size', 'fsdp_size', 'cpu_offload', 'activation_swap',
                'offload_device_mb', 'offload_host_mb'):
        assert left[key] == right[key], (rank, key)
    for key in ('model', 'steps', 'allocator_fraction', 'deterministic'):
        assert la['config'][key] == lb['config'][key], (rank, key)
    assert la['source'] == lb['source'], 'diagnostic pair must execute the same source'
    for key in ('torch', 'torch_npu', 'deterministic', 'cpu_affinity'):
        assert la[key] == lb[key], (rank, key)
    for field in ('parameter', 'gradient'):
        assert {r['name']: r[field] for r in a['tensors']} == {r['name']: r[field] for r in b['tensors']}, (rank, field)
    for first, second in [(left, right)] + list(zip(left['extra_step_results'], right['extra_step_results'])):
        for key in ('loss', 'probe_grad_max', 'parameter_update_max'):
            assert first[key] == second[key], (rank, first.get('index', 1), key)
    if lb['config'].get('autotune'):
        tuning = right['joint_tuning']
        assert tuning['phase'] == 'stable' and tuning['frozen_at_step'] <= 12
        assert all(row['joint_policy'] == tuning['policy'] and row['tuning_phase'] == 'stable'
                   for row in right['extra_step_results'] if row['index'] >= 13)
report = dict(status='diagnostic_numeric_audit_passed', performance_usable=False,
              reference=str(args.reference), candidate=str(args.candidate),
              selected_tuning=[record[0].get('joint_tuning') for record in candidate],
              qualification='Both ranks: all16 step probes and all579 final parameter/gradient hashes exact. '
                            'Source immutable. Profile timings excluded from performance claims.')
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key != 'selected_tuning'}))
