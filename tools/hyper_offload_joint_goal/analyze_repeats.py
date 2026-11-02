"""Audit three independent, predeclared pairs; steps are not replicates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('plan', type=Path)
parser.add_argument('--root', type=Path, default=Path(__file__).parent)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
assert [item['role'] for item in plan['runs']] == ['A', 'B', 'B', 'A', 'A', 'B']
args.output.parent.mkdir(parents=True, exist_ok=True)
pairs = []
for pair in (1, 2, 3):
    runs = {item['role']: item for item in plan['runs'] if item['pair'] == pair}
    assert set(runs) == {'A', 'B'}
    paths = {role: args.root / item['tag'] / (item['trial'] + '_result') for role, item in runs.items()}
    for role, item in runs.items():
        config_path = args.root / item['tag'] / 'configs' / (item['trial'] + '.json')
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == item['config_sha256']
        expected_config = json.loads(config_path.read_text())
        proof = json.loads((paths[role] / 'source-before.json').read_text())
        assert proof['archive_sha256'] == item['source_sha256']
        for rank in (0, 1):
            launch = json.loads((paths[role] / f'launch_rank{rank}.json').read_text())
            assert launch['config'] == expected_config
    output = args.output.parent / f'pair{pair}-audit.json'
    subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_run.py')),
                    str(paths['A']), str(paths['B']), '--first-step', '13',
                    '--memory-mode', 'fixed-capacity', '--output', str(output)], check=True,
                   stdout=subprocess.DEVNULL)
    value = json.loads(output.read_text())
    assert value['status'] == 'screen_audit_passed' and value['samples'] == 24
    assert value['eligible'], 'a capacity-ineligible pair cannot support a performance claim'
    physical = {}
    for role, folder in paths.items():
        cutoff = min(json.loads((folder / f'full_audit_rank{rank}.json').read_text())['started'] for rank in (0, 1))
        rows = [json.loads(line) for line in (folder / 'device-memory.jsonl').read_text().splitlines()]
        physical[role] = {card: max(row['memory_usage'] for row in rows
                                    if row['card'] == card and row['time'] < cutoff) for card in (6, 7)}
    pairs.append(dict(pair=pair, baseline_seconds=value['baseline_seconds'],
                      candidate_seconds=value['candidate_seconds'],
                      ratio=value['candidate_seconds'] / value['baseline_seconds'],
                      memory=value['memory'], sampled_training_peak_mib=physical,
                      frozen_steps=[entry['frozen_at_step'] for entry in value['tuning']],
                      policies=[entry['policy'] for entry in value['tuning']]))
logs = [math.log(pair['ratio']) for pair in pairs]
center = statistics.mean(logs)
margin = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
ratio = math.exp(center)
interval = [math.exp(center-margin), math.exp(center+margin)]
report = dict(status='three_pairs_audited', primary='fixed original hardware/allocator/Host capacity',
              frozen_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              independent_pairs=3, stable_steps='13..36, every step included, max across ranks',
              geometric_time_reduction_percent=100*(1-ratio),
              paired_log_t95_reduction_interval_percent=[100*(1-interval[1]), 100*(1-interval[0])],
              point_target_met=ratio <= .9, interval_supports_target=interval[1] <= .9,
              all_pairs_meet_target=all(pair['ratio'] <= .9 for pair in pairs),
              baseline_peak_matched=all(row['peak_matched'] for pair in pairs for row in pair['memory']),
              numerical='All final parameter/gradient hashes and all per-step probes exact for each pair.',
              caveat='Three runs per arm give limited uncertainty resolution. Physical peak samples are every50ms and include warmup, ending before final audit.',
              pairs=pairs)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}, indent=2))
