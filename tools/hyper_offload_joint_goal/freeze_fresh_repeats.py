"""Freeze a new comparison using fresh baseline and candidate runs only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('candidate_tag')
parser.add_argument('screen_trial')
parser.add_argument('queue_name')
parser.add_argument('--predecessor', required=True)
parser.add_argument('--predecessor-count', type=int, required=True)
args = parser.parse_args()
assert all(value.isalnum() for value in (args.candidate_tag, args.queue_name, args.predecessor))
assert args.screen_trial.replace('_', '').isalnum() and args.predecessor_count > 0
root = Path(__file__).resolve().parent
candidate = root / args.candidate_tag
screen = json.loads((candidate / 'configs' / (args.screen_trial + '.json')).read_text())
assert screen['autotune'] and screen['tuning_steps'] <= 12
assert screen['source'] == f'/tmp/liangzhibo-joint-tuning-20261101/{args.candidate_tag}/source'
assert screen['mode'] == 'performance' and not screen.get('profile_stable')
screen_folder = candidate / (args.screen_trial + '_result')
source = json.loads((candidate / 'source-manifest.json').read_text())
assert json.loads((screen_folder / 'source-before.json').read_text())['archive_sha256'] == source['archive_sha256']
for rank in (0, 1):
    assert json.loads((screen_folder / f'launch_rank{rank}.json').read_text())['config'] == screen
baseline_root = root / 'rawrepeat'
baseline_source = json.loads((baseline_root / 'source-manifest.json').read_text())
assert baseline_source['archive_sha256'] == 'a836698d8b5e5a32155f1d21f51fd698e945082e72f41d40b534a7ab8d4b99eb'
template = json.loads((baseline_root / 'configs/baseline_repeat1_36.json').read_text())
assert template['recompute_layers'] == 33 and not template['autotune']
predecessor_plan = root / args.predecessor / 'plan.json'
assert len(json.loads(predecessor_plan.read_text())['runs']) == args.predecessor_count
out = root / args.queue_name
assert not out.exists()
configs, runs = [], []
for role, pair in (('A', 1), ('B', 1), ('B', 2), ('A', 2), ('A', 3), ('B', 3)):
    tag = 'rawrepeat' if role == 'A' else args.candidate_tag
    trial = args.queue_name + ('_baseline' if role == 'A' else '_candidate') + f'_repeat{pair}_36'
    path = root / tag / 'configs' / (trial + '.json')
    assert not path.exists() and not (root / tag / (trial + '_result')).exists()
    config = dict(template if role == 'A' else screen, trial=trial, steps=36, capture_trace=False,
                  output=f'/tmp/liangzhibo-joint-tuning-20261101/{tag}/results/{trial}')
    assert config['allocator_fraction'] == .95 and config['device_mb'] == 2048
    assert config['source'] == f'/tmp/liangzhibo-joint-tuning-20261101/{tag}/source'
    assert bool(config['autotune']) == (role == 'B')
    encoded = json.dumps(config, indent=2)
    configs.append((path, encoded))
    runs.append(dict(tag=tag, trial=trial, role=role, pair=pair,
                     config_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                     source_sha256=(baseline_source if role == 'A' else source)['archive_sha256']))
out.mkdir()
audit_path = out / 'screen-audit.json'
subprocess.run([sys.executable, str(root / 'analyze_run.py'), str(root / 'baseline_result'),
                str(screen_folder), '--first-step', '13', '--memory-mode', 'fixed-capacity',
                '--output', str(audit_path)], check=True, stdout=subprocess.DEVNULL)
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'screen_audit_passed' and audit['eligible']
for path, encoded in configs:
    path.write_text(encoded)
plan = dict(frozen_at_epoch=time.time(), candidate=args.candidate_tag, screening_trial=args.screen_trial,
            screen_summary_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((root / 'REPEAT_PROTOCOL.json').read_bytes()).hexdigest(),
            predecessor=args.predecessor, predecessor_count=args.predecessor_count,
            predecessor_plan_sha256=hashlib.sha256(predecessor_plan.read_bytes()).hexdigest(),
            primary_metric='Geometric mean of three paired ratios of max-rank stable13..36 medians',
            baseline_reuse=False, runs=runs)
(out / 'plan.json').write_text(json.dumps(plan, indent=2))
print(json.dumps(dict(plan=str(out / 'plan.json'), runs=len(runs), baseline_reuse=False)))
