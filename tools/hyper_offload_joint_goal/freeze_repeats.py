"""Freeze one selected autotuning family before independent AB/BA/AB repeats."""
import argparse
import hashlib
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('candidate_tag')
parser.add_argument('screen_trial')
parser.add_argument('queue_name')
args = parser.parse_args()
assert args.candidate_tag.isalnum() and args.queue_name.isalnum()
root = Path(__file__).resolve().parent
out = root / args.queue_name
out.mkdir(exist_ok=False)
candidate = root / args.candidate_tag
screen_config = candidate / 'configs' / (args.screen_trial + '.json')
screen = json.loads(screen_config.read_text())
assert screen['autotune'] and not screen.get('profile_stable')
assert screen['mode'] == 'performance' and screen['tuning_steps'] <= 12
summary = json.loads((candidate / 'qwen-screen-summary.json').read_text())
assert summary['status'] == 'screen_audit_passed' and summary.get('eligible', True)
sources = {tag: json.loads((root / tag / 'source-manifest.json').read_text())['archive_sha256']
           for tag in ('rawrepeat', args.candidate_tag)}
runs = []
for role, pair in (('A', 1), ('B', 1), ('B', 2), ('A', 2), ('A', 3), ('B', 3)):
    tag = 'rawrepeat' if role == 'A' else args.candidate_tag
    trial = ('baseline' if role == 'A' else 'candidate') + f'_repeat{pair}_36'
    path = root / tag / 'configs' / (trial + '.json')
    if role == 'B':
        config = dict(screen, trial=trial, steps=36, capture_trace=False,
                      output=f'/tmp/liangzhibo-joint-tuning-20261101/{tag}/results/{trial}')
        text = json.dumps(config, indent=2)
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    config = json.loads(path.read_text())
    assert config['steps'] == 36 and not config.get('profile_stable')
    assert config['allocator_fraction'] == .95 and config['device_mb'] == 2048
    assert bool(config['autotune']) == (role == 'B')
    runs.append(dict(tag=tag, trial=trial, role=role, pair=pair,
                     config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     source_sha256=sources[tag]))
plan = dict(frozen_at_epoch=time.time(), candidate=args.candidate_tag,
            screening_trial=args.screen_trial,
            screen_summary_sha256=hashlib.sha256((candidate / 'qwen-screen-summary.json').read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((root / 'REPEAT_PROTOCOL.json').read_bytes()).hexdigest(),
            primary_metric='Geometric mean of three paired ratios of max-rank stable13..36 medians',
            runs=runs)
(out / 'plan.json').write_text(json.dumps(plan, indent=2))
print(json.dumps(dict(plan=str(out / 'plan.json'), candidate=args.candidate_tag, runs=len(runs))))
