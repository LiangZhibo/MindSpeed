"""Collect a frozen, matched diagnostic pair after numerical qualification.

Profiler results are excluded from all performance claims.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path('/data/disk/liangzhibo/HyperOffload-research/joint-tuning-20261101')
plan_path = Path(sys.argv[1]).resolve()
assert plan_path.parent.parent == base and plan_path.name == 'plan.json'
root = plan_path.parent
plan = json.loads(plan_path.read_text())
assert plan['kind'] == 'matched_diagnostic_profiles' and len(plan['runs']) == 2
assert [item['role'] for item in plan['runs']] == ['raw', 'auto']
assert len({item['tag'] for item in plan['runs']}) == 1
assert plan['predecessor'].isalnum()
lock = (root / 'controller.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (root / 'status.json').exists(), 'do not restart an existing queue'
state = dict(status='waiting_for_predecessor', pid=os.getpid(), started=time.time(), completed=[],
             start_ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
             plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())


def save(**updates):
    state.update(updates)
    temporary = root / 'status.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(root / 'status.json')


def check_deadline():
    if time.time() >= 1793575800:
        raise TimeoutError('goal deadline; no new trial')


try:
    save()
    predecessor = base / plan['predecessor']
    assert hashlib.sha256((predecessor / 'plan.json').read_bytes()).hexdigest() == plan['predecessor_plan_sha256']
    while True:
        check_deadline()
        previous = json.loads((predecessor / 'status.json').read_text())
        if previous['status'] in ('complete', 'failed'):
            assert previous['status'] == 'complete' and len(previous['completed']) == plan['predecessor_count'], previous
            break
        identity = Path('/proc') / str(previous['pid']) / 'stat'
        assert identity.exists() and identity.read_text().rsplit(')', 1)[1].split()[19] == str(previous['start_ticks'])
        time.sleep(10)
    for item in plan['runs']:
        check_deadline()
        tag, trial = item['tag'], item['trial']
        assert tag.isalnum() and trial.replace('_', '').isalnum()
        source = json.loads((base / tag / 'source-manifest.json').read_text())
        assert source['archive_sha256'] == item['source_sha256']
        config = base / tag / 'configs' / (trial + '.json')
        assert hashlib.sha256(config.read_bytes()).hexdigest() == item['config_sha256']
        settings = json.loads(config.read_text())
        assert settings['profile_stable'] and settings['profile_runtime_tags']
        assert settings['steps'] == 16 and settings['profile_skip_first'] == 12
        assert settings['autotune'] == (item['role'] == 'auto')
        raw = subprocess.check_output([sys.executable, '/tmp/liangzhibo-joint-start-trial.py', tag, trial], text=True)
        launch = json.loads(raw)
        save(status='running', current=item, launch=launch)
        folder = base / tag / trial
        while True:
            path = folder / 'status.json'
            if path.exists():
                progress = json.loads(path.read_text())
                if progress['status'] in ('complete', 'failed'):
                    break
            if time.time() >= 1793575800 + 90:
                raise TimeoutError('trial deadline guard did not finish; inspect owned processes')
            identity = Path('/proc') / str(launch['pid']) / 'stat'
            assert identity.exists(), 'trial controller exited without terminal status'
            assert identity.read_text().rsplit(')', 1)[1].split()[19] == str(launch['start_ticks'])
            time.sleep(10)
        assert progress['status'] == 'complete' and progress['exit_code'] == 0, progress
        assert not progress.get('interference')
        assert (folder / 'source-before.json').read_text() == (folder / 'source-after.json').read_text()
        reference = base / plan['runs'][0]['tag'] / plan['runs'][0]['trial']
        audit_path = root / (trial + '-audit.json')
        subprocess.run([sys.executable, '/tmp/liangzhibo-joint-audit-profile.py', str(reference), str(folder),
                        '--output', str(audit_path)], check=True)
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'diagnostic_numeric_audit_passed' and audit['performance_usable'] is False
        state['completed'].append(dict(item=item, status=progress))
        save()
    save(status='complete', current=None, finished=time.time())
except BaseException as error:
    save(status='failed', error=repr(error), finished=time.time())
    raise
