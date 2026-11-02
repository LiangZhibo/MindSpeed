"""Run one frozen AB/BA/AB experiment queue on the authorized host.

Each child retains the existing NPU lease, foreign-process checks, ownership
guard and deadline. This controller stops on failure and never substitutes a
run or changes a configuration in response to measured performance.
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
assert len(plan['runs']) == 6 and [x['role'] for x in plan['runs']] == ['A', 'B', 'B', 'A', 'A', 'B']
lock = (root / 'controller.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (root / 'status.json').exists(), 'never restart an existing queue implicitly'
state = dict(status='starting', pid=os.getpid(), started=time.time(), completed=[],
             plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())


def save(**updates):
    state.update(updates)
    temp = root / 'status.tmp'
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(root / 'status.json')


try:
    save()
    for item in plan['runs']:
        if time.time() >= 1793575800:
            raise TimeoutError('goal deadline; no further runs')
        tag, trial = item['tag'], item['trial']
        config = base / tag / 'configs' / (trial + '.json')
        assert hashlib.sha256(config.read_bytes()).hexdigest() == item['config_sha256']
        source = json.loads((base / tag / 'source-manifest.json').read_text())
        assert source['archive_sha256'] == item['source_sha256']
        raw = subprocess.check_output([sys.executable, '/tmp/liangzhibo-joint-start-trial.py', tag, trial], text=True)
        launch = json.loads(raw)
        save(status='running', current=item, launch=launch)
        path = base / tag / trial / 'status.json'
        while True:
            if path.exists():
                progress = json.loads(path.read_text())
                if progress['status'] in ('complete', 'failed'):
                    break
            if time.time() >= 1793575800 + 90:
                raise TimeoutError('child deadline guard did not finish; inspect exact owned processes')
            if not (Path('/proc') / str(launch['pid'])).exists():
                raise RuntimeError('controller exited without a final status')
            time.sleep(10)
        assert progress['status'] == 'complete' and progress['exit_code'] == 0, progress
        folder = path.parent
        assert (folder / 'source-before.json').read_text() == (folder / 'source-after.json').read_text()
        for rank in (0, 1):
            result = json.loads((folder / f'result_rank{rank}.json').read_text())
            audit = json.loads((folder / f'full_audit_rank{rank}.json').read_text())
            assert result['status'] == 'steps_complete' and result['steps'] == 36
            assert audit['status'] == 'complete' and len(audit['tensors']) == 579
        state['completed'].append(dict(item=item, status=progress))
        save()
    save(status='complete', current=None, finished=time.time())
except BaseException as error:
    save(status='failed', error=repr(error), finished=time.time())
    raise
