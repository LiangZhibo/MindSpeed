"""Install a checked harness once and start one private guarded trial."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

tag, trial = sys.argv[1:3]
assert tag.isalnum() and re.fullmatch(r'[a-zA-Z0-9_]+', trial)
root = Path('/data/disk/liangzhibo/HyperOffload-research/joint-tuning-20261101') / tag
private = '/tmp/liangzhibo-joint-tuning-20261101/' + tag
container = 'mindspeed-qwen25-32b-fsdp'
py = '/usr/local/python3.12.13/bin/python'
assert json.loads((root / 'cpu-tests-status.json').read_text())['exit_code'] == 0
assert not (root / trial).exists() and not (root / (trial + '-launch.json')).exists()
if trial not in ('small_npu', 'small_npu_rng'):
    config = json.loads((root / 'configs' / (trial + '.json')).read_text())
    assert config['trial'] == trial
    assert config['source'] == private + '/source', 'config must execute the verified source revision'
    assert config['output'] == private + '/results/' + trial, 'config must use this trial output'
manifest = json.loads((root / 'harness-manifest.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    code = 'from pathlib import Path; import hashlib; p=Path(%r); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING")' % (private + '/' + name)
    existing = subprocess.check_output(['docker', 'exec', container, py, '-c', code], text=True).strip()
    if existing == 'MISSING':
        subprocess.run(['docker', 'cp', str(root / name), container + ':' + private + '/' + name], check=True)
    else:
        assert existing == digest, 'refuse to mutate an installed harness'
if trial not in ('small_npu', 'small_npu_rng'):
    if tag == 'rawrepeat':
        # The untouched original source was already qualified by the full
        # 24-step Qwen baseline, before these independent repeated runs.
        source = json.loads((root / 'source-manifest.json').read_text())
        assert source['archive_sha256'] == 'a836698d8b5e5a32155f1d21f51fd698e945082e72f41d40b534a7ab8d4b99eb'
        baseline = Path('/data/disk/liangzhibo/HyperOffload-research/joint-baseline-20261101/baseline_r33_24')
        status = json.loads((baseline / 'status.json').read_text())
        assert status['status'] == 'complete' and status['exit_code'] == 0 and not status.get('interference')
        assert (baseline / 'source-before.json').read_text() == (baseline / 'source-after.json').read_text()
        for rank in (0, 1):
            result = json.loads((baseline / f'result_rank{rank}.json').read_text())
            audit = json.loads((baseline / f'full_audit_rank{rank}.json').read_text())
            assert result['steps'] == 24 and result['status'] == 'steps_complete'
            assert audit['status'] == 'complete' and len(audit['tensors']) == 579
    else:
        assert all(json.loads((root / 'small_npu' / f'qualification_rank{i}.json').read_text())['status'] == 'passed' for i in (0, 1))
    subprocess.run(['docker', 'exec', container, 'mkdir', '-p', private + '/configs'], check=True)
    subprocess.run(['docker', 'cp', str(root / 'configs' / (trial + '.json')),
                    container + ':' + private + '/configs/' + trial + '.json'], check=True)
with (root / (trial + '-controller.log')).open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', '/tmp/liangzhibo-joint-run-guarded.py', tag, trial],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(pid=process.pid, start_ticks=(Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
              started=time.time(), tag=tag, trial=trial)
(root / (trial + '-launch.json')).write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
