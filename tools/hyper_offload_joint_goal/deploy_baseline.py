"""Run on the existing test host; create only this goal's private experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

base = Path('/data/disk/liangzhibo/HyperOffload-research')
root = base / 'joint-baseline-20261101'
private = '/tmp/liangzhibo-joint-baseline-20261101'
container = 'mindspeed-qwen25-32b-fsdp'
py = '/usr/local/python3.12.13/bin/python'
kit = Path('/tmp/liangzhibo-joint-baseline-kit-20261101.tar.gz')
assert hashlib.sha256(kit.read_bytes()).hexdigest() == 'a0effe274fdb9310140634e9af89ce42a81c13cde6c19d31a1b0102aed0aabac'
assert not root.exists(), 'Inspect existing experiment instead of launching twice'
subprocess.run(['docker', 'exec', container, py, '-c',
                'from pathlib import Path; assert not Path(' + repr(private) + ').exists()'], check=True)
root.mkdir()
with tarfile.open(kit) as tf:
    tf.extractall(root)
subprocess.run(['docker', 'cp', str(root), container + ':' + private], check=True)
extract = '''import hashlib,json,tarfile
from pathlib import Path
r=Path(%r)
for arm,item in json.loads((r/'source-manifest.json').read_text()).items():
 p=r/item['archive']; assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 dest=r/'source'/arm; dest.mkdir(parents=True)
 with tarfile.open(p) as tf: tf.extractall(dest)
''' % private
subprocess.run(['docker', 'exec', container, py, '-c', extract], check=True)
subprocess.run(['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1', container, py,
                private + '/verify_sources.py'], check=True)
with (root / 'controller.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(root / 'control_pr11.py')],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               cwd=root, start_new_session=True)
record = dict(pid=process.pid, started=time.time(), root=str(root),
              start_ticks=(Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
(root / 'controller-launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
