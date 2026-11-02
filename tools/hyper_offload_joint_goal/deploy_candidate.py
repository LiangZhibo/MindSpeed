"""Deploy one approved immutable candidate revision and run CPU regressions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

tag = sys.argv[1]
assert tag.isalnum()
container = 'mindspeed-qwen25-32b-fsdp'
root = Path('/data/disk/liangzhibo/HyperOffload-research/joint-tuning-20261101') / tag
private = '/tmp/liangzhibo-joint-tuning-20261101/' + tag
kit = Path('/tmp/liangzhibo-joint-' + tag + '.tar.gz')
manifest_path = Path('/tmp/liangzhibo-joint-' + tag + '-manifest.json')
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(kit.read_bytes()).hexdigest() == manifest['archive_sha256']
assert not root.exists(), 'revision already deployed; inspect it instead'
root.mkdir(parents=True)
(root / 'source-manifest.json').write_bytes(manifest_path.read_bytes())
subprocess.run(['docker', 'exec', container, 'mkdir', '-p', private], check=True)
subprocess.run(['docker', 'cp', str(kit), container + ':' + private + '/source.tar.gz'], check=True)
subprocess.run(['docker', 'cp', str(manifest_path), container + ':' + private + '/source-manifest.json'], check=True)
extract = '''import hashlib,json,tarfile
from pathlib import Path
r=Path(%r); m=json.loads((r/'source-manifest.json').read_text())
assert hashlib.sha256((r/'source.tar.gz').read_bytes()).hexdigest()==m['archive_sha256']
dest=r/'source'; dest.mkdir()
with tarfile.open(r/'source.tar.gz') as tf: tf.extractall(dest,filter='data')
for name,digest in m['files'].items(): assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest,name
print('VERIFIED',len(m['files']))
''' % private
subprocess.run(['docker', 'exec', container, '/usr/local/python3.12.13/bin/python', '-c', extract], check=True)
# Wait for the actual previously started baseline process to exit, rather than
# inferring completion from a stale state file or an observation timeout.
while Path('/proc/3151508/stat').exists():
    parts = Path('/proc/3151508/stat').read_text().rsplit(')', 1)[1].split()
    if parts[19] != '36670252' or parts[0] == 'Z':
        break
    if time.time() >= 1793575800:
        raise TimeoutError('goal deadline')
    time.sleep(5)
command = ['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1', '-e', 'OMP_NUM_THREADS=1',
           '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1',
           '-w', private + '/source', container, 'taskset', '-c', '72-75',
           '/usr/local/python3.12.13/bin/python', '-m', 'pytest',
           'tests_extend/unit_tests/mindspeed/core/memory/hyper_offload', '-q', '--tb=short']
with (root / 'cpu-tests.log').open('x') as log:
    test = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    record = dict(pid=test.pid, started=time.time(), command=command)
    (root / 'cpu-tests-launch.json').write_text(json.dumps(record, indent=2))
    try:
        code = test.wait(timeout=900)
    except BaseException:
        test.terminate()
        raise
record.update(exit_code=code, finished=time.time())
(root / 'cpu-tests-status.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
raise SystemExit(code)
