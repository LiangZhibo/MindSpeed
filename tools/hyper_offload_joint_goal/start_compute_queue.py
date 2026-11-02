"""Start one previously frozen probe queue; never restart existing runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

base = Path('/data/disk/liangzhibo/HyperOffload-research/joint-tuning-20261101')
name = sys.argv[1]
assert name.isalnum()
root = base / name
plan = root / 'plan.json'
assert plan.is_file() and not (root / 'launch.json').exists() and not (root / 'status.json').exists()
with (root / 'controller.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', '/tmp/liangzhibo-run-compute-queue.py', str(plan)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
record = dict(pid=process.pid, started=time.time(),
              start_ticks=(Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
              plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
(root / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
