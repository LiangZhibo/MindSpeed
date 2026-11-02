"""Host-side single-trial launcher with an advisory lease and ownership checks."""
import fcntl
import json
import os
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
previous = Path('/data/disk/liangzhibo/HyperOffload-research/joint-baseline-20261101')
sys.path.insert(0, str(previous))
from resource_guard import OwnershipTracker, start_time

folder = root / trial
folder.mkdir(exist_ok=False)
state = dict(status='preparing', pid=os.getpid(), start_ticks=start_time(os.getpid()),
             started=time.time(), tag=tag, trial=trial, cards=[6, 7])


def save(**updates):
    state.update(updates)
    target = folder / 'status.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(target)


def resource_state():
    raw = [subprocess.check_output(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(card)], text=True)
           for card in (6, 7)]
    active = {int(pid) for data in raw for pid in re.findall(r'Process id:\s*(\d+)', data)}
    top = subprocess.check_output(['docker', 'top', container, '-eo', 'pid,args'], text=True)
    own = {int(line.split()[0]) for line in top.splitlines()[1:]
           if private in line and ('small_npu.py' in line or 'run_wrapper.py' in line)}
    return active, own


def stop_own():
    # Exact script arguments, never a broad kill of Python/NPU processes.
    targets = [private + '/source/tools/hyper_offload_joint_goal/small_npu.py', private + '/run_wrapper.py']
    code = '''import os,signal
from pathlib import Path
targets=%r
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=os.getpid():
  try:
   args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
   if any(a in targets for a in args): os.kill(int(p.name),signal.SIGTERM)
  except (OSError,UnicodeError): pass
''' % targets
    subprocess.run(['docker', 'exec', container, '/usr/local/python3.12.13/bin/python', '-c', code], check=True)


def verify():
    code = '''import hashlib,json
from pathlib import Path
r=Path(%r); m=json.loads((r/'source-manifest.json').read_text())
for name,digest in m['files'].items(): assert hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(status='verified',files=len(m['files']),archive_sha256=m['archive_sha256'])))
''' % private
    return subprocess.check_output(['docker', 'exec', container, '/usr/local/python3.12.13/bin/python', '-c', code], text=True)


def main():
    save()
    leases = []
    try:
        for card in (6, 7):
            lease = open(f'/tmp/pypto-npu-{card}.lock', 'a+')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            leases.append(lease)
        while True:
            active, own = resource_state()
            if own:
                raise RuntimeError('matching worker already exists; do not duplicate')
            if not active:
                break
            if time.time() >= 1793575800:
                raise TimeoutError('goal deadline')
            save(status='waiting_for_resources', foreign=sorted(active))
            time.sleep(10)
        (folder / 'source-before.json').write_text(verify())
        stop_file = folder / 'monitor.stop'
        with (folder / 'monitor.log').open('x') as monlog, (folder / 'run.log').open('x') as log:
            monitor = subprocess.Popen([sys.executable, str(previous / 'device_monitor.py'),
                                        '--output', str(folder / 'device-memory.jsonl'),
                                        '--stop-file', str(stop_file), '--cards', '6,7'],
                                       stdout=monlog, stderr=subprocess.STDOUT)
            child = subprocess.Popen(['docker', 'exec', container, 'bash', private + '/run_trial.sh', trial],
                                     stdout=log, stderr=subprocess.STDOUT)
            save(status='running', child_pid=child.pid, child_start_ticks=start_time(child.pid))
            ownership = OwnershipTracker()
            try:
                while child.poll() is None:
                    active, own = resource_state()
                    with (folder / 'resource-monitor.jsonl').open('a') as output:
                        output.write(json.dumps(dict(time=time.time(), active=sorted(active), own=sorted(own))) + '\n')
                    foreign = ownership.foreign(active, own)
                    if foreign:
                        save(interference=True, foreign=sorted(foreign))
                        raise RuntimeError('foreign NPU process detected')
                    if time.time() >= 1793575800 or time.time() - state['started'] > 2700:
                        raise TimeoutError('trial or goal deadline')
                    if monitor.poll() is not None:
                        raise RuntimeError('memory monitor exited prematurely')
                    time.sleep(5)
            except BaseException:
                stop_own()
                child.wait(timeout=60)
                raise
            finally:
                stop_file.touch()
                monitor.wait(timeout=30)
        subprocess.run(['docker', 'cp', container + ':' + private + '/results/' + trial + '/.', str(folder)], check=False)
        (folder / 'source-after.json').write_text(verify())
        save(status='complete' if child.returncode == 0 else 'failed', exit_code=child.returncode, finished=time.time())
        return child.returncode
    except BaseException as error:
        save(status='failed', error=repr(error), finished=time.time())
        raise
    finally:
        for lease in leases:
            lease.close()


raise SystemExit(main())
