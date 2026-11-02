"""One immutable diagnostic queue, advisory device lease and ownership guard."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from resource_guard import OwnershipTracker, start_time

ROOT = Path(__file__).resolve().parent
PRIVATE = '/tmp/liangzhibo-joint-baseline-20261101'
CONTAINER = 'mindspeed-qwen25-32b-fsdp'
CARDS = [6, 7]
state = dict(pid=os.getpid(), start_ticks=start_time(os.getpid()), started=time.time(),
             command=sys.argv, status='starting', trials=[], cards=CARDS, node=2)


def save(**updates):
    state.update(updates)
    path = ROOT / 'suite-state.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)


def resources():
    raw = {str(card): subprocess.check_output(
        ['npu-smi', 'info', '-t', 'proc-mem', '-i', str(card)], text=True, timeout=30)
        for card in CARDS}
    active = {int(p) for text in raw.values() for p in re.findall(r'Process id:\s*(\d+)', text)}
    top = subprocess.check_output(['docker', 'top', CONTAINER, '-eo', 'pid,args'], text=True, timeout=30)
    own = {int(line.split()[0]) for line in top.splitlines()[1:]
           if PRIVATE in line and ('run_wrapper.py' in line or 'target_check.py' in line)}
    return active, own, raw


def stop_own():
    script = '''import os,signal
from pathlib import Path
targets={%r,%r}
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=os.getpid():
  try:
   args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
   if any(a in targets for a in args): os.kill(int(p.name),signal.SIGTERM)
  except (OSError,UnicodeError): pass
''' % (PRIVATE + '/run_wrapper.py', PRIVATE + '/target_check.py')
    subprocess.run(['docker', 'exec', CONTAINER, '/usr/local/python3.12.13/bin/python', '-c', script], check=True)


DEADLINE = 1793575800  # 2026-11-01 23:30 UTC

def wait_idle(label):
    while True:
        if time.time() >= DEADLINE:
            raise TimeoutError('goal deadline')
        active, own, raw = resources()
        if own:
            raise RuntimeError('Existing matching workers found; refusing a duplicate launch')
        if not active:
            return
        save(status='waiting_for_resources', current=label, active=sorted(active), resources=raw)
        time.sleep(15)


def verify():
    return subprocess.check_output(['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        CONTAINER, '/usr/local/python3.12.13/bin/python', PRIVATE + '/verify_sources.py'], text=True)


def run(q):
    label = q['trial']
    wait_idle(label)
    proof = verify()
    folder = ROOT / label
    folder.mkdir(exist_ok=False)
    (folder / 'source-before.json').write_text(proof)
    (folder / 'config.json').write_text(json.dumps(q, indent=2))
    rec = dict(trial=label, arm=q['arm'], profile=q.get('profile'), started=time.time(),
               interference=False, cards=CARDS)
    stop = folder / 'monitor.stop'
    with (folder / 'monitor.log').open('w') as monlog, (folder / 'run.log').open('w') as log:
        monitor = subprocess.Popen([sys.executable, str(ROOT / 'device_monitor.py'),
            '--output', str(folder / 'device-memory.jsonl'), '--stop-file', str(stop),
            '--cards', '6,7'], stdout=monlog, stderr=subprocess.STDOUT)
        child = subprocess.Popen(['docker', 'exec', CONTAINER, 'bash', PRIVATE + '/run_trial.sh',
                                  label, q['arm']], stdout=log, stderr=subprocess.STDOUT)
        rec.update(pid=child.pid, start_ticks=start_time(child.pid))
        save(status='running', current=label, child_pid=child.pid, child_start_ticks=rec['start_ticks'])
        ownership = OwnershipTracker()
        try:
            while child.poll() is None:
                if time.time() >= DEADLINE:
                    raise TimeoutError('goal deadline')
                active, own, raw = resources()
                with (folder / 'resource-monitor.jsonl').open('a') as out:
                    out.write(json.dumps(dict(time=time.time(), active=sorted(active), own=sorted(own))) + '\n')
                foreign = ownership.foreign(active, own)
                if foreign and not rec['interference']:
                    rec.update(interference=True, foreign=sorted(foreign), resources=raw)
                    save(status='resource_interference', current=label, foreign=sorted(foreign))
                    stop_own()
                if monitor.poll() is not None and not stop.exists():
                    raise RuntimeError('device memory monitor exited unexpectedly')
                time.sleep(5)
        except BaseException:
            stop_own()
            raise
        finally:
            stop.touch()
            monitor.wait(timeout=30)
    if label != 'target_check':
        subprocess.run(['docker', 'cp', CONTAINER + ':' + PRIVATE + '/results/' + label + '/.',
                        str(folder)], check=True)
    (folder / 'source-after.json').write_text(verify())
    rec.update(exit_code=child.returncode, finished=time.time(), monitor_exit=monitor.returncode)
    rec['status'] = 'complete' if child.returncode == 0 and not rec['interference'] else 'failed'
    if rec['status'] == 'complete' and label != 'target_check':
        for rank in (0, 1):
            result = json.loads((folder / f'result_rank{rank}.json').read_text())
            audit = json.loads((folder / f'full_audit_rank{rank}.json').read_text())
            if result['status'] != 'steps_complete' or result['steps'] != q['steps']:
                raise RuntimeError(f'incomplete result {label} rank{rank}')
            assert result['hyper_offload'] == q['offload']
            assert result['recompute_layers'] == q['recompute_layers']
            if q['offload']:
                assert result['offload_communication_masking']
                assert result.get('offload_plan_actions', {}).get('COPY_D2H', 0) > 0 and result.get('d2h_gib', 0) > 0
            else:
                assert not result.get('d2h_gib', 0)
            if q['mode'] == 'correctness':
                initial=json.loads((folder / f'initial_state_rank{rank}.json').read_text())
                inputs=json.loads((folder / f'input_rank{rank}.json').read_text())
                assert initial['status']=='complete' and inputs['status']=='complete'
                assert len(initial['tensors'])==579
                for step in q['audit_steps']:
                    full=json.loads((folder / f'full_step{step}_rank{rank}.json').read_text())
                    assert full['status']=='complete' and full['step']==step and len(full['tensors'])==579
                    assert all(t['parameter']['finite'] and t['gradient'] is not None and t['gradient']['finite'] for t in full['tensors'])
            if audit['status'] != 'complete':
                raise RuntimeError('full state audit incomplete')
    (folder / 'status.json').write_text(json.dumps(rec, indent=2))
    state['trials'].append(rec)
    save()
    if rec['status'] != 'complete':
        raise RuntimeError(f'{label} failed; preserving evidence and stopping queue')


def main():
    lock = (ROOT / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'suite-state.json').exists():
        raise RuntimeError('Existing state found; explicit identity audit required, no automatic restart')
    save()
    leases = []
    try:
        for card in CARDS:
            handle = open(f'/tmp/pypto-npu-{card}.lock', 'a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            leases.append(handle)
        (ROOT / 'RESERVATION.json').write_text(json.dumps(dict(
            pid=os.getpid(), start_ticks=state['start_ticks'], cards=CARDS,
            node=2, cpus='48-71', acquired=time.time(), advisory=True), indent=2))
        for q in json.loads((ROOT / 'queue.json').read_text()):
            run(q)
        save(status='queue_complete', finished=time.time(), current=None)
    except BaseException as exc:
        save(status='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        for handle in leases:
            handle.close()


if __name__ == '__main__':
    main()
