"""Read compact progress from one owned trial, without touching its workers."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

tag, trial = sys.argv[1:3]
assert tag.isalnum() and re.fullmatch(r'[a-zA-Z0-9_]+', trial)
root = Path('/data/disk/liangzhibo/HyperOffload-research/joint-tuning-20261101') / tag / trial
status = json.loads((root / 'status.json').read_text())
print(json.dumps(status))


def summarize_policy(policy):
    regions = policy.get('recompute_regions')
    return dict(count=policy['recompute_count'], scale=policy['transfer_window_scale'],
                resident_mb=policy['resident_mb'], prefetch_on_copy_stream=policy.get('prefetch_on_copy_stream', False),
                trim_before_replay=policy.get('trim_before_replay', True),
                defer_h2d_release=policy.get('defer_h2d_release', False),
                trim_after_offload=policy.get('trim_after_offload', False),
                limit_prefetch_window=policy.get('limit_prefetch_window', True),
                resident_regions=dict(Counter(name.partition('.')[2] or 'layer' for name in policy.get('resident_regions', ()))),
                regions=None if regions is None else dict(Counter(name.partition('.')[2] or 'layer' for name in regions)))


for rank in (0, 1):
    if trial in ('small_npu', 'small_npu_rng'):
        file = root / f'qualification_rank{rank}.json'
        if file.exists():
            data = json.loads(file.read_text())
            print(json.dumps({key: data.get(key) for key in ('rank', 'status', 'steps', 'frozen_at_step', 'copies', 'trace_collections', 'plan_seconds', 'deferred_h2d_releases', 'post_offload_trim_calls', 'dropout_probability', 'rng_state_exact_steps')}))
        else:
            print(json.dumps(dict(rank=rank, qualification='not yet copied to host')))
        continue
    file = root / f'result_rank{rank}.json'
    if file.exists():
        data = json.loads(file.read_text())
    else:
        private = f'/tmp/liangzhibo-joint-tuning-20261101/{tag}/results/{trial}/result_rank{rank}.json'
        code = 'from pathlib import Path;p=Path(%r);print(p.read_text() if p.exists() else "null")' % private
        raw = subprocess.check_output(['docker', 'exec', 'mindspeed-qwen25-32b-fsdp',
                                       '/usr/local/python3.12.13/bin/python', '-c', code], text=True)
        data = json.loads(raw)
    if data is None:
        print(json.dumps(dict(rank=rank, result='not yet available')))
        continue
    tuning = data.get('joint_tuning', {})
    steps = data.get('extra_step_results', [])
    result = dict(rank=rank, status=data.get('status'), steps=data.get('steps'), error=data.get('error'),
                  phase=tuning.get('phase'), frozen=tuning.get('frozen_at_step'),
                  search_seconds=tuning.get('search_seconds'), plan_seconds=tuning.get('plan_seconds'),
                  host_gib=data.get('host_pool_allocated_gib'),
                  trials=[dict(policy=summarize_policy(t['policy']), samples=t['samples'], feasible=t['feasible'],
                               peak_gib=t.get('device_peak_bytes', 0) / 2**30)
                          for t in tuning.get('trials', [])],
                  recent_steps=[dict(step=s['index'], seconds=s['elapsed_seconds'], phase=s.get('tuning_phase'),
                                     peak_gib=s.get('npu_peak_allocated_gib'))
                                for s in steps[-4:]])
    if 'policy' in tuning:
        result['policy'] = summarize_policy(tuning['policy'])
    print(json.dumps(result))
