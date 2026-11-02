"""Prepare a private, frozen reproduction of the best verified raw baseline."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
OLD = Path('/Users/liangzhibo/Desktop/MindSpeed/tools/qwen_performance_optimization/host_goal_20261031/performance_review')
PRIVATE = '/tmp/liangzhibo-joint-baseline-20261101'
out = HERE / 'baseline'
out.mkdir(exist_ok=False)
old_private = '/tmp/liangzhibo-host-performance-20261031'
for name in ('control_pr11.py', 'run_trial.sh', 'run_wrapper.py', 'verify_sources.py',
             'resource_guard.py', 'device_monitor.py'):
    code = (OLD / name).read_text().replace(old_private, PRIVATE)
    if name == 'verify_sources.py':
        code = code.replace("for arm in ('B0', 'B1'):", 'for arm in manifest:')
    if name == 'control_pr11.py':
        code = code.replace('def wait_idle(label):', 'DEADLINE = 1793575800  # 2026-11-01 23:30 UTC\n\ndef wait_idle(label):')
        code = code.replace('        active, own, raw = resources()',
                            "        if time.time() >= DEADLINE:\n            raise TimeoutError('goal deadline')\n        active, own, raw = resources()", 1)
        code = code.replace('            while child.poll() is None:',
                            "            while child.poll() is None:\n                if time.time() >= DEADLINE:\n                    raise TimeoutError('goal deadline')")
    (out / name).write_text(code)
manifest = json.loads((OLD / 'source-manifest.json').read_text())
manifest = {'B1': manifest['B1']}
(out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2))
shutil.copy2(OLD / manifest['B1']['archive'], out / manifest['B1']['archive'])
(out / 'configs').mkdir()
cfg = dict(trial='baseline_r33_24', arm='B1', comparison_arm='CURRENT_BEST_RAW',
           profile=False, steps=24, source=PRIVATE + '/source/B1',
           output=PRIVATE + '/results/baseline_r33_24', allocator_fraction=.95,
           model='/data/disk1/machangwei/Qwen2.5-32B', commit=manifest['B1']['commit'],
           deterministic=True, recompute_layers=33, offload=True,
           mask_communication=True, mode='performance', audit_steps=[])
(out / 'configs/baseline_r33_24.json').write_text(json.dumps(cfg, indent=2))
(out / 'queue.json').write_text(json.dumps([cfg], indent=2))
plan = dict(deadline_beijing='2026-11-02T07:30:00+08:00',
            target='stable median max-rank step time <= 0.90 * matched best raw baseline',
            excluded='compression; warmup amortization', baseline=cfg,
            tuning_budget_steps=10, maximum_tuning_steps=12,
            stable_metric='median of max-rank elapsed_seconds; predefined stable suffix; independent paired runs',
            constraints='same model/input/optimizer/dtype/parallelism/allocator/Host and device memory constraints',
            initial_status='baseline reproduction started; adaptive policy not implemented or measured yet')
(HERE / 'GOAL.json').write_text(json.dumps(plan, indent=2))
kit = HERE / 'joint-baseline-kit.tar.gz'
with tarfile.open(kit, 'w:gz') as tf:
    for path in sorted(out.rglob('*')):
        if path.is_file():
            tf.add(path, arcname=str(path.relative_to(out)), recursive=False)
print(json.dumps(dict(kit=str(kit), bytes=kit.stat().st_size,
                      sha256=hashlib.sha256(kit.read_bytes()).hexdigest())))
