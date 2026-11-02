"""Prepare this revision's standalone, fixed experiment harness."""
import hashlib
import json
from pathlib import Path
import sys

tag = sys.argv[1]
assert tag.isalnum()
root = Path(__file__).resolve().parent
out = root / tag
private = '/tmp/liangzhibo-joint-tuning-20261101/' + tag
assert out.is_dir(), 'package the immutable source first'
assert not any((out / name).exists() for name in ('run_wrapper.py', 'run_trial.sh', 'harness-manifest.json')), (
    'never overwrite an existing frozen harness')
old = root / 'baseline/run_wrapper.py'
assert hashlib.sha256(old.read_bytes()).hexdigest() == '3a42e38f9dde7a6ebbdf69132ca6335fda116ac6c58c9cabd54941cbb0d21749'
wrapper = old.read_text()
wrapper = wrapper.replace("'--offload-device-mb','2048','--offload-host-mb','131072'",
                          "'--offload-device-mb',str(config.get('device_mb',2048)),'--offload-host-mb','131072'")
wrapper = wrapper.replace("sys.path.insert(0,str(entry.parent))", """if config.get('autotune'):
    sys.argv += ['--hyper-offload-autotune','--offload-tuning-steps',str(config.get('tuning_steps',12))]
    sys.argv += ['--offload-tuning-probe',config.get('probe','norm')]
    if 'window_probe_multiplier' in config:
        sys.argv += ['--offload-window-probe-multiplier',str(config['window_probe_multiplier'])]
    if config.get('window_probe_limit_prefetch') is False:
        sys.argv += ['--offload-window-probe-unlimited-prefetch']
    if 'resident_mlp_fraction' in config:
        sys.argv += ['--offload-resident-mlp-fraction',str(config['resident_mlp_fraction'])]
    if config.get('device_peak_fraction') is not None:
        peak_mb = int(torch.npu.get_device_properties(rank).total_memory * config['device_peak_fraction'] / 1024**2)
        sys.argv += ['--offload-tuning-device-peak-mb',str(peak_mb)]
    elif config.get('device_peak_mb') is not None:
        sys.argv += ['--offload-tuning-device-peak-mb',str(config['device_peak_mb'])]
sys.path.insert(0,str(entry.parent))""")
wrapper = wrapper.replace("captured={}", """captured={}
if config.get('profile_stable'):
    import torch_npu
    def stable_profile(args):
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, profile_memory=True,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, l2_cache=False),
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=config.get('profile_skip_first',12)),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / 'profiler')))
    module.make_profile = stable_profile
if config.get('profile_runtime_tags'):
    assert config.get('profile_stable'), 'runtime tags are diagnostic only'
    import functools
    from torch.utils._python_dispatch import _disable_current_modes
    from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager, PhysicalBuffer
    offload_api = importlib.import_module('mindspeed.core.memory.hyper_offload.api.session')
    def tagged(name, function):
        @functools.wraps(function)
        def invoke(*args, **kwargs):
            # Profiling metadata must never become a model activation op.
            with _disable_current_modes(), torch.profiler.record_function('HyperOffload::' + name):
                return function(*args, **kwargs)
        return invoke
    for method in ('copy_d2h', 'copy_h2d', 'release_device', 'release_host', 'wait_for_transfers'):
        setattr(ResidencyManager, method, tagged(method, getattr(ResidencyManager, method)))
    for method in ('prepare_for_write', 'device_storage'):
        setattr(PhysicalBuffer, method, tagged(method, getattr(PhysicalBuffer, method)))
    offload_api._empty_accelerator_cache = tagged('empty_cache', offload_api._empty_accelerator_cache)
if config.get('capture_trace'):
    from dataclasses import asdict
    original_build = module.PeakResidencyPlanner.build
    trace_index = [0]
    def save_trace(self, trace):
        began = time.perf_counter()
        schedule = original_build(self, trace)
        planning_seconds = time.perf_counter() - began
        trace_index[0] += 1
        if rank == 0:
            value = asdict(trace)
            value['retained_sids'] = sorted(trace.retained_sids)
            for op in value['ops']:
                for access in op['accesses']:
                    access['kind'] = access['kind'].name
            value['diagnostic'] = dict(planning_seconds=planning_seconds,
                masking=self._enable_communication_masking, window_scale=self._transfer_window_scale,
                limit_prefetch_window=self._limit_prefetch_window)
            (output / f'trace{trace_index[0]}_rank{rank}.json').write_text(json.dumps(value))
        return schedule
    module.PeakResidencyPlanner.build = save_trace
""")
wrapper = wrapper.replace('performance_usable=not diagnostic',
                          "performance_usable=not diagnostic and not config.get('profile_stable',False)")
(out / 'run_wrapper.py').write_text(wrapper)
script = '''#!/usr/bin/env bash
set -euo pipefail
root=PRIVATE
repo="$root/source"
deps=/data/disk1/machangwei/deps
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
source "$repo/examples/qwen25_32b_fsdp2_a2/native_fsdp2/network.env"
export PYTHONPATH="$deps/megatron_03c_compat:$repo:$deps/Megatron-LM-core_r0.18.0:$deps/MegatronAdaptor-core_r0.18.0:$deps/TransformerEngineNPU-26.2.0"
export ASCEND_RT_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH="$root/results/process-logs" TORCH_EXTENSIONS_DIR="$root/torch-extensions"
cd "$repo"
python_bin="$deps/venv/bin/python"
trial=$1
if [[ "$trial" == small_npu || "$trial" == small_npu_rng ]]; then
  command=("$repo/tools/hyper_offload_joint_goal/small_npu.py" --output "$root/results/$trial")
  if [[ "$trial" == small_npu_rng ]]; then
    command+=(--dropout-probability 0.1)
  fi
else
  command=("$root/run_wrapper.py" "$root/configs/$trial.json")
fi
exec "$python_bin" "$repo/examples/qwen25_32b_fsdp2_a2/native_fsdp2/numa_exec.py" --node 2 --memory-node 2 -- \
 "$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 "${command[@]}"
'''.replace('PRIVATE', private)
(out / 'run_trial.sh').write_text(script)
manifest = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ('run_wrapper.py', 'run_trial.sh')}
(out / 'harness-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
