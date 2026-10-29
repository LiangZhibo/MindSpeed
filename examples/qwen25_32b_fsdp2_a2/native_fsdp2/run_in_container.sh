#!/usr/bin/env bash
set -euo pipefail

run_id="$1"
npu_devices="$2"
numa_node="$3"
[[ "$run_id" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "RUN_ID must contain only letters, digits, dot, underscore, or dash" >&2; exit 2; }
workspace="${QWEN_WORKSPACE:-/data/disk1/machangwei}"
repo="${MINDSPEED_REPO:-/MindSpeed}"
script_dir="$repo/examples/qwen25_32b_fsdp2_a2/native_fsdp2"
dependency_root="${DEPENDENCY_ROOT:-$workspace/deps}"
model_path="${MODEL_PATH:-$workspace/Qwen2.5-32B}"
master_port="${MASTER_PORT:-29621}"
output_root="${OUTPUT_ROOT:-$workspace/profile}"
python_bin="$dependency_root/venv/bin/python"
lock_file="$output_root/.native_fsdp2_comparison.lock"

[[ "$npu_devices" =~ ^[0-7],[0-7]$ ]] || { echo "NPU_DEVICES must be a pair, got $npu_devices" >&2; exit 2; }
[[ "$numa_node" =~ ^[0-7]$ ]] || { echo "NUMA_NODE must be in 0..7" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "missing Python environment: $python_bin" >&2; exit 2; }
[[ -f "$model_path/model.safetensors.index.json" ]] || { echo "missing model index: $model_path" >&2; exit 2; }
[[ -r /etc/hccn.conf ]] || { echo "container must mount /etc/hccn.conf" >&2; exit 2; }
command -v hccn_tool >/dev/null || { echo "container must provide hccn_tool" >&2; exit 2; }

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
source "$script_dir/network.env"

mkdir -p "$output_root" "$workspace/logs"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "another native FSDP2 comparison is running" >&2
    exit 75
fi

export PATH=/opt/codex-node/bin:/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH="$dependency_root/megatron_03c_compat:$repo:$dependency_root/Megatron-LM-core_r0.18.0:$dependency_root/MegatronAdaptor-core_r0.18.0:$dependency_root/TransformerEngineNPU-26.2.0"
export ASCEND_RT_VISIBLE_DEVICES="$npu_devices"

case_root="native-fsdp2-1030-$run_id"
baseline_name="$case_root-recompute-r34"
offload_name="$case_root-hyperoffload-peak-mask"

run_case() {
    local name="$1"
    local recompute_layers="$2"
    local hyper_offload="$3"
    local result_dir="$output_root/$name"
    local log_file="$workspace/logs/$name.log"
    local offload_args=()
    local planner=none
    local communication_masking=false

    [[ ! -e "$result_dir" && ! -e "$log_file" ]] || {
        echo "output already exists: $name; choose another RUN_ID" >&2
        return 2
    }
    mkdir -p "$result_dir" "$workspace/logs"
    if [[ "$hyper_offload" == 1 ]]; then
        planner=peak
        communication_masking=true
        offload_args+=(--hyper-offload --offload-device-mb 2048 --offload-host-mb 131072
            --offload-planner peak --offload-communication-masking)
    fi

    printf 'started_at=%s\nbranch=%s\ncommit=%s\ncase=%s\nvisible_npus=%s\nnuma_node=%s\nlayers=48\nsequence_length=4096\nrecompute_layers=%s\nhyper_offload=%s\noffload_planner=%s\ncommunication_masking=%s\noffload_device_mb=%s\noffload_host_mb=%s\nallocator_conf=%s\nhccl_connect_timeout=%s\nhccl_exec_timeout=%s\nhccl_host_socket_port_range=%s\nhccl_npu_socket_port_range=%s\nsteps=10\nprofile=0\n' \
        "$(date -Is)" "$(git -c safe.directory="$repo" -C "$repo" branch --show-current)" \
        "$(git -c safe.directory="$repo" -C "$repo" rev-parse HEAD)" \
        "$name" "$npu_devices" "$numa_node" "$recompute_layers" "$hyper_offload" \
        "$planner" "$communication_masking" "$([[ "$hyper_offload" == 1 ]] && printf 2048 || printf none)" \
        "$([[ "$hyper_offload" == 1 ]] && printf 131072 || printf none)" "$PYTORCH_NPU_ALLOC_CONF" \
        "$HCCL_CONNECT_TIMEOUT" "$HCCL_EXEC_TIMEOUT" "$HCCL_HOST_SOCKET_PORT_RANGE" \
        "$HCCL_NPU_SOCKET_PORT_RANGE" > "$result_dir/launch.txt"

    local command=("$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2
        --master_port="$master_port" "$script_dir/run_qwen25_native_mindspeed_fsdp2.py"
        --model-path "$model_path" --seq-length 4096 --layers 48 --no-profile
        --profile-path "$result_dir" --recompute-layers "$recompute_layers"
        --extra-steps 9 --use-torch-fsdp2 --fsdp2-config-path "$script_dir/native_fsdp2_config.yaml"
        "${offload_args[@]}")
    printf 'command=%q ' "${command[@]}" >> "$result_dir/launch.txt"
    printf '\n' >> "$result_dir/launch.txt"

    set +e
    "$python_bin" "$script_dir/numa_exec.py" --node "$numa_node" --memory-node "$numa_node" -- \
        "${command[@]}" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    printf 'finished_at=%s\nexit_code=%s\n' "$(date -Is)" "$status" > "$result_dir/exit.txt"
    ((status == 0)) || return "$status"

    "$python_bin" - "$result_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rank in (0, 1):
    result = json.loads((root / f"result_rank{rank}.json").read_text())
    assert result["status"] == "steps_complete", (rank, result["status"])
    assert len(result["extra_step_results"]) == 9, (rank, len(result["extra_step_results"]))
    for step in [result, *result["extra_step_results"]]:
        assert step["probe_grad_max"] > 0, (rank, step)
        assert step["parameter_update_max"] > 0, (rank, step)
PY
}

run_case "$baseline_name" 34 0
run_case "$offload_name" 0 1

"$python_bin" - "$output_root/$baseline_name" "$output_root/$offload_name" <<'PY'
import json
import statistics
import sys
from pathlib import Path

for label, path in zip(("recompute", "peak_mask"), sys.argv[1:]):
    result = json.loads((Path(path) / "result_rank0.json").read_text())
    replay = [step["elapsed_seconds"] for step in result["extra_step_results"]]
    print(f"{label}: first_step={result['first_step_elapsed_seconds']:.3f}s "
          f"replay_mean={statistics.mean(replay):.3f}s steps=10")
PY
