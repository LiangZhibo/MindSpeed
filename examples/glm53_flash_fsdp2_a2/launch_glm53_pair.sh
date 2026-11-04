#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/disk1/machangwei
ENV_ROOT="$ROOT/glm53-env"
MS_ROOT="$ROOT/MindSpeed"
MM_ROOT="$ROOT/MindSpeed-MM"
EXAMPLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXTERNAL_RESERVATION=/data/disk/liangzhibo/HyperOffload-research/generalized-tuning-20261102/reservation.json
QWEN_LOCK="$ROOT/logs/qwen25-r28-r31-continuation.lock"

usage() {
    echo "Usage: $0 {strict_baseline|hyperoffload} NPU_A,NPU_B" >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }
MODE=$1
PAIR=$2
[[ "$MODE" =~ ^(strict_baseline|hyperoffload)$ ]] || { usage; exit 2; }
[[ "$PAIR" =~ ^[0-7],[0-7]$ ]] || { usage; exit 2; }
IFS=, read -r NPU_A NPU_B <<< "$PAIR"
[[ "$NPU_A" != "$NPU_B" ]] || { echo "two different NPU IDs are required" >&2; exit 2; }

declare -A BDF_BY_NPU=(
    [0]=0000:c1:00.0 [1]=0000:c2:00.0
    [2]=0000:81:00.0 [3]=0000:82:00.0
    [4]=0000:01:00.0 [5]=0000:02:00.0
    [6]=0000:41:00.0 [7]=0000:42:00.0
)
NODE_A=$(cat "/sys/bus/pci/devices/${BDF_BY_NPU[$NPU_A]}/numa_node")
NODE_B=$(cat "/sys/bus/pci/devices/${BDF_BY_NPU[$NPU_B]}/numa_node")
[[ "$NODE_A" -ge 0 && "$NODE_A" == "$NODE_B" ]] || {
    echo "NPUs $NPU_A,$NPU_B are not attached to one NUMA node" >&2
    exit 1
}
MIN_NUMA_FREE_MIB=${GLM53_MIN_NUMA_FREE_MIB:-163840}
NUMA_FREE_KIB=$(awk '/MemFree:/ {print $4; exit}' "/sys/devices/system/node/node$NODE_A/meminfo")
[[ "$MIN_NUMA_FREE_MIB" =~ ^[0-9]+$ && "$NUMA_FREE_KIB" =~ ^[0-9]+$ ]] || {
    echo "cannot determine NUMA free-memory requirement" >&2; exit 1;
}
(( NUMA_FREE_KIB / 1024 >= MIN_NUMA_FREE_MIB )) || {
    echo "NUMA node $NODE_A has $((NUMA_FREE_KIB / 1024)) MiB free; requires $MIN_NUMA_FREE_MIB MiB" >&2
    exit 1
}

check_npu() {
    local npu=$1 info health hbm
    health=$(npu-smi info -t health -i "$npu" | awk -F: '/Health[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    [[ "$health" == OK ]] || { echo "NPU $npu health is ${health:-unknown}" >&2; return 1; }
    info=$(npu-smi info -t proc-mem -i "$npu")
    if grep -Eq 'Process id:[[:space:]]*[0-9]+' <<< "$info"; then
        echo "NPU $npu has an existing process; refusing to start" >&2
        printf '%s\n' "$info" >&2
        return 1
    fi
    hbm=$(npu-smi info -t usages -i "$npu" | awk -F: '/HBM Usage Rate/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
    [[ "$hbm" =~ ^[0-9]+$ && "$hbm" -le 10 ]] || {
        echo "NPU $npu HBM usage is ${hbm:-unknown}%; expected at most 10%" >&2
        return 1
    }
}

check_npu "$NPU_A"
check_npu "$NPU_B"
if [[ "$NPU_A$NPU_B" == 67 || "$NPU_A$NPU_B" == 76 ]]; then
    reservation_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "$EXTERNAL_RESERVATION" 2>/dev/null || true)
    [[ "$reservation_status" == released ]] || { echo "external NPU6/7 reservation is not released" >&2; exit 1; }
    flock -n "$QWEN_LOCK" -c true || { echo "Qwen continuation still holds its NPU lock" >&2; exit 1; }
fi
if pgrep -af '[t]orchrun.*(run_glm53.py|mindspeed_mm/fsdp/train/trainer.py)|[t]orch.distributed.run.*(run_glm53.py|mindspeed_mm/fsdp/train/trainer.py)' >/dev/null; then
    echo "another GLM/MindSpeed-MM training process is already running" >&2
    exit 1
fi
docker inspect -f '{{.State.Running}}' mindspeed-glm53-flash-env 2>/dev/null | grep -qx true || {
    echo "container mindspeed-glm53-flash-env is not running" >&2
    exit 1
}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ENV_ROOT/experiments/1103_glm53/${STAMP}_${MODE}_npu${NPU_A}${NPU_B}"
mkdir -p "$ENV_ROOT/experiments/1103_glm53"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/metrics"
case "$MODE" in
    strict_baseline) RUNNER_MODE=baseline ;;
    hyperoffload) RUNNER_MODE=hyperoffload ;;
esac
PORT=${MASTER_PORT:-29653}
if [[ "$MODE" == hyperoffload ]]; then
    PORT=${MASTER_PORT:-29654}
fi
TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
GLM53_DISABLE_RECOMPUTE_LAYERS=${GLM53_DISABLE_RECOMPUTE_LAYERS:-0,3}
GLM53_STOP_AFTER_ITERS=${GLM53_STOP_AFTER_ITERS:-10}
GLM53_OFFLOAD_DEVICE_MB=${GLM53_OFFLOAD_DEVICE_MB:-4096}
GLM53_OFFLOAD_HOST_MB=${GLM53_OFFLOAD_HOST_MB:-32768}
GLM53_OFFLOAD_PLANNER=${GLM53_OFFLOAD_PLANNER:-peak}
GLM53_GROUPED_MOE_EP1=${GLM53_GROUPED_MOE_EP1:-1}
GLM53_TRITON_CACHE_DIR=${GLM53_TRITON_CACHE_DIR:-}
GLM53_PROFILE_STEP=${GLM53_PROFILE_STEP:-}
if [[ -n "$GLM53_PROFILE_STEP" ]]; then
    [[ "$GLM53_PROFILE_STEP" =~ ^[0-9]+$ && "$GLM53_PROFILE_STEP" -ge 1 ]] || {
        echo "GLM53_PROFILE_STEP must be a positive training iteration" >&2; exit 2;
    }
    PROFILE_LIMIT=$GLM53_STOP_AFTER_ITERS
    [[ "$PROFILE_LIMIT" -gt 0 ]] || PROFILE_LIMIT=20
    (( GLM53_PROFILE_STEP <= PROFILE_LIMIT )) || {
        echo "profile step $GLM53_PROFILE_STEP exceeds the $PROFILE_LIMIT observed iterations" >&2; exit 2;
    }
fi
printf 'mode=%s\nrunner_mode=%s\nnpu_pair=%s\nnuma_node=%s\nmaster_port=%s\ntask_queue_enable=%s\ngrouped_moe_ep1=%s\n' \
    "$MODE" "$RUNNER_MODE" "$PAIR" "$NODE_A" "$PORT" "$TASK_QUEUE_ENABLE" "$GLM53_GROUPED_MOE_EP1" > "$RUN_DIR/launch.txt"
printf 'configured_train_iters=20\nstop_after_iters=%s\n' "$GLM53_STOP_AFTER_ITERS" >> "$RUN_DIR/launch.txt"
printf 'numa_free_kib_before_launch=%s\nminimum_numa_free_mib=%s\n' "$NUMA_FREE_KIB" "$MIN_NUMA_FREE_MIB" >> "$RUN_DIR/launch.txt"
if [[ "$MODE" == hyperoffload ]]; then
    printf 'disable_recompute_language_layers=%s\n' "$GLM53_DISABLE_RECOMPUTE_LAYERS" >> "$RUN_DIR/launch.txt"
fi
printf 'mindspeed_root=%s\nmindspeed_commit=%s\nmindspeed_mm_commit=%s\nexample_root=%s\nallocator=expandable_segments:True\n' \
    "$MS_ROOT" "$(git -C "$MS_ROOT" rev-parse HEAD)" "$(git -C "$MM_ROOT" rev-parse HEAD)" "$EXAMPLE_ROOT" >> "$RUN_DIR/launch.txt"
if [[ "$MODE" == hyperoffload ]]; then
    printf 'offload_device_mb=%s\noffload_host_mb=%s\noffload_planner=%s\n' \
        "$GLM53_OFFLOAD_DEVICE_MB" "$GLM53_OFFLOAD_HOST_MB" "$GLM53_OFFLOAD_PLANNER" >> "$RUN_DIR/launch.txt"
fi
if [[ -n "$GLM53_TRITON_CACHE_DIR" ]]; then
    printf 'triton_cache_dir=%s\n' "$GLM53_TRITON_CACHE_DIR" >> "$RUN_DIR/launch.txt"
fi
if [[ -n "$GLM53_PROFILE_STEP" ]]; then
    printf 'profile_step=%s\nprofile_rank=0\nprofile_path=%s/profiler\n' \
        "$GLM53_PROFILE_STEP" "$RUN_DIR" >> "$RUN_DIR/launch.txt"
fi
git -C "$MS_ROOT" status --short > "$RUN_DIR/mindspeed_status.txt"
git -C "$MM_ROOT" status --short > "$RUN_DIR/mindspeed_mm_status.txt"
git -C "$MM_ROOT" diff --binary > "$RUN_DIR/mindspeed_mm_local.patch"
sha256sum "$EXAMPLE_ROOT/"*.py "$EXAMPLE_ROOT/"*.sh "$EXAMPLE_ROOT/"config.*.yaml "$EXAMPLE_ROOT/assets.sha256.json" \
    "$MM_ROOT/mindspeed_mm/fsdp/optimizer/swap_optimizer.py" > "$RUN_DIR/source.sha256"
cp "$EXAMPLE_ROOT/assets.sha256.json" "$RUN_DIR/assets.sha256.json"
printf 'GLM53_RUN_DIR=%s\n' "$RUN_DIR"

set +e
docker exec -e "TASK_QUEUE_ENABLE=$TASK_QUEUE_ENABLE" -e "GLM53_GROUPED_MOE_EP1=$GLM53_GROUPED_MOE_EP1" -e "GLM53_DISABLE_RECOMPUTE_LAYERS=$GLM53_DISABLE_RECOMPUTE_LAYERS" -e "GLM53_STOP_AFTER_ITERS=$GLM53_STOP_AFTER_ITERS" \
    -e "GLM53_OFFLOAD_DEVICE_MB=$GLM53_OFFLOAD_DEVICE_MB" -e "GLM53_OFFLOAD_HOST_MB=$GLM53_OFFLOAD_HOST_MB" -e "GLM53_OFFLOAD_PLANNER=$GLM53_OFFLOAD_PLANNER" \
    -e "GLM53_TRITON_CACHE_DIR=$GLM53_TRITON_CACHE_DIR" -e "GLM53_PROFILE_STEP=$GLM53_PROFILE_STEP" \
    mindspeed-glm53-flash-env bash -lc '
set -euo pipefail
ROOT=/data/disk1/machangwei
ENV_ROOT="$ROOT/glm53-env"
MS_ROOT="$ROOT/MindSpeed"
MM_ROOT="$ROOT/MindSpeed-MM"
EXAMPLE_ROOT="$8"
MODE="$1"
RUNNER_MODE="$2"
NPU_A="$3"
NPU_B="$4"
NODE="$5"
PORT="$6"
RUN_DIR="$7"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "$ENV_ROOT/venv/bin/activate"
export NON_MEGATRON=true
export ASCEND_RT_VISIBLE_DEVICES="$NPU_A,$NPU_B"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export ASCEND_LAUNCH_BLOCKING=0
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
if [[ -n "${GLM53_TRITON_CACHE_DIR:-}" ]]; then
    export TRITON_CACHE_DIR="$GLM53_TRITON_CACHE_DIR"
fi
export PYTHONPATH="$MS_ROOT:$MM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="$RUN_DIR/config.yaml"
PROFILE_ARGS=()
if [[ -n "${GLM53_PROFILE_STEP:-}" ]]; then
    PROFILE_ARGS=(--profile-step "$GLM53_PROFILE_STEP" --profile-save-path "$RUN_DIR/profiler")
    mkdir -p "$RUN_DIR/profiler"
fi
python "$EXAMPLE_ROOT/render_config.py" --mode "$MODE" --base-dir "$ENV_ROOT" --output "$CONFIG" \
    --disable-language-recompute "$GLM53_DISABLE_RECOMPUTE_LAYERS" "${PROFILE_ARGS[@]}"
python "$EXAMPLE_ROOT/validate_runtime.py" --mode "$MODE" --config "$CONFIG" --disable-language-recompute "$GLM53_DISABLE_RECOMPUTE_LAYERS"
cd "$MM_ROOT"
if [[ "$RUNNER_MODE" == hyperoffload ]]; then
    EXTRA=(--offload-device-mb "${GLM53_OFFLOAD_DEVICE_MB:-4096}" --offload-host-mb "${GLM53_OFFLOAD_HOST_MB:-32768}" --planner "${GLM53_OFFLOAD_PLANNER:-peak}")
else
    EXTRA=()
fi
python "$EXAMPLE_ROOT/numa_exec.py" "$NODE" -- torchrun \
    --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" \
    "$EXAMPLE_ROOT/run_glm53.py" \
    --mode "$RUNNER_MODE" --config "$CONFIG" --metrics-dir "$RUN_DIR/metrics" \
    --stop-after-iters "$GLM53_STOP_AFTER_ITERS" "${EXTRA[@]}"
' _ "$MODE" "$RUNNER_MODE" "$NPU_A" "$NPU_B" "$NODE_A" "$PORT" "$RUN_DIR" "$EXAMPLE_ROOT" 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
printf 'exit_code=%s\nfinished=%s\n' "$rc" "$(date -Is)" > "$RUN_DIR/exit.txt"
exit "$rc"
