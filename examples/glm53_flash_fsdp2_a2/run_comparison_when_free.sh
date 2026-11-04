#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/disk1/machangwei
ENV_ROOT="$ROOT/glm53-env"
EXAMPLE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export GLM53_DISABLE_RECOMPUTE_LAYERS=${GLM53_DISABLE_RECOMPUTE_LAYERS:-0,3}
export GLM53_STOP_AFTER_ITERS=${GLM53_STOP_AFTER_ITERS:-10}
RESERVATION=/data/disk/liangzhibo/HyperOffload-research/generalized-tuning-20261102/reservation.json
QWEN_LOCK="$ROOT/logs/qwen25-r28-r31-continuation.lock"
LOG="$ENV_ROOT/logs/1103_glm53_comparison_wait.log"
STATE="$ENV_ROOT/logs/1103_glm53_comparison.state"
LOCK="$ENV_ROOT/logs/1103_glm53_comparison.lock"
STOP="$ENV_ROOT/logs/stop-1103-glm53-comparison"
mkdir -p "$ENV_ROOT/logs"
exec 9>"$LOCK"
flock -n 9 || { echo "GLM-5.3 comparison monitor is already running" >&2; exit 1; }

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
state() { printf '%s %s\n' "$(date -Is)" "$*" > "$STATE"; log "$*"; }

numa_node() {
    case "$1" in
        0) bdf=0000:c1:00.0 ;; 1) bdf=0000:c2:00.0 ;;
        2) bdf=0000:81:00.0 ;; 3) bdf=0000:82:00.0 ;;
        4) bdf=0000:01:00.0 ;; 5) bdf=0000:02:00.0 ;;
        6) bdf=0000:41:00.0 ;; 7) bdf=0000:42:00.0 ;;
        *) return 1 ;;
    esac
    cat "/sys/bus/pci/devices/$bdf/numa_node"
}

npu_available() {
    local npu=$1 health processes hbm
    health=$(npu-smi info -t health -i "$npu" | awk -F: '/Health[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    [[ "$health" == OK ]] || return 1
    processes=$(npu-smi info -t proc-mem -i "$npu")
    ! grep -Eq 'Process id:[[:space:]]*[0-9]+' <<< "$processes" || return 1
    hbm=$(npu-smi info -t usages -i "$npu" | awk -F: '/HBM Usage Rate/ {gsub(/[^0-9]/, "", $2); print $2; exit}')
    [[ "$hbm" =~ ^[0-9]+$ && "$hbm" -le 10 ]]
}

pair_available() {
    local a=$1 b=$2 node_a node_b free_kib minimum_free_mib
    node_a=$(numa_node "$a") || return 1
    node_b=$(numa_node "$b") || return 1
    [[ "$node_a" -ge 0 && "$node_a" == "$node_b" ]] || return 1
    minimum_free_mib=${GLM53_MIN_NUMA_FREE_MIB:-163840}
    free_kib=$(awk '/MemFree:/ {print $4; exit}' "/sys/devices/system/node/node$node_a/meminfo")
    [[ "$minimum_free_mib" =~ ^[0-9]+$ && "$free_kib" =~ ^[0-9]+$ ]] || return 1
    (( free_kib / 1024 >= minimum_free_mib )) || return 1
    npu_available "$a" && npu_available "$b" || return 1
    if [[ "$a$b" == 67 || "$a$b" == 76 ]]; then
        local status
        status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "$RESERVATION" 2>/dev/null || true)
        [[ "$status" == released ]] || return 1
        flock -n "$QWEN_LOCK" -c true >/dev/null 2>&1 || return 1
    fi
}

find_pair() {
    local pair a b
    for pair in 0,1 2,3 4,5 6,7; do
        IFS=, read -r a b <<< "$pair"
        if pair_available "$a" "$b"; then printf '%s\n' "$pair"; return 0; fi
    done
    return 1
}

run_mode() {
    local mode=$1 pair=$2 log_path=$3 rc=0 run_dir
    bash "$EXAMPLE_ROOT/launch_glm53_pair.sh" "$mode" "$pair" > "$log_path" 2>&1 || rc=$?
    cat "$log_path" >> "$LOG"
    run_dir=$(sed -n 's/^GLM53_RUN_DIR=//p' "$log_path" | tail -1)
    printf '%s\n' "$run_dir"
    return "$rc"
}

wait_same_pair() {
    local pair=$1 a b
    IFS=, read -r a b <<< "$pair"
    while true; do
        [[ -e "$STOP" ]] && return 2
        if pair_available "$a" "$b"; then
            log "NPU pair $pair is idle again; requiring 60-second stability before next case"
            sleep 60
            pair_available "$a" "$b" && return 0
        fi
        sleep 60
    done
}

state "waiting for a healthy, idle NUMA-local pair; run strict baseline, then the accepted two-layer HyperOffload case"
last_wait_log=0
while true; do
    [[ -e "$STOP" ]] && { state "stopped by $STOP before launch"; exit 0; }
    docker inspect -f '{{.State.Running}}' mindspeed-glm53-flash-env 2>/dev/null | grep -qx true || {
        sleep 60
        continue
    }
    if pgrep -af '[t]orchrun.*(run_glm53.py|mindspeed_mm/fsdp/train/trainer.py)|[t]orch.distributed.run.*(run_glm53.py|mindspeed_mm/fsdp/train/trainer.py)' >/dev/null; then
        state "another GLM/MindSpeed-MM process is active; polling without interfering"
        sleep 60
        continue
    fi
    pair=$(find_pair || true)
    if [[ -z "$pair" ]]; then
        now=$(date +%s)
        if (( now - last_wait_log >= 300 )); then log "no eligible pair; next check in 60 seconds"; last_wait_log=$now; fi
        sleep 60
        continue
    fi
    log "candidate pair $pair is idle; checking 60-second stability"
    sleep 60
    IFS=, read -r a b <<< "$pair"
    pair_available "$a" "$b" || { log "pair $pair became busy; retrying"; continue; }
    break
done

comparison_id="$(date +%Y%m%d_%H%M%S)_npu${pair/,/}"
comparison_dir="$ENV_ROOT/experiments/1103_glm53/comparison_$comparison_id"
mkdir -p "$comparison_dir"
state "starting strict full-recompute baseline on NPU pair $pair; output=$comparison_dir"
baseline_dir=""
baseline_rc=0
run_mode strict_baseline "$pair" "$comparison_dir/strict_baseline.launch.log" > "$comparison_dir/strict_baseline.run_dir" || baseline_rc=$?
baseline_dir=$(cat "$comparison_dir/strict_baseline.run_dir")
state "strict baseline exit=$baseline_rc run_dir=${baseline_dir:-unknown}"

baseline_peak_bytes=$(python3 -c '
import json
import pathlib
import sys

metrics_dir = pathlib.Path(sys.argv[1])
values = []
for metrics_file in metrics_dir.glob("rank*.jsonl"):
    for line in metrics_file.read_text(encoding="utf-8").splitlines():
        values.append(json.loads(line)["npu_peak_allocated_bytes"])
print(max(values, default=0))
' "$baseline_dir/metrics")
baseline_peak_mib=$(( baseline_peak_bytes / 1048576 ))
minimum_peak_mib=${GLM53_MIN_BASELINE_ALLOCATED_MIB:-58000}
state "strict actual peak allocated=${baseline_peak_bytes} bytes (${baseline_peak_mib} MiB); HyperOffload gate=${minimum_peak_mib} MiB"
if (( baseline_rc != 0 )); then
    state "strict baseline failed; skipping HyperOffload case"
    exit "$baseline_rc"
fi
if (( baseline_peak_mib < minimum_peak_mib )); then
    state "strict baseline peak is below the allocated-memory target; skipping HyperOffload case"
    exit 3
fi

if ! wait_same_pair "$pair"; then state "stopped while waiting for NPU pair $pair before HyperOffload case"; exit 2; fi
state "starting the accepted two-layer partial-recompute HyperOffload case on the same NPU pair $pair"
hyper_rc=0
run_mode hyperoffload "$pair" "$comparison_dir/hyperoffload.launch.log" > "$comparison_dir/hyperoffload.run_dir" || hyper_rc=$?
hyper_dir=$(cat "$comparison_dir/hyperoffload.run_dir")
summary_rc=0
python3 "$EXAMPLE_ROOT/summarize_comparison.py" \
    --baseline-dir "$baseline_dir/metrics" \
    --hyperoffload-dir "$hyper_dir/metrics" \
    --output "$comparison_dir/summary.json" > "$comparison_dir/summary.stdout.log" 2>&1 || summary_rc=$?
cat "$comparison_dir/summary.stdout.log" >> "$LOG"
state "comparison finished; pair=$pair strict_baseline_exit=$baseline_rc hyperoffload_exit=$hyper_rc summary_exit=$summary_rc summary=$comparison_dir/summary.json"
exit "$(( baseline_rc != 0 || hyper_rc != 0 || summary_rc != 0 ))"
