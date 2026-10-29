#!/usr/bin/env bash
set -euo pipefail

# Reproduce the measured two-NPU, 10-step Qwen2.5-32B experiment on A2.
# Default mode: r34 baseline, then r0 HyperOffload, with profile on steps 5/6.

BASE_DIR="${BASE_DIR:-/home/m00951751/qwen25-32b-fsdp2}"
MODEL_DIR="$BASE_DIR/Qwen2.5-32B"
MINDSPEED_DIR="${MINDSPEED_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FSDP_DIR="$BASE_DIR/runtime/fsdp"
EXPERIMENT_DIR="$MINDSPEED_DIR/examples/qwen25_32b_fsdp2_a2"
LOG_DIR="$BASE_DIR/logs"
PROFILE_DIR="$BASE_DIR/profile"
USER_HOME="$BASE_DIR/.home"
CONTAINER_NAME="${MINDSPEED_CONTAINER_NAME:-mindspeed-qwen25-32b-fsdp2-a2}"
IMAGE="${MINDSPEED_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-core:v26.1.0_core_r0.12.1-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12}"
BRANCH=
NPU_DEVICES="${NPU_DEVICES:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MASTER_PORT="${MASTER_PORT:-29621}"
MODE="${1:-all}"
RUN_UID="$(id -u)"
RUN_GID="$(id -g)"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

for cmd in docker git python3 npu-smi; do need "$cmd"; done
mkdir -p "$BASE_DIR" "$LOG_DIR" "$PROFILE_DIR" "$USER_HOME"

prepare_source() {
    git -C "$MINDSPEED_DIR" rev-parse --show-toplevel >/dev/null 2>&1 ||
        die "$MINDSPEED_DIR is not a git checkout"
    BRANCH="$(git -C "$MINDSPEED_DIR" branch --show-current)"
    for file in run_qwen25_mindspeed_fsdp2.py correctness.py verify_results.py; do
        [[ -f "$EXPERIMENT_DIR/$file" ]] || die "checkout is missing $file"
    done
    printf 'MindSpeed branch=%s commit=%s\n' "${BRANCH:-detached}" "$(git -C "$MINDSPEED_DIR" rev-parse HEAD)"
}

prepare_fsdp() {
    local image_id
    image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
    if [[ -e "$FSDP_DIR" ]]; then
        [[ -f "$FSDP_DIR/mindspeed_parallel_engine.py" && -f "$FSDP_DIR/.source-image-id" ]] ||
            die "incomplete FSDP runtime at $FSDP_DIR"
        [[ "$(cat "$FSDP_DIR/.source-image-id")" == "$image_id" ]] ||
            die "FSDP runtime came from a different image: $FSDP_DIR"
        return
    fi
    mkdir -p "$BASE_DIR/runtime"
    docker run --rm --user "$RUN_UID:$RUN_GID" --entrypoint bash \
        -v "$BASE_DIR/runtime:/export:rw" "$IMAGE" -c '
            set -euo pipefail
            test -f /MindSpeed/mindspeed/fsdp/mindspeed_parallel_engine.py
            cp -R /MindSpeed/mindspeed/fsdp /export/fsdp
        '
    [[ -f "$FSDP_DIR/mindspeed_parallel_engine.py" ]] || die "FSDP extraction failed"
    printf '%s\n' "$image_id" > "$FSDP_DIR/.source-image-id"
    printf 'FSDP runtime extracted from image: %s\n' "$IMAGE"
}

prepare_container() {
    [[ -e /usr/local/Ascend/driver ]] || die "Ascend driver not found at /usr/local/Ascend/driver"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
    prepare_fsdp
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        local mounts container_image_id
        mounts="$(docker inspect --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' "$CONTAINER_NAME")"
        container_image_id="$(docker inspect --format '{{.Image}}' "$CONTAINER_NAME")"
        [[ "$container_image_id" == "$(docker image inspect --format '{{.Id}}' "$IMAGE")" ]] || die "existing container uses a different image"
        grep -Fq "$BASE_DIR -> $BASE_DIR" <<<"$mounts" || die "existing container does not mount $BASE_DIR"
        grep -Fq "$MINDSPEED_DIR -> /MindSpeed" <<<"$mounts" || die "existing container does not mount $MINDSPEED_DIR at /MindSpeed"
        grep -Fq "$FSDP_DIR -> /MindSpeed/mindspeed/fsdp" <<<"$mounts" || die "existing container does not mount the image FSDP runtime"
        docker start "$CONTAINER_NAME" >/dev/null
    else
        local mounts=()
        add_mount() {
            local source="$1" mode="${2:-rw}"
            [[ -e "$source" ]] && mounts+=(-v "$source:$source:$mode")
        }
        add_mount /usr/local/Ascend/driver
        add_mount /usr/local/dcmi
        add_mount /usr/local/bin/npu-smi ro
        add_mount /usr/local/sbin/npu-smi ro
        add_mount /etc/ascend_install.info ro
        add_mount /etc/hccn.conf ro
        add_mount /usr/bin/hccn_tool ro
        add_mount /var/log/npu
        add_mount /usr/src/kernels ro
        docker run -d --name "$CONTAINER_NAME" --runtime=ascend --privileged \
            --network=host --ipc=host --shm-size=64g --ulimit memlock=-1 \
            --ulimit stack=67108864 -e "HOME=$USER_HOME" \
            -e PATH=/usr/local/python3.12.13/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/bin:/usr/bin:/bin \
            "${mounts[@]}" -v "$BASE_DIR:$BASE_DIR" -v "$MINDSPEED_DIR:/MindSpeed" \
            -v "$FSDP_DIR:/MindSpeed/mindspeed/fsdp:ro" \
            -w "$BASE_DIR" "$IMAGE" bash -lc 'sleep infinity' >/dev/null
    fi

    docker exec "$CONTAINER_NAME" bash -lc '
        source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
        set +u
        source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
        set -u
        export PATH=/usr/local/python3.12.13/bin:$PATH
        export PYTHONPATH=/MindSpeed:${PYTHONPATH:-}
        python -c "import torch, torch_npu, transformers, huggingface_hub; print(\"torch=\", torch.__version__, \"torch_npu=\", torch_npu.__version__, \"transformers=\", transformers.__version__, \"npu_count=\", torch.npu.device_count())"
    '
    if ! docker exec "$CONTAINER_NAME" bash -lc 'python -c "import accelerate"' >/dev/null 2>&1; then
        docker exec "$CONTAINER_NAME" bash -lc 'python -m pip install "accelerate==1.11.0"'
    fi
    docker exec "$CONTAINER_NAME" bash -lc 'PYTHONPATH=/MindSpeed python -c "from mindspeed.fsdp.mindspeed_parallel_engine import MindSpeedParallelEngine; from mindspeed.core.memory.hyper_offload import OffloadSession; print(\"FSDP2 and HyperOffload imports OK\")"'
}

weights_complete() {
    python3 - "$MODEL_DIR" <<'PY'
import json, os, sys
root = sys.argv[1]
try:
    with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
        names = set(json.load(f)["weight_map"].values())
except (OSError, KeyError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if names and all(os.path.isfile(os.path.join(root, n)) for n in names) else 1)
PY
}

download_model() {
    if weights_complete; then printf 'Model weights complete: %s\n' "$MODEL_DIR"; return; fi
    local free_kb
    free_kb="$(df -Pk "$BASE_DIR" | awk 'NR==2 {print $4}')"
    [[ "$free_kb" =~ ^[0-9]+$ ]] && ((free_kb >= 80000000)) || die "need at least 80 GB free under $BASE_DIR"
    local token_args=()
    [[ -n "${HF_TOKEN:-}" ]] && token_args+=(-e "HF_TOKEN=$HF_TOKEN")
    docker exec -u "$RUN_UID:$RUN_GID" -e "HOME=$USER_HOME" "${token_args[@]}" -i "$CONTAINER_NAME" bash -s -- "$MODEL_DIR" <<'DOWNLOAD'
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export PATH=/usr/local/python3.12.13/bin:$PATH
python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-32B", local_dir=sys.argv[1],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"], max_workers=8), flush=True)
PY
DOWNLOAD
    weights_complete || die "download ended before all model shards arrived"
}

choose_devices() {
    local device process_info usage health
    local candidates=()
    if [[ -z "$NPU_DEVICES" ]]; then
        for device in 0 1 2 3 4 5 6 7; do
            process_info="$(npu-smi info -t proc-mem -i "$device" 2>/dev/null || true)"
            grep -q 'No process in device\.' <<<"$process_info" || continue
            usage="$(npu-smi info -t usages -i "$device" 2>/dev/null | awk -F: '/HBM Usage Rate/{gsub(/[^0-9]/, "", $2); print $2; exit}')"
            health="$(npu-smi info -t health -i "$device" 2>/dev/null | awk -F: '/Health/{gsub(/[[:space:]]/, "", $2); print $2; exit}')"
            [[ "$usage" =~ ^[0-9]+$ && "$usage" -le 10 && "$health" == OK ]] && candidates+=("$device")
        done
        ((${#candidates[@]} >= 2)) || die "fewer than two healthy idle NPUs; set NPU_DEVICES after checking npu-smi"
        NPU_DEVICES="${candidates[0]},${candidates[1]}"
    fi
    [[ "$NPU_DEVICES" =~ ^[0-7],[0-7]$ ]] || die "NPU_DEVICES must be two IDs in 0..7, e.g. 0,1"
    local ids=("${NPU_DEVICES%,*}" "${NPU_DEVICES#*,}")
    [[ "${ids[0]}" != "${ids[1]}" ]] || die "NPU_DEVICES must name two distinct devices"
    for device in "${ids[@]}"; do
        process_info="$(npu-smi info -t proc-mem -i "$device" 2>/dev/null || true)"
        grep -q 'No process in device\.' <<<"$process_info" || die "NPU $device is occupied: $process_info"
        usage="$(npu-smi info -t usages -i "$device" 2>/dev/null | awk -F: '/HBM Usage Rate/{gsub(/[^0-9]/, "", $2); print $2; exit}')"
        health="$(npu-smi info -t health -i "$device" 2>/dev/null | awk -F: '/Health/{gsub(/[[:space:]]/, "", $2); print $2; exit}')"
        [[ "$usage" =~ ^[0-9]+$ && "$usage" -le 10 ]] || die "NPU $device HBM usage is ${usage:-unknown}%"
        [[ "$health" == OK ]] || die "NPU $device health is ${health:-unknown}"
    done
    printf 'using_npus=%s\n' "$NPU_DEVICES"
}

run_case() {
    local label="$1" recompute="$2" offload="$3" correctness_mode="${4:-none}" reference_dir="${5:-}"
    local name="qwen25-32b-fsdp2-${label}-${RUN_ID}"
    local out="$PROFILE_DIR/$name" log="$LOG_DIR/$name.log"
    [[ ! -e "$out" && ! -e "$log" ]] || die "output exists: $name; choose another RUN_ID"
    pgrep -af '[t]orchrun.*run_qwen25_mindspeed_fsdp2.py' >/dev/null 2>&1 && die "Qwen torchrun already runs on host"
    local inside
    inside="$(docker exec "$CONTAINER_NAME" bash -lc 'pgrep -af "[t]orchrun.*run_qwen25_mindspeed_fsdp2.py" || true')"
    [[ -z "$inside" ]] || die "Qwen torchrun already runs in container: $inside"
    choose_devices
    mkdir -p "$out" "$LOG_DIR"
    printf 'started_at=%s\ncase=%s\nbranch=%s\nnpus=%s\nrecompute_layers=%s\nhyper_offload=%s\nsteps=10\ncorrectness_mode=%s\nprofile_steps=%s\n' \
        "$(date -Is)" "$name" "$BRANCH" "$NPU_DEVICES" "$recompute" "$offload" "$correctness_mode" "$([[ "$correctness_mode" == none ]] && printf 5,6 || printf none)" > "$out/launch.txt"
    local offload_args=()
    if [[ "$correctness_mode" != none ]]; then
        offload_args+=(--correctness-mode "$correctness_mode" --reference-dir "$reference_dir"
                       --correctness-steps "${CORRECTNESS_STEPS:-1,2,10}" --no-profile)
    fi
    [[ "$offload" == 1 ]] && offload_args+=(--hyper-offload --offload-device-mb 2048 --offload-host-mb 131072)
    set +e
    docker exec -u "$RUN_UID:$RUN_GID" -e "HOME=$USER_HOME" -i "$CONTAINER_NAME" \
        bash -s -- "$MODEL_DIR" "$NPU_DEVICES" "$recompute" "$out" "$MASTER_PORT" "${offload_args[@]}" <<'TRAIN' 2>&1 | tee "$log"
set -euo pipefail
MODEL_DIR="$1"; NPU_DEVICES="$2"; RECOMPUTE="$3"; PROFILE_PATH="$4"; MASTER_PORT="$5"
shift 5
OFFLOAD_ARGS=("$@")
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH=/MindSpeed:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:2048 TOKENIZERS_PARALLELISM=false
torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port="$MASTER_PORT" \
  /MindSpeed/examples/qwen25_32b_fsdp2_a2/run_qwen25_mindspeed_fsdp2.py \
  --model-path "$MODEL_DIR" --profile-path "$PROFILE_PATH" \
  --recompute-layers "$RECOMPUTE" --extra-steps 9 "${OFFLOAD_ARGS[@]}"
TRAIN
    local status=$?
    set -e
    printf 'finished_at=%s\ntraining_exit_code=%s\n' "$(date -Is)" "$status" > "$out/exit.txt"
    if ((status != 0)); then
        printf 'exit_code=%s\n' "$status" >> "$out/exit.txt"
        die "$name failed; inspect $log and $out"
    fi
    if python3 "$EXPERIMENT_DIR/verify_results.py" "$out" --steps 10 --correctness-mode "$correctness_mode"; then
        status=0
    else
        status=$?
    fi
    printf 'verification_exit_code=%s\nexit_code=%s\n' "$status" "$status" >> "$out/exit.txt"
    ((status == 0)) || die "$name validation failed; inspect $out"

    printf 'completed=%s\nlog=%s\nprofile=%s\n' "$name" "$log" "$out"
}

case "$MODE" in
    setup)
        prepare_source
        prepare_container
        printf 'setup ready: %s\n' "$BASE_DIR"
        ;;
    download)
        prepare_source
        prepare_container
        download_model
        ;;
    baseline)
        prepare_source
        prepare_container
        download_model
        run_case baseline-r34 34 0
        ;;
    hyperoffload)
        prepare_source
        prepare_container
        download_model
        run_case hyperoffload-r0 0 1
        ;;
    correctness)
        prepare_source
        prepare_container
        download_model
        reference_dir="$PROFILE_DIR/qwen25-32b-reference-$RUN_ID"
        [[ ! -e "$reference_dir" ]] || die "reference exists: $reference_dir; choose another RUN_ID"
        run_case baseline-correctness-r34 34 0 record "$reference_dir"
        run_case hyperoffload-correctness-r0 0 1 compare "$reference_dir"
        ;;
    all)
        prepare_source
        prepare_container
        download_model
        run_case baseline-r34 34 0
        run_case hyperoffload-r0 0 1
        ;;
    *) die "usage: $0 {setup|download|baseline|hyperoffload|correctness|all}" ;;
esac
