#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../../.." && pwd)"
container_script=/MindSpeed/examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_in_container.sh
workspace="${QWEN_WORKSPACE:-/data/disk1/machangwei}"
container="${MINDSPEED_CONTAINER_NAME:-mindspeed-qwen25-32b-fsdp}"
npu_devices="${NPU_DEVICES:-4,5}"
numa_node="${NUMA_NODE:-0}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

[[ "$npu_devices" =~ ^[0-7],[0-7]$ ]] || { echo "NPU_DEVICES must be two IDs, got $npu_devices" >&2; exit 2; }
[[ "$numa_node" =~ ^[0-7]$ ]] || { echo "NUMA_NODE must be in 0..7" >&2; exit 2; }
[[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]] || {
    echo "container is not running: $container" >&2
    exit 1
}
[[ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$container")" == host ]] || {
    echo "container must use host networking" >&2
    exit 1
}
docker inspect "$container" --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}' |
    grep -Fq "$repo -> /MindSpeed" || { echo "container must mount this checkout at /MindSpeed" >&2; exit 1; }
docker inspect "$container" --format '{{range .Mounts}}{{println .Destination}}{{end}}' |
    grep -Fxq /etc/hccn.conf || { echo "container must mount /etc/hccn.conf" >&2; exit 1; }
docker exec "$container" test -r "$container_script" || {
    echo "container cannot read launcher: $container_script" >&2
    exit 1
}

IFS=, read -r npu_a npu_b <<< "$npu_devices"
[[ "$npu_a" != "$npu_b" ]] || { echo "NPU_DEVICES must be distinct" >&2; exit 2; }
for npu in "$npu_a" "$npu_b"; do
    process_info="$(npu-smi info -t proc-mem -i "$npu")"
    grep -q 'No process in device\.' <<< "$process_info" || {
        echo "NPU $npu is occupied" >&2
        printf '%s\n' "$process_info" >&2
        exit 1
    }
    usage="$(npu-smi info -t usages -i "$npu" | awk -F: '/HBM Usage Rate/{gsub(/[^0-9]/, "", $2); print $2; exit}')"
    health="$(npu-smi info -t health -i "$npu" | awk -F: '/Health/{gsub(/[[:space:]]/, "", $2); print $2; exit}')"
    [[ "$usage" =~ ^[0-9]+$ && "$usage" -le 10 && "$health" == OK ]] || {
        echo "NPU $npu is not healthy and idle: health=${health:-unknown} HBM=${usage:-unknown}%" >&2
        exit 1
    }
done

inside="$(docker exec "$container" bash -lc 'pgrep -af "[t]orchrun.*run_qwen25_native_mindspeed_fsdp2.py" || true')"
[[ -z "$inside" ]] || { echo "native FSDP2 torchrun already active in container" >&2; exit 1; }

docker exec -i \
    -e QWEN_WORKSPACE="$workspace" \
    -e MINDSPEED_REPO=/MindSpeed \
    -e NPU_DEVICES="$npu_devices" \
    -e NUMA_NODE="$numa_node" \
    -e RUN_ID="$run_id" \
    "$container" bash "$container_script" "$run_id" "$npu_devices" "$numa_node"
