# Qwen2.5-32B FSDP2 A2 Reproduction

This directory contains the runner and one-file host setup script for the
two-NPU, 10-step baseline/HyperOffload comparison measured in the original
Ascend 910B workspace.

## Run

On the A2 host, clone branch `1029` and run the deployment script:

```bash
git clone --depth=1 --branch 1029 https://github.com/Mcw-mystory/MindSpeed.git /tmp/MindSpeed-1029
bash /tmp/MindSpeed-1029/examples/qwen25_32b_fsdp2_a2/deploy_and_reproduce.sh all
```

The script creates `/home/m00951751/qwen25-32b-fsdp2`, extracts the FSDP2
runtime from the pinned Ascend image into `runtime/fsdp`, checks or creates the
Ascend container, downloads the Qwen2.5-32B safetensors, selects two healthy
idle NPUs, and runs the cases sequentially. Set `NPU_DEVICES=0,1` to choose a
specific pair. Set `HF_TOKEN` if the Hugging Face endpoint requires
authentication. `MINDSPEED_IMAGE` can override the default image when the A2
host has a different compatible Ascend driver/CANN release.

Modes are `setup`, `download`, `baseline`, `hyperoffload`, and `all`. The
default is `all`. Use a new `RUN_ID` to preserve existing output directories.

## Fixed Experiment

- Model: Qwen2.5-32B, first 48 transformer layers, BF16, sequence length 4096.
- Parallelism: FSDP2 across exactly two NPU workers; the container sees the
  host's available NPUs.
- Baseline: 34 recompute layers, HyperOffload disabled.
- HyperOffload: zero recompute layers, Peak planner, 2048 MiB device budget
  and 131072 MiB pinned Host budget per rank.
- Each case: 10 training steps; profiler captures steps 5 and 6.
- Optimizer: SGD, learning rate 0.01. Inputs are deterministic random token IDs
  per rank, matching the original benchmark harness rather than a dataset run.

The 32B weights occupy about 62 GB. Keep at least 80 GB free under
`/home/m00951751`. The HyperOffload case allocated about 88.3 GiB of pinned
Host memory per rank in the measured run; run it on a host with sufficient free
RAM and an unrestricted memlock limit.

Results are written to `logs/` and `profile/` below the experiment root. The
script verifies both ranks reached 10 steps and recorded nonzero first-step
gradients and parameter updates. Hardware-specific timings and memory usage can
differ from the original machine.

## MindSpeed Changes In This Branch

- Pinned-memory pool operations and activation D2H/H2D/release operations run
  outside the activation dispatch mode. This prevents internal buffer
  allocations and copies from recursively entering the activation tracker.
- Warmup now recycles activation buffers whose wrappers are no longer live
  before enforcing the device residency budget. This releases stale Host-pool
  allocations during trace capture.
- The HyperOffload branch does not include `mindspeed/fsdp/`, while the measured
  FSDP2 runner depends on it. The setup script extracts that package from the
  selected `mindspeed-core:v26.1.0` image and mounts it read-only under
  `/MindSpeed/mindspeed/fsdp`; the image package is not copied into this branch.
- Unit tests cover dispatch isolation and warmup reclamation behavior.
