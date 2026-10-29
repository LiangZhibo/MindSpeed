# Qwen2.5-32B FSDP2 A2 Reproduction

This directory runs a two-NPU, 10-step baseline/HyperOffload comparison and
provides a separate numerical correctness workflow for the same large model.

## Run

Run the script from the checkout containing these files. It mounts that checkout
at `/MindSpeed`, including the validation code; it does not clone a different
branch into the experiment directory.

```bash
NPU_DEVICES=4,5 bash examples/qwen25_32b_fsdp2_a2/deploy_and_reproduce.sh all
```

The script creates `/home/m00951751/qwen25-32b-fsdp2` for models and results,
extracts the FSDP2 runtime from the pinned Ascend image into `runtime/fsdp`,
checks or creates the container, downloads the weights, and runs both cases.
`BASE_DIR` overrides the experiment directory. `MINDSPEED_DIR` can select another
checkout; it must contain all three Python files in this directory.
Use a fresh `RUN_ID` to preserve existing outputs.

Set `NPU_DEVICES` to a healthy idle pair belonging to the **same NUMA node**.
Configure the container's CPU affinity and host-memory node to match that pair
before running. The script's automatic device selection checks health and
occupancy but does not establish NUMA locality or reserve devices against other
jobs. `MINDSPEED_CONTAINER_NAME` selects an existing configured container.
`MINDSPEED_IMAGE` overrides the default image for a compatible driver/CANN release.
Set `HF_TOKEN` if the model endpoint requires authentication.

Modes: `setup`, `download`, `baseline`, `hyperoffload`, `correctness`, and `all`.
The default `all` runs the two performance cases. Numerical comparison is explicit
and runs separately because reference storage, full tensor scans and comparisons
change execution time and memory usage.

## Fixed experiment

- Qwen2.5-32B pretrained weights, first 48 transformer layers (24,962,192,384
  parameters), BF16, sequence length 4096, batch 1 per rank.
- FSDP2 across exactly two NPU workers.
- Baseline: 34 recompute layers, HyperOffload disabled.
- HyperOffload: zero recompute layers, Peak planner, 2048 MiB device budget
  and 131072 MiB pinned host budget per rank.
- SGD, learning rate 0.01, deterministic random inputs per rank.
- Each case has 10 training steps; performance runs profile steps 5 and 6.

The pretrained weights occupy about 62 GB. Keep at least 80 GB free for the model.
A previous HyperOffload run allocated about 88.3 GiB of pinned host-pool memory
per rank. This includes reusable pool buffers and is not a measure of live
activations or a theoretical minimum. Provide sufficient RAM on the selected NUMA
node and an unrestricted memlock limit.

## Checks on every run

The runner checks finite loss and finite, positive sampled gradient/update maxima
on **every step on both ranks**. These probes cover the first 4096 local elements
of the first layer's MLP down-projection weight. They detect obvious failures but
are not full numerical correctness validation. Validation failure is propagated
across ranks and terminates the job with a nonzero exit status.

The exact loaded `chunk_loss` implementation also runs an independent CPU FP32
cross-entropy self-check, including a partial final chunk and a non-unit upstream
gradient. This is a small loss-function test, not a full-model reference.

The host verifier rejects incomplete or duplicate step records, NaN/Inf,
zero/negative probe maxima, and missing or failed requested numerical comparisons.
It reports basic checks separately from reference capture/comparison.

## Large-model numerical correctness

```bash
NPU_DEVICES=4,5 RUN_ID=correctness-001 \
  bash examples/qwen25_32b_fsdp2_a2/deploy_and_reproduce.sh correctness
```

This runs the **same 48-layer model and 4096-token workload** in two fresh jobs:

1. Baseline with 34 recompute layers: record the reference.
2. HyperOffload with zero recompute layers: compare against that reference.

Both jobs disable profiling. The validator requires identical initial parameter
and input hashes, seeds, workload settings, optimizer and library versions. It
compares loss at all 10 steps and scans every trainable gradient and parameter for
NaN/Inf at every step. By default, it additionally saves/compares **all** local
FSDP gradient shards and **all** updated parameter shards at steps **1, 2 and 10**.
The two ranks together cover the full model. This covers warmup and replay; no
unselected parameter is substituted with a sampled probe.

The full reference requires roughly **279 GiB** of tensor files at BF16 for these
three snapshot steps, in addition to weights and logs. A 10% disk-space margin is
checked before recording. `CORRECTNESS_STEPS=1,10` reduces storage, at the cost of
omitting the first replay-step snapshot. CPU copying/comparison processes one
parameter at a time and compares in bounded chunks; it still needs extra CPU
memory beyond the offload pool.

For manual or resumed comparisons, use the runner directly, with matching total
step counts, `--no-profile`, and a fresh reference directory for recording:

```bash
# Add to the usual torchrun command and model/offload arguments:
--correctness-mode record --reference-dir /shared/reference --no-profile
# In the second job:
--correctness-mode compare --reference-dir /shared/reference --no-profile
```

References are rank-specific and marked complete only after all requested checks
are recorded. Partial references, different weights/inputs/settings, missing
snapshots, non-finite values or out-of-tolerance results cause failure.

A value passes when `abs(actual-reference) <= atol + rtol * abs(reference)`:

| Quantity | Default rtol | Default atol |
|---|---:|---:|
| Loss | 1e-3 | 1e-3 |
| Full gradients | 2e-2 | 1e-5 |
| Updated parameters | 1e-2 | 1e-5 |

The direct runner exposes `--loss-rtol/--loss-atol`,
`--gradient-rtol/--gradient-atol`, and `--parameter-rtol/--parameter-atol`.
Tolerances must be finite and nonnegative and are saved with the results. A failed
comparison remains a failure; there is no automatic tolerance relaxation.

Results include `correctness.mode`, `correctness.status`, `correctness.passed`,
selected steps, tolerances, checked element counts and maximum absolute differences.
A successful reference recording is not itself a successful comparison. The
baseline and candidate share the model/loss implementation, so agreement measures
HyperOffload/recompute consistency and does not establish convergence or eliminate
shared model/loss bugs. The separate loss self-check is also limited to its inputs.

```bash
python3 examples/qwen25_32b_fsdp2_a2/verify_results.py /path/to/candidate-results \
  --steps 10 --correctness-mode compare
```

## Runtime dependency

The branch does not include `mindspeed/fsdp/`. Setup extracts it from the selected
Ascend image and mounts it read-only. The HyperOffload runtime includes dispatch
isolation for internal transfers/pool allocations and warmup activation reclamation.

## CPU regression tests

```bash
python -m pytest tests_extend/unit_tests/examples/test_qwen_correctness.py -q
```

These tests exercise full gradient/parameter reference capture and comparison,
corruption beyond the old 4096-element probe, partial/mismatched references,
non-finite values on unsnapshotted steps, loss-function checking and host CLI
failure status. They do not replace a full two-NPU run of the large model.
