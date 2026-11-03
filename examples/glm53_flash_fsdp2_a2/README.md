# GLM-5.3-Flash A2 FSDP2 baseline and HyperOffload test

## Fixed model and baseline

The accepted baseline is the proportional **7L/64E/top-8/5V** GLM proxy (12,437,456,762 parameters), with all seven language layers and all five vision blocks recomputed. It runs MindSpeed-MM FSDP2 with shard size 2, BF16, microbatch 1, 4096 token cutoff, AdamWSwap, and the established 64-sample synthetic image/text dataset. MindSpeed-MM activation stash, op replay, and HyperOffload are disabled. It starts from random initialization and has no accuracy or useful-training claim.

The proxy preserves GLM widths, layer/module types, KDA/DSA operators, MoE routing, top-k, and vision-block structure; only the number of layers, experts, and vision blocks is reduced proportionally. Do not edit the widths or substitute the older 8L/56E/5V stash-enabled preset.

The frozen measured reference is `results/baseline_reference.json`, from `20261103_150643_strict_baseline_npu23`. Both ranks completed 20 steps and reached **60,294.27 MiB actual allocated**. On the first ten steps, iterations 2–10 averaged 23.158/23.181 seconds for ranks 0/1. Reserved memory is diagnostic and is not the capacity metric. The exact rendered configuration is `config.accepted_strict_baseline.7l64e5v.yaml`; its matching renderer is `config.strict_baseline.template.yaml`.

The tested partial-recompute boundary removes recompute from language layers 0 and 3 together, leaving layers 1,2,4,5,6 and all visual blocks recomputed. Without HyperOffload it NPU-OOMed on the first backward step: 58.03 GiB allocated, 2 GiB requested, and 1.84–1.85 GiB free. The run is `20261103_160406_partial_no_offload_npu23`. Removing only layer 3 or only layer 5 separately completed its ten-step observation window. The boundary applies to the tested layer sets; it is not a sweep of all combinations.

## A2 environment

The A2 host has eight Ascend 910B3 NPUs with 64 GiB HBM each. Reuse `mindspeed-glm53-flash-env`, which keeps all eight cards visible. Each training process uses exactly two workers on one NUMA-local pair. The single MindSpeed Core runtime is `/data/disk1/machangwei/MindSpeed` on the reproducibility branch `1103_glm5.3_baseline`; GLM tools live here under `glm53-env/tools/glm53_flash_fsdp2_a2/`. The deployed MindSpeed-MM tree is the separate checkout at commit `a63faaf` plus `mindspeed_mm_a2_swap_optimizer.patch`.

Required stack: CANN 9.1, PyTorch 2.10.0, torch_npu 2.10.0.post4, Transformers 5.16.1, triton-ascend 3.2.2, matching triton-ascend-kernels, and A2 flash-linear-attention-npu 26.7.0.dev0 (`FLA_NPU_SOC=ascend910b`). The existing environment is `/data/disk1/machangwei/glm53-env/venv`.

Assets are already prepared under `/data/disk1/machangwei/glm53-env/`:

```text
model-assets/GLM-5.3-Flash-l7e64v5-proportional-proxy/
data/mock_vl/mock_data_pic_num_1_textlen_2048_num_samples_64.json
data/mock_vl/test_pic_h224_w224.jpg
```

The proxy config metadata is versioned in `model_metadata/GLM-5.3-Flash-l7e64v5/config.json`. It contains no checkpoint tensors. The image/tokenizer and synthetic samples remain local prepared assets. `make_fit_proxy.py --layers 7 --experts 64 --vision-depth 5 --name GLM-5.3-Flash-l7e64v5-proportional-proxy` recreates model metadata from the local official config/tokenizer without downloading weights.

The configuration and deployment scripts are versioned on `1103_glm5.3_baseline` under `examples/glm53_flash_fsdp2_a2/`. Keep one live Core checkout on this branch; deploy the script snapshot to the external tools directory with:

```bash
mkdir -p glm53-env/tools/glm53_flash_fsdp2_a2
git -C MindSpeed archive 1103_glm5.3_baseline:examples/glm53_flash_fsdp2_a2 | \
  tar -x -C glm53-env/tools/glm53_flash_fsdp2_a2
```

`assets.sha256.json` freezes the proxy metadata, tokenizer/processor configuration, dataset, and image. The renderer refuses changed assets or changes to the accepted YAML beyond the selected recompute layers and relocated asset paths. The historical reference used Core `5f28f9dd`; the shared runtime is now `11b3ef7f`. Historical timing is not a new measurement on the updated Core. Every follow-up must record the actual runtime revision.

## Reproduce the baseline

Before launch, check device health/process/HBM, NUMA topology and host memory, reservations, existing torchrun processes, and the persistent container. The launcher repeats the NPU and reservation checks. These commands use 6/7 as an example only; select an actually free same-NUMA pair.

The launcher and monitor require at least 160 GiB free host memory on that NUMA node (`GLM53_MIN_NUMA_FREE_MIB=163840`). This protects model initialization and the CPU optimizer/offload buffers. An idle NPU pair whose NUMA host memory is exhausted is not eligible. The measured free memory and threshold are recorded in `launch.txt`.

```bash
TASK_QUEUE_ENABLE=1 GLM53_STOP_AFTER_ITERS=10 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/launch_glm53_pair.sh strict_baseline 6,7
```

`GLM53_STOP_AFTER_ITERS=0` uses the complete 20-step configured schedule. By default, runs stop after ten observed iterations while preserving the original 20-step LR schedule. Each run gets a unique directory under `glm53-env/experiments/1103_glm53/` with the rendered YAML, launch details, actual Core/MindSpeed-MM revisions, source checksums, logs, exit code, and per-rank JSONL metrics. The launcher records actual allocated and reserved peaks separately.

## Test the same OOM case with HyperOffload

The accepted no-offload OOM above is the capacity evidence. For a paired performance result, use the comparison monitor: it launches a fresh strict full-recompute baseline and then the same `{0,3}` partial-recompute case with HyperOffload on the same pair and Core revision. It does not repeat the OOM probe:

```bash
TASK_QUEUE_ENABLE=1 GLM53_STOP_AFTER_ITERS=10 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/run_comparison_when_free.sh
```

For a manually selected pair, launch the HyperOffload case directly after a successful strict run:

```bash
TASK_QUEUE_ENABLE=1 GLM53_DISABLE_RECOMPUTE_LAYERS=0,3 GLM53_STOP_AFTER_ITERS=10 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/launch_glm53_pair.sh hyperoffload 6,7
```

The fixed initial HyperOffload policy uses the peak planner, 2048 MiB resident activation budget, and 32768 MiB host activation budget per rank. These can be overridden with `GLM53_OFFLOAD_PLANNER`, `GLM53_OFFLOAD_DEVICE_MB`, and `GLM53_OFFLOAD_HOST_MB`; record the values when comparing. HyperOffload skips recomputed modules and disables MindSpeed-MM stash/op replay. The renderer and runtime validator ensure the two disabled layer IDs are exact.

`bash glm53-env/tools/glm53_flash_fsdp2_a2/run_comparison_when_free.sh` waits for a healthy, idle same-NUMA pair, runs a fresh strict baseline, and then the accepted two-layer HyperOffload case on that pair. It uses the existing no-offload OOM as capacity evidence and does not repeat the OOM probe. This command starts the test sequence; it is not part of configuration preparation.

The first iteration includes tracing/planning and is excluded from the steady-time mean. Compare iterations 2–10 on the same pair, runtime, model, seed, allocator (`expandable_segments:True`), dataset, and optimizer. Include all iterations in peak-memory reporting. HyperOffload is an experimental GLM/FSDP integration; claim a performance result only if both ranks finish ten steps and record replay-phase transfers.

The HyperOffload runner fetches and moves each batch before entering its dispatch session. `TrainEngine` creates the DataLoader iterator lazily, so fetching a batch inside the active `TorchDispatchMode` can fork DataLoader workers that inherit the dispatch context. Those workers may spin in CPU preprocessing without returning a batch or launching NPU work. `run_glm53.py` explicitly disables active dispatch modes around `get_batch`; keep data loading outside the activation tracking scope.

## Logs and evidence

Keep runs in their unique experiment directories. The launcher refuses an existing run directory, so metrics and train.log cannot append to an earlier run. Before a new test series, archive and clear `glm53-env/logs/`. Historical loose GLM logs are now compressed in `glm53-env/log_archives/20261103_164915/legacy_glm_logs.tar.gz`, with a per-file SHA256 manifest. The current logs directory is empty. To read a historical file without restoring it, use `tar -xOf ARCHIVE glm53-env/logs/FILENAME`. Do not remove the accepted baseline or OOM run directories, their metrics/configuration/logs, Qwen measurements, or Qwen OOM evidence.
