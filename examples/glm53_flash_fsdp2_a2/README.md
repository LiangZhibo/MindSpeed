# GLM-5.3-Flash FSDP2 + HyperOffload reproduction

This example contains the run configurations and launch switches for reproducing the tested strict-recompute and HyperOffload runs. It does not bundle measured result files.

## Fixed setup

- Model proxy: 7 language layers, 64 routed experts, top-8 routing, 5 vision blocks; BF16; random initialization; no checkpoint weights.
- Parallelism: MindSpeed-MM FSDP2, shard size 2, exactly two workers on one healthy same-NUMA NPU pair.
- Input/training: fixed 64-sample synthetic image/text set, cutoff 4096, microbatch 1, seed 42, AdamWSwap.
- Memory/runtime switches: `TASK_QUEUE_ENABLE=1`, `GLM53_GROUPED_MOE_EP1=1`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `ASCEND_LAUNCH_BLOCKING=0`, `CPU_AFFINITY_CONF=1`, `MULTI_STREAM_MEMORY_REUSE=2`.
- MindSpeed-MM activation stash is off (`features.enable_activation_offload: false`); recompute op replay scopes are empty; chunk loss is enabled; chunk microbatching is off.

The strict configuration recomputes all seven language layers and all five vision blocks. HyperOffload disables recompute for language layers 0 and 3; it recomputes layers 1, 2, 4, 5, 6 and all vision blocks, then offloads eligible activations. The HyperOffload settings used for the paired run are peak planner, 4096 MiB device budget and 32768 MiB host budget per rank.

`config.strict_baseline.template.yaml` and `config.hyperoffload.template.yaml` are the only training configurations. The renderer checks that the templates differ only in the selected recompute layers. Runtime model/data paths are filled from the local asset root by `render_config.py`.

## Required assets and runtime

Provide these files under the runtime asset root; `assets.sha256.json` verifies their contents before rendering:

```text
model-assets/GLM-5.3-Flash-l7e64v5-proportional-proxy/{config.json,tokenizer.json,tokenizer_config.json,processor_config.json}
data/mock_vl/mock_data_pic_num_1_textlen_2048_num_samples_64.json
data/mock_vl/test_pic_h224_w224.jpg
```

The model metadata is also versioned in `model_metadata/GLM-5.3-Flash-l7e64v5/config.json`. It contains no model weights. The model/tokenizer and synthetic data are required local inputs and are not copied into this example.

Use the tested A2 software stack: CANN 9.1.0, PyTorch 2.10.0, torch_npu 2.10.0.post4, Transformers 5.16.1, triton-ascend 3.2.2, matching triton-ascend-kernels, and flash-linear-attention-npu 26.7.0.dev0 built with `FLA_NPU_SOC=ascend910b`. MindSpeed-MM was tested at `a63faaf1a0291438d055618445129522d6ef4aca` with `mindspeed_mm_a2_swap_optimizer.patch` applied:

```bash
git -C MindSpeed-MM apply --check ../MindSpeed/examples/glm53_flash_fsdp2_a2/mindspeed_mm_a2_swap_optimizer.patch
git -C MindSpeed-MM apply ../MindSpeed/examples/glm53_flash_fsdp2_a2/mindspeed_mm_a2_swap_optimizer.patch
```

Skip patch application if it is already applied. The launch scripts target the tested host layout `/data/disk1/machangwei` and persistent `mindspeed-glm53-flash-env` container. On another host, adapt the root, container name, NPU PCI/NUMA mapping and reservation paths in both shell scripts before launching.

Both modes enable the same local grouped MoE kernels with `GLM53_GROUPED_MOE_EP1=1`, retaining expert parallel size 1. HyperOffload additionally wraps recomputed modules and native-pointer kernels/metadata methods with `skip_offload`; these wrappers are installed automatically by `run_glm53.py`.

## Run a paired comparison

Deploy the example from the checked out branch into the runtime tools directory:

```bash
mkdir -p glm53-env/tools/glm53_flash_fsdp2_a2
git -C MindSpeed archive 1104:examples/glm53_flash_fsdp2_a2 | \
  tar -x -C glm53-env/tools/glm53_flash_fsdp2_a2
```

Check NPU health/occupancy, NUMA topology and free host memory, reservations, existing training processes, and the persistent container before launch. The monitor selects a healthy idle same-NUMA pair, requires 160 GiB free on its NUMA node by default, then runs strict followed by HyperOffload on that same pair. To use a different host-memory gate, set `GLM53_MIN_NUMA_FREE_MIB` explicitly.

```bash
TASK_QUEUE_ENABLE=1 \
GLM53_GROUPED_MOE_EP1=1 \
GLM53_STOP_AFTER_ITERS=10 \
GLM53_DISABLE_RECOMPUTE_LAYERS=0,3 \
GLM53_OFFLOAD_PLANNER=peak \
GLM53_OFFLOAD_DEVICE_MB=4096 \
GLM53_OFFLOAD_HOST_MB=32768 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/run_comparison_when_free.sh
```

The 20-step LR schedule remains configured; the launcher stops after 10 observed steps. Iteration 1 performs HyperOffload warmup/planning, so use iterations 2–10 for steady-time comparisons. Runs and their generated logs/metrics go under `glm53-env/experiments/1103_glm53/`, outside the versioned example.

To run one mode directly on an already selected pair, use `strict_baseline` or `hyperoffload`:

```bash
TASK_QUEUE_ENABLE=1 GLM53_GROUPED_MOE_EP1=1 GLM53_STOP_AFTER_ITERS=10 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/launch_glm53_pair.sh strict_baseline 2,3

TASK_QUEUE_ENABLE=1 GLM53_GROUPED_MOE_EP1=1 GLM53_DISABLE_RECOMPUTE_LAYERS=0,3 \
GLM53_STOP_AFTER_ITERS=10 GLM53_OFFLOAD_PLANNER=peak \
GLM53_OFFLOAD_DEVICE_MB=4096 GLM53_OFFLOAD_HOST_MB=32768 \
  bash glm53-env/tools/glm53_flash_fsdp2_a2/launch_glm53_pair.sh hyperoffload 2,3
```

The launcher also sets the two-NPU visibility, `NON_MEGATRON=true`, NUMA CPU/memory binding, and the allocator/runtime switches listed above. It renders and validates the final YAML before starting training, and records the actual Core/MindSpeed-MM revisions and launch parameters with each run.
