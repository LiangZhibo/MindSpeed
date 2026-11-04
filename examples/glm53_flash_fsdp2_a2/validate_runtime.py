#!/usr/bin/env python3
"""Read-only A2 environment and GLM proxy preflight."""

import argparse
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("strict_baseline", "hyperoffload"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--disable-language-recompute", default="0,3")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_dir = Path(config["model"]["model_name_or_path"])
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dataset = Path(config["data"]["dataset_param"]["basic_parameters"]["dataset"])
    samples = json.loads(dataset.read_text(encoding="utf-8"))
    if len(samples) != 64 or not (dataset.parent / samples[0]["images"][0]).is_file():
        raise SystemExit("expected the 64-sample GLM mock vision-language dataset")
    shape = (
        model_config["text_config"]["num_hidden_layers"],
        model_config["text_config"]["n_routed_experts"],
        model_config["text_config"]["num_experts_per_tok"],
        model_config["vision_config"]["depth"],
    )
    expected_shape = (7, 64, 8, 5)
    if shape != expected_shape:
        raise SystemExit(f"expected proportional proxy shape {expected_shape}, got {shape}")
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin", "*.pt")
    if any(path for pattern in weight_patterns for path in model_dir.glob(pattern)):
        raise SystemExit("proxy directory contains checkpoint weights; this run is intended to use random init")
    if torch.npu.device_count() != 2:
        raise SystemExit(f"training workers must see exactly two NPUs, got {torch.npu.device_count()}")

    import mindspeed_mm.fsdp  # noqa: F401
    from mindspeed_mm.fsdp.distributed.fully_shard_parallel import fully_shard_parallel_modules  # noqa: F401
    from mindspeed_mm.fsdp.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from mindspeed_mm.fsdp.models.glm5_next import modeling_glm5_next
    from torch.distributed.fsdp import fully_shard

    if "hook_module" not in inspect.signature(fully_shard).parameters:
        raise SystemExit("MindSpeed-MM FSDP2 hook_module adaptation is not active")
    if not fully_shard.__module__.startswith("mindspeed_mm.fsdp.ops.fully_shard"):
        raise SystemExit(f"unexpected FSDP implementation: {fully_shard.__module__}")
    parsed_config = Glm5NextConfig.from_pretrained(model_dir, local_files_only=True)
    modeling_glm5_next.Glm5NextForConditionalGeneration.overwrite_transformer_config(
        parsed_config,
        SimpleNamespace(**config["model"]),
        SimpleNamespace(**config["features"]),
    )
    with torch.device("meta"):
        meta_model = modeling_glm5_next.Glm5NextForConditionalGeneration(parsed_config)
    parameter_count = sum(parameter.numel() for parameter in meta_model.parameters())
    if parameter_count <= 0:
        raise SystemExit(f"invalid proxy parameter count: {parameter_count}")
    from mindspeed_mm.fsdp.utils.str_match import module_name_match

    module_names = [name for name, _ in meta_model.named_modules()]
    recompute_patterns = config["features"]["recompute_plan"]["apply_modules"]
    recompute_matches = {
        name for pattern in recompute_patterns for name in module_names
        if name and module_name_match(pattern, name)
    }
    if not recompute_matches:
        raise SystemExit("recompute plan does not match any GLM modules")
    if args.mode == "hyperoffload":
        language_layers = {
            name for name in recompute_matches
            if name.startswith("model.language_model.layers.") and name.count(".") == 3
        }
        try:
            disabled_layers = {int(item) for item in args.disable_language_recompute.split(",") if item}
        except ValueError as exc:
            raise SystemExit("disabled language layer IDs must be comma-separated integers") from exc
        if not disabled_layers or any(index < 0 or index >= shape[0] for index in disabled_layers):
            raise SystemExit(f"disabled language layer IDs must be within 0..{shape[0] - 1}")
        expected_recomputed_layers = {
            f"model.language_model.layers.{index}"
            for index in range(shape[0]) if index not in disabled_layers
        }
        if language_layers != expected_recomputed_layers:
            raise SystemExit(f"comparison must recompute {sorted(expected_recomputed_layers)}, got {sorted(language_layers)}")
    model_options = config["model"]
    if model_options.get("kda_implementation") == "ascendc":
        try:
            from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd  # noqa: F401
            from triton_ascend_kernels.attention.fla.kda.l2norm_kda import l2norm_fwd  # noqa: F401
        except Exception as exc:
            raise SystemExit(f"A2 KDA AscendC stack is incomplete: {exc}") from exc
    try:
        triton_version = metadata.version("triton-ascend")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit("optimized GLM config requires triton-ascend==3.2.2") from exc
    if triton_version != "3.2.2":
        raise SystemExit(f"expected triton-ascend==3.2.2, got {triton_version}")
    if model_options.get("dsa_implementation") == "sfa" and not hasattr(torch_npu, "npu_sparse_flash_attention"):
        raise SystemExit("torch_npu does not expose sparse flash attention")

    if args.mode == "hyperoffload":
        from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload  # noqa: F401
        from mindspeed.core.memory.hyper_offload.planning import PeakResidencyPlanner  # noqa: F401
        if config["features"]["enable_activation_offload"]:
            raise SystemExit("HyperOffload cannot be combined with MindSpeed-MM activation stash")

    os_version = torch.__version__
    print(
        f"runtime ready: torch={os_version} torch_npu={torch_npu.__version__} "
        f"visible_npus={torch.npu.device_count()} fsdp2={fully_shard.__module__} "
        f"proxy={shape[0]}L/{shape[1]}E/top-{shape[2]}/{shape[3]}V params={parameter_count:,} "
        f"recompute_modules={len(recompute_matches)} mode={args.mode}"
    )


if __name__ == "__main__":
    main()
