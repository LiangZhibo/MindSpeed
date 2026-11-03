#!/usr/bin/env python3
"""Render and validate a portable GLM-5.3-Flash A2 run configuration."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
TEMPLATES = {
    "baseline": HERE / "config.baseline.template.yaml",
    "strict_baseline": HERE / "config.strict_baseline.template.yaml",
    "partial_no_offload": HERE / "config.hyperoffload.template.yaml",
    "hyperoffload": HERE / "config.hyperoffload.template.yaml",
}
PROXY_NAMES = {
    "baseline": "GLM-5.3-Flash-l8e56v5-proportional-proxy",
    "strict_baseline": "GLM-5.3-Flash-l7e64v5-proportional-proxy",
    "partial_no_offload": "GLM-5.3-Flash-l7e64v5-proportional-proxy",
    "hyperoffload": "GLM-5.3-Flash-l7e64v5-proportional-proxy",
}
DATASET_NAME = "mock_data_pic_num_1_textlen_2048_num_samples_64.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=TEMPLATES, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-language-recompute", default="0,3")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    model_dir = base_dir / "model-assets" / PROXY_NAMES[args.mode]
    data_dir = base_dir / "data" / "mock_vl"
    dataset_path = data_dir / DATASET_NAME
    for required_path in (
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        dataset_path,
        data_dir / "test_pic_h224_w224.jpg",
    ):
        if not required_path.is_file():
            raise SystemExit(f"required GLM proxy asset is missing: {required_path}")

    if args.mode != "baseline":
        manifest = json.loads((HERE / "assets.sha256.json").read_text(encoding="utf-8"))
        for relative_path, expected_digest in manifest.items():
            asset_path = base_dir / relative_path
            if not asset_path.is_file():
                raise SystemExit(f"frozen baseline asset is missing: {asset_path}")
            actual_digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise SystemExit(f"frozen baseline asset has changed: {asset_path}")

    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = model_config["text_config"]
    actual_shape = (
        text_config["num_hidden_layers"],
        text_config["n_routed_experts"],
        text_config["num_experts_per_tok"],
        model_config["vision_config"]["depth"],
    )
    expected_shape = (8, 56, 8, 5) if args.mode == "baseline" else (7, 64, 8, 5)
    if actual_shape != expected_shape:
        raise SystemExit(f"expected proportional proxy shape {expected_shape}, got {actual_shape}")

    config = yaml.safe_load(TEMPLATES[args.mode].read_text(encoding="utf-8"))
    if args.mode in ("partial_no_offload", "hyperoffload"):
        try:
            disabled_layers = {int(item) for item in args.disable_language_recompute.split(",") if item}
        except ValueError as exc:
            raise SystemExit("disabled language layer IDs must be comma-separated integers") from exc
        layer_count = text_config["num_hidden_layers"]
        if not disabled_layers or any(index < 0 or index >= layer_count for index in disabled_layers):
            raise SystemExit(f"disabled language layer IDs must be within 0..{layer_count - 1}")
        non_layer_patterns = [
            pattern for pattern in config["features"]["recompute_plan"]["apply_modules"]
            if not pattern.startswith("model.language_model.layers.")
        ]
        config["features"]["recompute_plan"]["apply_modules"] = non_layer_patterns + [
            f"model.language_model.layers.{index}"
            for index in range(layer_count) if index not in disabled_layers
        ]
    config["model"]["model_name_or_path"] = str(model_dir)
    dataset = config["data"]["dataset_param"]
    dataset["preprocess_parameters"]["model_name_or_path"] = str(model_dir)
    dataset["basic_parameters"]["dataset_dir"] = str(data_dir)
    dataset["basic_parameters"]["dataset"] = str(dataset_path)
    if config["parallel"]["fully_shard_parallel_size"] != 2:
        raise SystemExit("this reproduction requires FSDP2 size 2")
    if config["training"]["micro_batch_size"] != 1 or config["training"]["log_interval"] != 1:
        raise SystemExit("microbatch=1 and log_interval=1 are required for comparable measurements")
    if config["features"]["recompute"] is not True:
        raise SystemExit("both comparison groups must use MindSpeed-MM recompute")
    if args.mode in ("strict_baseline", "partial_no_offload", "hyperoffload") and config["features"]["recompute_plan"].get("op_replay_scopes"):
        raise SystemExit("the primary comparison disables other activation-storage features")
    if args.mode in ("partial_no_offload", "hyperoffload") and config["features"]["enable_activation_offload"]:
        raise SystemExit("the comparison disables MindSpeed-MM activation stash")

    if args.mode != "baseline":
        frozen = yaml.safe_load((HERE / "config.accepted_strict_baseline.7l64e5v.yaml").read_text(encoding="utf-8"))
        frozen["model"]["model_name_or_path"] = str(model_dir)
        frozen_dataset = frozen["data"]["dataset_param"]
        frozen_dataset["preprocess_parameters"]["model_name_or_path"] = str(model_dir)
        frozen_dataset["basic_parameters"]["dataset_dir"] = str(data_dir)
        frozen_dataset["basic_parameters"]["dataset"] = str(dataset_path)
        if args.mode in ("partial_no_offload", "hyperoffload"):
            frozen["features"]["recompute_plan"]["apply_modules"] = config["features"]["recompute_plan"]["apply_modules"]
        if frozen != config:
            raise SystemExit("configuration differs from the accepted baseline beyond the selected recompute layers")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"rendered {args.mode} config: {output}")
    print(f"proxy shape={actual_shape[0]}L/{actual_shape[1]}E/top-{actual_shape[2]}/{actual_shape[3]}V; random init; no checkpoint weights")


if __name__ == "__main__":
    main()
