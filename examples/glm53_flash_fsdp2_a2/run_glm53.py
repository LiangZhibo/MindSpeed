#!/usr/bin/env python3
"""Run MindSpeed-MM GLM-5.3-Flash with optional HyperOffload instrumentation."""

import argparse
import importlib
from functools import wraps
import json
import runpy
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import yaml
from torch.utils._python_dispatch import _disable_current_modes

def install_transfer_counters(session) -> None:
    manager = session.residency_manager
    counters = {"d2h_calls": 0, "d2h_bytes": 0, "h2d_calls": 0, "h2d_bytes": 0}
    original_d2h = manager.copy_d2h
    original_h2d = manager.copy_h2d

    def counted_d2h(storage_id):
        buffer = manager._residency.get(storage_id)
        size = None
        if buffer is not None and buffer.device_buffer is not None and buffer.host_buffer is None:
            size = buffer.device_buffer.numel()
        result = original_d2h(storage_id)
        if size is not None:
            counters["d2h_calls"] += 1
            counters["d2h_bytes"] += size
        return result

    def counted_h2d(storage_id):
        buffer = manager._residency.get(storage_id)
        size = None
        if buffer is not None and buffer.device_buffer is None and buffer.host_buffer is not None:
            size = buffer.host_buffer.numel()
        result = original_h2d(storage_id)
        if size is not None:
            counters["h2d_calls"] += 1
            counters["h2d_bytes"] += size
        return result

    manager.copy_d2h = counted_d2h
    manager.copy_h2d = counted_h2d
    session._glm53_transfer_counters = counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "hyperoffload"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--offload-device-mb", type=int, default=2048)
    parser.add_argument("--offload-host-mb", type=int, default=32768)
    parser.add_argument("--planner", choices=("peak", "greedy"), default="peak")
    parser.add_argument("--stop-after-iters", type=int, default=0)
    run_args = parser.parse_args()
    if run_args.stop_after_iters < 0:
        raise SystemExit("stop-after-iters must be nonnegative")

    config = yaml.safe_load(run_args.config.read_text(encoding="utf-8"))
    scheduled_train_iters = config["training"]["train_iters"]
    model_config_path = Path(config["model"]["model_name_or_path"]) / "config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_text = model_config["text_config"]
    model_shape = (
        model_text["num_hidden_layers"],
        model_text["n_routed_experts"],
        model_text["num_experts_per_tok"],
        model_config["vision_config"]["depth"],
    )
    is_hyperoffload = run_args.mode == "hyperoffload"
    if is_hyperoffload and config["features"]["enable_activation_offload"]:
        raise SystemExit("disable MindSpeed-MM activation stash before enabling HyperOffload")

    if is_hyperoffload:
        from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
        from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner

        planner = PeakResidencyPlanner() if run_args.planner == "peak" else GreedyResidencyPlanner()
        offload_config = OffloadConfig(
            max_resident_activation_mb=run_args.offload_device_mb,
            max_offload_activation_mb=run_args.offload_host_mb,
            planner=planner,
        )
    else:
        OffloadConfig = OffloadSession = skip_offload = None
        offload_config = None

    from mindspeed_mm.fsdp.train.train_engine import TrainEngine
    from mindspeed_mm.fsdp.utils.str_match import module_name_match

    original_step = TrainEngine.train_step
    original_log = TrainEngine.training_log
    original_get_batch = TrainEngine.get_batch

    @wraps(original_get_batch)
    def instrumented_get_batch(engine, train_dataloader_iter):
        if not is_hyperoffload:
            return original_get_batch(engine, train_dataloader_iter)
        # DataLoader workers are created lazily on the first next(). Forking
        # while TorchDispatchMode is active makes workers inherit the mode and
        # sends CPU-side preprocessing through the activation tracker.
        with _disable_current_modes():
            return original_get_batch(engine, train_dataloader_iter)

    def prepare_skip_regions(engine):
        if getattr(engine, "_glm53_skip_regions_ready", False):
            return
        modules = dict(engine.model.named_modules())
        selected = []
        for pattern in config["features"]["recompute_plan"]["apply_modules"]:
            selected.extend(
                (name, module) for name, module in modules.items()
                if name and module_name_match(pattern, name)
            )
        unique = {name: module for name, module in selected}
        if not unique:
            raise RuntimeError("HyperOffload found no recompute modules to mark as opaque")
        for name, module in unique.items():
            if getattr(module, "_glm53_skip_offload", False):
                continue
            module.forward = skip_offload(module.forward)
            module._glm53_skip_offload = True
        engine._glm53_skip_regions_ready = True
        print(f"[glm53] HyperOffload opaque recompute modules={','.join(unique)}", flush=True)

        # These custom autograd functions launch Triton kernels that consume
        # native storage pointers. Isolate just those calls so adjacent eager
        # activations remain eligible for offload.
        modeling = importlib.import_module(
            "mindspeed_mm.fsdp.models.glm5_next.modeling_glm5_next"
        )
        for function_name in ("hc_split_sinkhorn", "causal_conv1d", "chunk_kda_ascendc"):
            function = getattr(modeling, function_name, None)
            if function is None:
                raise RuntimeError(f"HyperOffload cannot find GLM kernel function {function_name}")
            if not getattr(function, "_glm53_skip_offload", False):
                wrapped_function = skip_offload(function, track_backward_inputs=True)
                wrapped_function._glm53_skip_offload = True
                setattr(modeling, function_name, wrapped_function)
        print(
            "[glm53] HyperOffload opaque custom kernels="
            "hc_split_sinkhorn,causal_conv1d,chunk_kda_ascendc",
            flush=True,
        )

    @wraps(original_step)
    def instrumented_step(engine, train_dataloader_iter):
        torch.npu.reset_peak_memory_stats()
        engine._glm53_step_start = time.perf_counter()
        if not is_hyperoffload:
            engine._glm53_phase_for_log = "disabled"
            return original_step(engine, train_dataloader_iter)

        prepare_skip_regions(engine)
        if not hasattr(engine, "_glm53_offload_session"):
            engine._glm53_offload_session = OffloadSession(offload_config)
            install_transfer_counters(engine._glm53_offload_session)
        session = engine._glm53_offload_session
        engine._glm53_phase_for_log = session.mode
        try:
            with session:
                return original_step(engine, train_dataloader_iter)
        except Exception:
            counters = session._glm53_transfer_counters
            manager = session.residency_manager
            print(
                "[glm53] HyperOffload failure diagnostics "
                f"phase={session.mode} d2h_calls={counters['d2h_calls']} "
                f"d2h_bytes={counters['d2h_bytes']} h2d_calls={counters['h2d_calls']} "
                f"h2d_bytes={counters['h2d_bytes']} resident_activation_bytes={manager.resident_bytes}",
                flush=True,
            )
            try:
                print(
                    "[glm53] NPU failure memory "
                    f"allocated={torch.npu.memory_allocated()} reserved={torch.npu.memory_reserved()}",
                    flush=True,
                )
            except Exception as memory_error:  # NPU may reject queries after a device failure.
                print(f"[glm53] NPU memory query failed: {memory_error}", flush=True)
            raise

    @wraps(original_log)
    def instrumented_log(engine, iteration, elapsed_time_per_iteration, *call_args, **kwargs):
        result = original_log(engine, iteration, elapsed_time_per_iteration, *call_args, **kwargs)
        rank = torch.distributed.get_rank()
        payload = {
            "mode": run_args.mode,
            "model_shape": f"{model_shape[0]}L/{model_shape[1]}E/top-{model_shape[2]}/{model_shape[3]}V",
            "rank": rank,
            "iteration": iteration,
            "iteration_seconds": elapsed_time_per_iteration,
            "train_step_wall_seconds": time.perf_counter() - engine._glm53_step_start,
            "npu_peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "npu_peak_reserved_bytes": torch.npu.max_memory_reserved(),
            "hyperoffload_phase": engine._glm53_phase_for_log,
        }
        if is_hyperoffload:
            counters = engine._glm53_offload_session._glm53_transfer_counters
            previous = getattr(engine, "_glm53_previous_transfer_counters", {key: 0 for key in counters})
            payload["hyperoffload_transfers"] = {
                key: counters[key] - previous[key] for key in counters
            }
            engine._glm53_previous_transfer_counters = counters.copy()
        run_args.metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = run_args.metrics_dir / f"rank{rank}.jsonl"
        with metrics_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if run_args.stop_after_iters and iteration >= run_args.stop_after_iters:
            if engine.args.training.train_iters > iteration:
                engine.args.training.train_iters = iteration
                print(
                    f"[glm53] stopping after {iteration} observed iterations; "
                    f"the configured {scheduled_train_iters}-step schedule was initialized unchanged",
                    flush=True,
                )
        return result

    TrainEngine.train_step = instrumented_step
    TrainEngine.training_log = instrumented_log
    TrainEngine.get_batch = instrumented_get_batch

    import mindspeed_mm
    trainer_path = Path(mindspeed_mm.__file__).resolve().parent / "fsdp/train/trainer.py"
    if not trainer_path.is_file():
        raise SystemExit(f"cannot find MindSpeed-MM FSDP trainer: {trainer_path}")
    previous_argv = sys.argv
    try:
        sys.argv = [str(trainer_path), str(run_args.config.resolve())]
        runpy.run_path(str(trainer_path), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
