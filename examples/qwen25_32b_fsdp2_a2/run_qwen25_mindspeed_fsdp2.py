#!/usr/bin/env python3
"""Run a cropped Qwen step with MindSpeed FSDP2 on two NPUs."""

import argparse
from contextlib import nullcontext
from functools import wraps
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from correctness import NumericalValidator, check_chunk_loss, finite_scalar, selected_steps

from mindspeed.fsdp.mindspeed_parallel_engine import MindSpeedParallelEngine
from mindspeed.fsdp.parallel_engine_config import FSDPPlanConfig, ParallelEngineConfig
from mindspeed.fsdp.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss
from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner

opaque_chunk_loss = skip_offload(chunk_loss)


class ChunkedHead(torch.nn.Module):
    def __init__(self, original, chunk_size):
        super().__init__()
        self.weight = original.weight
        self.chunk_size = chunk_size

    def forward(self, hidden, labels):
        normalizer = torch.tensor(labels.numel(), dtype=torch.float32, device=hidden.device)
        loss_kwargs = [
            dict(shift_labels=part, alpha=normalizer, reduction="sum")
            for part in labels.split(self.chunk_size, dim=1)
        ]
        return opaque_chunk_loss(hidden, self.weight, None, calculate_lm_loss, loss_kwargs, self.chunk_size)


class TrainingModel(torch.nn.Module):
    def __init__(self, hf_model, chunk_size):
        super().__init__()
        self.model = hf_model.model
        self.lm_head = ChunkedHead(hf_model.lm_head, chunk_size)

    def forward(self, input_ids, labels):
        hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return self.lm_head(hidden[:, :-1, :], labels[:, 1:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--loss-chunk", type=int, default=128)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--recompute-layers", type=int)
    parser.add_argument("--hyper-offload", action="store_true")
    parser.add_argument("--offload-device-mb", type=int, default=4096)
    parser.add_argument("--offload-host-mb", type=int, default=32768)
    parser.add_argument("--offload-planner", choices=("peak", "greedy"), default="peak")
    parser.add_argument("--extra-steps", type=int, default=0)
    parser.add_argument("--correctness-mode", choices=("none", "record", "compare"), default="none")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--correctness-steps", help="snapshot steps, default: first, second, last")
    for kind, rtol, atol in (("loss", 1e-3, 1e-3), ("gradient", 2e-2, 1e-5), ("parameter", 1e-2, 1e-5)):
        parser.add_argument(f"--{kind}-rtol", type=float, default=rtol)
        parser.add_argument(f"--{kind}-atol", type=float, default=atol)
    args = parser.parse_args()
    if args.correctness_mode != "none":
        if args.reference_dir is None:
            parser.error("--reference-dir is required for numerical validation")
        if not args.no_profile:
            parser.error("numerical validation requires --no-profile; run performance separately")
        try:
            args.validation_steps = selected_steps(args.correctness_steps, 1 + args.extra_steps)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def log(rank, message):
    print(f"[rank {rank}] {message}", flush=True)


def crop_layers(model, layers):
    source_layers = len(model.model.layers)
    if layers < 1 or layers > source_layers:
        raise ValueError(f"--layers must be in [1, {source_layers}], got {layers}")
    if layers < source_layers:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layers]))
        model.config.num_hidden_layers = layers
        model.model.config.num_hidden_layers = layers
    return model


def exclude_fsdp_hooks_from_offload():
    """Keep FSDP parameter gather/reshard outside activation tracking."""
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    from torch.utils._python_dispatch import _disable_current_modes

    def wrap_hook(cls, name):
        original = getattr(cls, name)

        @wraps(original)
        def without_dispatch_modes(self, *args, **kwargs):
            with _disable_current_modes():
                return original(self, *args, **kwargs)

        setattr(cls, name, without_dispatch_modes)

    for name in ("_pre_forward", "_post_forward", "_pre_backward", "_root_post_backward_final_callback"):
        wrap_hook(FSDPState, name)
    wrap_hook(FSDPParamGroup, "post_backward")


def make_engine(args, rank):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    crop_layers(model, args.layers)
    model.config.use_cache = False
    recompute_layers = args.recompute_layers
    if recompute_layers is None:
        recompute_layers = args.layers if args.recompute else 0
    if not 0 <= recompute_layers <= args.layers:
        raise ValueError(f"--recompute-layers must be in [0, {args.layers}]")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    layer_names = [f"model.layers.{index}" for index in range(args.layers)]
    training_model = TrainingModel(model, args.loss_chunk)
    names = ["model.embed_tokens", *layer_names, "model.norm", "lm_head"]
    fsdp_plan = FSDPPlanConfig(
        ignored_modules=[],
        apply_modules={name: {"reshard_after_forward": True} for name in names},
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
        cast_forward_inputs=True,
        num_to_forward_prefetch=1,
        num_to_backward_prefetch=1,
    )
    config = ParallelEngineConfig(
        fully_shard_parallel_size=2,
        fsdp_plan=fsdp_plan,
        recompute=recompute_layers > 0,
        recompute_plan=[f"model.layers.{{0-{recompute_layers - 1}}}"] if recompute_layers else [],
    )
    engine = MindSpeedParallelEngine(config, training_model)
    if args.hyper_offload:
        for layer in engine.model.model.layers[:recompute_layers]:
            layer.forward = skip_offload(layer.forward)
    log(
        rank,
        "ENGINE_READY "
        f"parameters={parameter_count} layers={args.layers} fsdp_size=2 "
        f"cpu_offload=False activation_swap=False recompute_layers={recompute_layers} "
        f"hyper_offload={args.hyper_offload}",
    )
    assert not any(name == "megatron" or name.startswith("megatron.") for name in sys.modules)
    return engine, model.config.vocab_size, parameter_count


def make_profile(args):
    if args.no_profile:
        return None
    profile_dir = Path(args.profile_path)
    profile_dir.mkdir(parents=True, exist_ok=True)
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        profile_memory=True,
        experimental_config=experimental,
        # Skip the first four calls, then capture calls 5 and 6.
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=2, repeat=1, skip_first=4),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    )


def validate_on_all_ranks(check):
    error = None
    try:
        value = check()
    except Exception as exc:
        error = str(exc)
        value = None
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device="npu")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError(error or "validation failed on another rank")
    return value


def main():
    args = parse_args()
    if args.extra_steps < 0:
        raise ValueError("--extra-steps must be nonnegative")
    if args.hyper_offload:
        exclude_fsdp_hooks_from_offload()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(16)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise RuntimeError(f"expected exactly two NPU workers, got world size {dist.get_world_size()}")
    apply_hccl_premul_sum_patch()
    loss_self_check = validate_on_all_ranks(lambda: check_chunk_loss(chunk_loss, calculate_lm_loss))
    log(rank, f"ENV torch={torch.__version__} torch_npu={torch_npu.__version__} visible={os.environ['ASCEND_RT_VISIBLE_DEVICES']}")
    engine, vocab_size, parameter_count = make_engine(args, rank)
    engine.train()
    optimizer = torch.optim.SGD(engine.parameters(), lr=1.0e-2)
    probe_parameter = engine.model.model.layers[0].mlp.down_proj.weight
    probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
    probe_before = probe_local.detach().reshape(-1)[:4096].clone()
    input_ids = torch.randint(vocab_size, (1, args.seq_length), device="npu")
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    validator = None
    if args.correctness_mode != "none":
        contract = dict(rank=rank, world_size=dist.get_world_size(), total_steps=1 + args.extra_steps,
                        seed=1234 + rank, layers=args.layers, seq_length=args.seq_length,
                        loss_chunk=args.loss_chunk, parameter_count=parameter_count,
                        optimizer="SGD", learning_rate=0.01, torch_version=torch.__version__,
                        torch_npu_version=torch_npu.__version__, parameter_dtype="bfloat16",
                        reduce_dtype="float32", input_shape=list(input_ids.shape))
        tolerances = {kind: dict(rtol=getattr(args, f"{kind}_rtol"), atol=getattr(args, f"{kind}_atol"))
                      for kind in ("loss", "gradient", "parameter")}
        validator = validate_on_all_ranks(lambda: NumericalValidator(
            engine, input_ids, args.reference_dir, rank, args.correctness_mode,
            args.validation_steps, contract, tolerances))
    profiler = make_profile(args)
    session = None
    offload_stats = {"d2h_calls": 0, "d2h_gib": 0.0}
    if args.hyper_offload:
        session = OffloadSession(OffloadConfig(
            max_resident_activation_mb=args.offload_device_mb,
            max_offload_activation_mb=args.offload_host_mb,
            planner=PeakResidencyPlanner() if args.offload_planner == "peak" else GreedyResidencyPlanner(),
        ))
        manager = session.residency_manager
        original_copy_d2h = manager.copy_d2h

        def counted_copy_d2h(storage_id):
            size = manager.device_resident_size(storage_id) or 0
            original_copy_d2h(storage_id)
            offload_stats["d2h_calls"] += 1
            offload_stats["d2h_gib"] += size / 2**30

        manager.copy_d2h = counted_copy_d2h
    started = time.perf_counter()
    result = {
        "rank": rank,
        "chunk_loss_self_check": loss_self_check,
        "correctness": {"mode": "none", "status": "not_requested"} if validator is None else validator.summary("pending"),
        "world_size": 2,
        "parameter_count": parameter_count,
        "layers": args.layers,
        "seq_length": args.seq_length,
        "steps": 0,
        "fsdp_size": 2,
        "cpu_offload": False,
        "activation_swap": False,
        "recompute": (args.recompute_layers if args.recompute_layers is not None else (args.layers if args.recompute else 0)) > 0,
        "recompute_layers": args.recompute_layers if args.recompute_layers is not None else (args.layers if args.recompute else 0),
        "hyper_offload": args.hyper_offload,
        "offload_device_mb": args.offload_device_mb if args.hyper_offload else None,
        "offload_host_mb": args.offload_host_mb if args.hyper_offload else None,
        "offload_planner": args.offload_planner if args.hyper_offload else None,
        "optimizer": "SGD",
        "learning_rate": 1.0e-2,
    }

    def check_backward(step, loss, grad_max):
        def check():
            finite_scalar(loss, f"step {step} loss")
            finite_scalar(grad_max, f"step {step} probe gradient", positive=True)
            if validator is not None:
                validator.backward(step, loss)
        validate_on_all_ranks(check)

    def check_update(step, update):
        def check():
            finite_scalar(update, f"step {step} parameter update", positive=True)
            if validator is not None:
                validator.update(step)
        validate_on_all_ranks(check)

    def write_progress():
        snapshot = dict(result)
        snapshot["elapsed_seconds"] = time.perf_counter() - started
        if session is not None:
            snapshot.update(offload_stats)
            snapshot["host_pool_allocated_gib"] = session.residency_manager._host_pool.total_allocated / 2**30
        result_path = Path(args.profile_path) / f"result_rank{rank}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(snapshot, indent=2))
        temporary_path.replace(result_path)

    try:
        if profiler is not None:
            profiler.start()
        with session if session is not None else nullcontext():
            loss = engine(input_ids, input_ids)
            log(rank, "FORWARD_COMPLETE")
            loss.backward()
        torch.npu.synchronize()
        log(rank, "BACKWARD_COMPLETE")
        probe_grad = probe_parameter.grad
        if probe_grad is None:
            raise RuntimeError("first-layer MLP probe parameter has no gradient")
        probe_grad_local = probe_grad.to_local() if isinstance(probe_grad, DTensor) else probe_grad
        grad_max = float(probe_grad_local.detach().reshape(-1)[:4096].abs().max())
        check_backward(1, float(loss.item()), grad_max)
        optimizer.step()
        if profiler is not None:
            profiler.step()
        probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
        update = float((probe_local.detach().reshape(-1)[:4096] - probe_before).abs().max())
        check_update(1, update)
        result.update(
            status="one_step_complete", steps=1, loss=float(loss.item()),
            probe_grad_max=grad_max, parameter_update_max=update,
            first_step_elapsed_seconds=time.perf_counter() - started,
            first_step_npu_peak_allocated_gib=torch.npu.max_memory_allocated() / 2**30,
            first_step_npu_peak_reserved_gib=torch.npu.max_memory_reserved() / 2**30,
        )
        if session is not None and session.mode == "replay":
            schedule = session.executor._schedule
            actions = [action for phase in (schedule.pre, schedule.post)
                       for group in phase.values() for action in group]
            result["offload_plan_actions"] = {
                kind: sum(action.kind.name == kind for action in actions)
                for kind in ("COPY_D2H", "COPY_H2D", "RELEASE_DEVICE", "RELEASE_HOST")
            }
        # Do not keep the custom ChunkLoss graph or its probe gradient alive
        # while later training steps allocate their working buffers.
        del loss, probe_grad, probe_grad_local
        if args.extra_steps:
            result["extra_step_results"] = []
            result["status"] = "running"
        write_progress()
        for step_index in range(args.extra_steps):
            optimizer.zero_grad(set_to_none=True)
            probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
            before = probe_local.detach().reshape(-1)[:4096].clone()
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            step_started = time.perf_counter()
            with session if session is not None else nullcontext():
                step_loss = engine(input_ids, input_ids)
                step_loss.backward()
            torch.npu.synchronize()
            step_grad = probe_parameter.grad
            if step_grad is None:
                raise RuntimeError("first-layer MLP probe parameter has no gradient")
            step_grad_local = step_grad.to_local() if isinstance(step_grad, DTensor) else step_grad
            step_grad_max = float(step_grad_local.detach().reshape(-1)[:4096].abs().max())
            check_backward(step_index + 2, float(step_loss.item()), step_grad_max)
            optimizer.step()
            probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
            step_result = {
                "index": step_index + 2,
                "loss": float(step_loss.item()),
                "probe_grad_max": step_grad_max,
                "parameter_update_max": float((probe_local.detach().reshape(-1)[:4096] - before).abs().max()),
                "elapsed_seconds": time.perf_counter() - step_started,
                "npu_peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30,
                "npu_peak_reserved_gib": torch.npu.max_memory_reserved() / 2**30,
                "offload_mode": session.mode if session is not None else None,
            }
            check_update(step_index + 2, step_result["parameter_update_max"])
            result["extra_step_results"].append(step_result)
            result["steps"] += 1
            log(rank, "EXTRA_STEP " + json.dumps(step_result))
            if profiler is not None:
                profiler.step()
            # The custom ChunkLoss backward saves sizeable gradient buffers.
            # Release the step output and probe references before the next
            # forward so they cannot pin allocations across iterations.
            del step_loss, step_grad, step_grad_local
            write_progress()
        if validator is not None:
            result["correctness"] = validate_on_all_ranks(validator.finish)
        if args.extra_steps:
            result["status"] = "steps_complete"
    except BaseException as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        if validator is not None:
            result["correctness"] = validator.summary("failed")
        raise
    finally:
        if profiler is not None:
            profiler.stop()
        if session is not None:
            result.update(offload_stats)
            result["host_pool_allocated_gib"] = session.residency_manager._host_pool.total_allocated / 2**30
        result.update(
            elapsed_seconds=time.perf_counter() - started,
            npu_peak_allocated_gib=torch.npu.max_memory_allocated() / 2**30,
            npu_peak_reserved_gib=torch.npu.max_memory_reserved() / 2**30,
        )
        Path(args.profile_path).mkdir(parents=True, exist_ok=True)
        (Path(args.profile_path) / f"result_rank{rank}.json").write_text(json.dumps(result, indent=2))
        log(rank, json.dumps(result))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
