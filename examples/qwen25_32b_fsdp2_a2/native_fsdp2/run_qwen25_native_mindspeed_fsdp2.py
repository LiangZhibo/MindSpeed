#!/usr/bin/env python3
"""Run a cropped Qwen step with MindSpeed FSDP2 on two NPUs."""

import argparse
from contextlib import nullcontext
from functools import wraps
import inspect
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
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from qwen_chunk_loss import calculate_lm_loss, chunk_loss
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession, skip_offload
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner

import megatron_adaptor  # noqa: F401
import mindspeed.megatron_adaptor  # noqa: F401
from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig


def apply_hccl_premul_sum_patch():
    """Translate PyTorch's scaled reduction into HCCL-supported operations."""
    from torch.distributed.distributed_c10d import ReduceOp

    def wrap(operation, output_name):
        @wraps(operation)
        def wrapped(*args, **kwargs):
            factor = None
            if kwargs.get("op") == ReduceOp.PREMUL_SUM:
                factor = kwargs["op"].__getstate__()[1]
                kwargs["op"] = ReduceOp.SUM
            handle = operation(*args, **kwargs)
            if handle is not None:
                handle.wait()
            if factor is not None:
                output = args[0] if args else kwargs[output_name]
                output.data.mul_(factor)
            return handle
        return wrapped

    dist.all_reduce = wrap(dist.all_reduce, "tensor")
    dist.reduce_scatter = wrap(dist.reduce_scatter, "output")
    dist.reduce_scatter_tensor = wrap(dist.reduce_scatter_tensor, "output")

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
    parser.add_argument("--offload-communication-masking", action="store_true")
    parser.add_argument("--extra-steps", type=int, default=0)
    parser.add_argument("--use-torch-fsdp2", action="store_true", required=True)
    parser.add_argument("--fsdp2-config-path", required=True)
    return parser.parse_args()


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
    training_model = TrainingModel(model, args.loss_chunk)
    transformer_config = TransformerConfig(
        num_layers=args.layers,
        hidden_size=model.config.hidden_size,
        num_attention_heads=model.config.num_attention_heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    ddp_config = DistributedDataParallelConfig()
    engine = TorchFullyShardedDataParallel(
        transformer_config,
        ddp_config,
        training_model,
        sub_modules_to_wrap={Qwen2DecoderLayer, torch.nn.Embedding, ChunkedHead},
        process_group=dist.group.WORLD,
    )
    if recompute_layers:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl, checkpoint_wrapper,
        )
        for index in range(recompute_layers):
            training_model.model.layers[index] = checkpoint_wrapper(
                training_model.model.layers[index], checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
    if args.hyper_offload:
        for layer in engine.module.model.layers[:recompute_layers]:
            layer.forward = skip_offload(layer.forward)
    log(
        rank,
        "ENGINE_READY "
        f"parameters={parameter_count} layers={args.layers} fsdp_size=2 "
        f"cpu_offload=False activation_swap=False recompute_layers={recompute_layers} "
        f"hyper_offload={args.hyper_offload}",
    )
    import mindspeed
    adapter_path = Path(inspect.getfile(TorchFullyShardedDataParallel.__init__)).resolve()
    mindspeed_root = Path(mindspeed.__file__).resolve().parent
    expected_adapter = (mindspeed_root / "core/distributed/torch_fully_sharded_data_parallel/"
                        "torch_fully_sharded_data_parallel_adaptor.py")
    repo_root = Path(__file__).resolve().parents[3]
    if repo_root not in mindspeed_root.parents:
        raise RuntimeError(f"MindSpeed was imported outside this checkout: {mindspeed_root}")
    if adapter_path != expected_adapter.resolve():
        raise RuntimeError(f"wrong MindSpeed FSDP adapter: {adapter_path}")
    if any(name == "mindspeed.fsdp" or name.startswith("mindspeed.fsdp.") for name in sys.modules):
        raise RuntimeError("legacy mindspeed.fsdp package loaded")
    log(rank, f"NATIVE_FSDP_PROOF mindspeed={mindspeed.__file__} adapter={adapter_path}")
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


def main():
    args = parse_args()
    from mindspeed.args_utils import get_mindspeed_args, set_full_args
    full_args = vars(get_mindspeed_args(get_defaults=True)).copy()
    full_args.update(vars(args))
    full_args["init_model_with_meta_device"] = False
    set_full_args(argparse.Namespace(**full_args))
    if args.extra_steps < 0:
        raise ValueError("--extra-steps must be nonnegative")
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(16)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise RuntimeError(f"expected exactly two NPU workers, got world size {dist.get_world_size()}")
    apply_hccl_premul_sum_patch()
    log(rank, f"ENV torch={torch.__version__} torch_npu={torch_npu.__version__} visible={os.environ['ASCEND_RT_VISIBLE_DEVICES']}")
    engine, vocab_size, parameter_count = make_engine(args, rank)
    engine.train()
    optimizer = torch.optim.SGD(engine.parameters(), lr=1.0e-2)
    probe_parameter = engine.module.model.layers[0].mlp.down_proj.weight
    probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
    probe_before = probe_local.detach().reshape(-1)[:4096].clone()
    input_ids = torch.randint(vocab_size, (1, args.seq_length), device="npu")
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    profiler = make_profile(args)
    session = None
    offload_stats = {"d2h_calls": 0, "d2h_gib": 0.0}
    if args.hyper_offload:
        session = OffloadSession(OffloadConfig(
            max_resident_activation_mb=args.offload_device_mb,
            max_offload_activation_mb=args.offload_host_mb,
            planner=(PeakResidencyPlanner(enable_communication_masking=args.offload_communication_masking)
                     if args.offload_planner == "peak" else GreedyResidencyPlanner()),
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
        "offload_communication_masking": args.offload_communication_masking if args.hyper_offload else None,
        "optimizer": "SGD",
        "learning_rate": 1.0e-2,
    }

    def write_progress():
        snapshot = dict(result)
        snapshot["elapsed_seconds"] = time.perf_counter() - started
        if session is not None:
            snapshot.update(offload_stats)
            snapshot["host_pool_allocated_gib"] = session.residency_manager._host_pool.total_allocated / 2**30
            snapshot["host_pool_pageable_gib"] = session.residency_manager._host_pool.pageable_allocated / 2**30
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
        optimizer.step()
        if profiler is not None:
            profiler.step()
        probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
        update = float((probe_local.detach().reshape(-1)[:4096] - probe_before).abs().max())
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
        if args.extra_steps:
            result["status"] = "steps_complete"
    except BaseException as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if profiler is not None:
            profiler.stop()
        if session is not None:
            result.update(offload_stats)
            result["host_pool_allocated_gib"] = session.residency_manager._host_pool.total_allocated / 2**30
            result["host_pool_pageable_gib"] = session.residency_manager._host_pool.pageable_allocated / 2**30
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
