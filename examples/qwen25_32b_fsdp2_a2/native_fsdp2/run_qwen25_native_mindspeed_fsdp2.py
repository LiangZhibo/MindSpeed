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
from transformers import AutoConfig, AutoModelForCausalLM

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
            dict(shift_labels=part, alpha=normalizer, reduction="sum") for part in labels.split(self.chunk_size, dim=1)
        ]
        return opaque_chunk_loss(hidden, self.weight, None, calculate_lm_loss, loss_kwargs, self.chunk_size)


class TrainingModel(torch.nn.Module):
    def __init__(self, hf_model, chunk_size):
        super().__init__()
        self.model = getattr(hf_model, hf_model.base_model_prefix)
        self.lm_head = ChunkedHead(hf_model.get_output_embeddings(), chunk_size)

    def forward(self, input_ids, labels):
        hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
        return self.lm_head(hidden[:, :-1, :], labels[:, 1:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", help="Local random-weight HF config for architecture qualification")
    parser.add_argument("--model-init-seed", type=int, default=1234)
    parser.add_argument("--offload-region-layout", choices=("legacy_qwen", "gated", "parallel_mlp"), default="gated")
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
    parser.add_argument("--hyper-offload-autotune", action="store_true")
    parser.add_argument("--offload-tuning-method", choices=("legacy", "measured"), default="measured")
    parser.add_argument("--offload-transfer-size-calibration", action="store_true",
                        help="Calibrate observed transfer sizes during bounded warmup")
    parser.add_argument("--offload-joint-refinement", action="store_true",
                        help="Test a joint region/cadence refinement in the second candidate slot")
    parser.add_argument("--offload-tuning-steps", type=int, default=10)
    parser.add_argument("--offload-tuning-candidates", type=int, default=3)
    parser.add_argument("--offload-window-probe-multiplier", type=float, default=8.0)
    parser.add_argument("--offload-window-probe-unlimited-prefetch", action="store_true")
    parser.add_argument(
        "--offload-tuning-probe",
        choices=(
            "norm",
            "mlp",
            "attention_resident",
            "suffix_resident",
            "mlp_resident",
            "compute_resident",
            "budget_resident",
            "stream_resident",
            "cache_resident",
            "pointwise_resident",
            "pointwise_window_resident",
            "gate_recompute_resident",
            "placement_resident",
            "window_resident",
            "release_resident",
            "partial_compute_resident",
            "gate_compute_resident",
            "phase_trim_resident",
            "phase_cache_resident",
        ),
        default="norm",
    )
    parser.add_argument("--offload-resident-mlp-fraction", type=float, default=0.5)
    parser.add_argument("--offload-tuning-device-peak-mb", type=int)
    parser.add_argument("--extra-steps", type=int, default=0)
    parser.add_argument("--use-torch-fsdp2", action="store_true", required=True)
    parser.add_argument("--fsdp2-config-path", required=True)
    return parser.parse_args()


def log(rank, message):
    print(f"[rank {rank}] {message}", flush=True)


def crop_layers(model, layers):
    base = getattr(model, model.base_model_prefix)
    source_layers = len(base.layers)
    if layers < 1 or layers > source_layers:
        raise ValueError(f"--layers must be in [1, {source_layers}], got {layers}")
    if layers < source_layers:
        base.layers = torch.nn.ModuleList(list(base.layers[:layers]))
        model.config.num_hidden_layers = layers
        base.config.num_hidden_layers = layers
    return model


def make_engine(args, rank):
    if args.model_config:
        config_values = json.loads(Path(args.model_config).read_text())
        model_type = config_values.pop('model_type')
        config = AutoConfig.for_model(model_type, **config_values)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.model_init_seed)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                                    attn_implementation='sdpa')
    else:
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
        sub_modules_to_wrap={type(training_model.model.layers[0]), torch.nn.Embedding, ChunkedHead},
        process_group=dist.group.WORLD,
    )
    if recompute_layers and not args.hyper_offload_autotune:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        for index in range(recompute_layers):
            training_model.model.layers[index] = checkpoint_wrapper(
                training_model.model.layers[index], checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
    if args.hyper_offload and not args.hyper_offload_autotune:
        for layer in engine.module.model.layers[:recompute_layers]:
            layer.forward = skip_offload(layer.forward)
    if args.hyper_offload_autotune:
        from mindspeed.core.memory.hyper_offload.tuning.regions import qwen_recompute_regions, decoder_recompute_regions

        engine._hyper_offload_regions = (
            qwen_recompute_regions(engine.module.model.layers) if args.offload_region_layout == 'legacy_qwen'
            else decoder_recompute_regions(engine.module.model.layers, layout=args.offload_region_layout)
        )
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
    expected_adapter = (
        mindspeed_root / "core/distributed/torch_fully_sharded_data_parallel/"
        "torch_fully_sharded_data_parallel_adaptor.py"
    )
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
    if args.hyper_offload_autotune and not args.hyper_offload:
        raise ValueError("joint tuning requires --hyper-offload")
    if (
        args.hyper_offload_autotune
        and (args.offload_tuning_method == 'measured' or args.offload_tuning_probe.endswith('_resident'))
        and args.offload_tuning_device_peak_mb is None
    ):
        raise ValueError('resident-region tuning requires an explicit whole-device peak limit')
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.set_num_threads(16)
    torch.manual_seed(1234 + rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 2:
        raise RuntimeError(f"expected exactly two NPU workers, got world size {dist.get_world_size()}")
    apply_hccl_premul_sum_patch()
    log(
        rank,
        f"ENV torch={torch.__version__} torch_npu={torch_npu.__version__} visible={os.environ['ASCEND_RT_VISIBLE_DEVICES']}",
    )
    engine, vocab_size, parameter_count = make_engine(args, rank)
    engine.train()
    optimizer = torch.optim.SGD(engine.parameters(), lr=1.0e-2)
    first_mlp = engine.module.model.layers[0].mlp
    probe_parameter = (first_mlp.down_proj.weight if hasattr(first_mlp, 'down_proj')
                       else next(p for p in first_mlp.parameters() if p.requires_grad and p.ndim == 2))
    probe_local = probe_parameter.to_local() if isinstance(probe_parameter, DTensor) else probe_parameter
    probe_before = probe_local.detach().reshape(-1)[:4096].clone()
    input_ids = torch.randint(vocab_size, (1, args.seq_length), device="npu")
    dist.barrier()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    profiler = make_profile(args)
    session_setup_started = time.perf_counter()
    session = None
    offload_stats = {"d2h_calls": 0, "d2h_gib": 0.0}
    if args.hyper_offload:
        if args.hyper_offload_autotune:
            from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, JointTuningController, TuningLimits
            from legacy_tuning import QwenRegionNeighborhood
            from mindspeed.core.memory.hyper_offload.tuning.session import JointOffloadSession

            count = (
                args.recompute_layers if args.recompute_layers is not None else (args.layers if args.recompute else 0)
            )
            initial = JointPolicy(0,args.offload_device_mb,args.offload_communication_masking,
                                  recompute_regions=tuple(str(i) for i in range(count)))
            limits = TuningLimits(max_steps=args.offload_tuning_steps,max_candidates=args.offload_tuning_candidates,
                device_peak_bytes=None if args.offload_tuning_device_peak_mb is None else args.offload_tuning_device_peak_mb * 1024**2,
                host_peak_bytes=args.offload_host_mb * 1024**2)
            if args.offload_tuning_method == 'measured':
                if args.offload_region_layout == 'legacy_qwen':
                    raise ValueError('measured tuning requires an explicit gated or parallel_mlp structural adapter')
                from mindspeed.core.memory.hyper_offload.tuning.measured_session import MeasuredOffloadSession
                session = MeasuredOffloadSession(initial,model=engine,regions=engine._hyper_offload_regions,
                                                limits=limits,host_mb=args.offload_host_mb,
                                                transfer_size_calibration=args.offload_transfer_size_calibration,
                                                joint_refinement=args.offload_joint_refinement)
            else:
                controller = JointTuningController(initial,QwenRegionNeighborhood(
                    args.layers,count,probe=args.offload_tuning_probe,
                    resident_mlp_fraction=args.offload_resident_mlp_fraction,
                    window_probe_multiplier=args.offload_window_probe_multiplier,
                    window_probe_limit_prefetch=not args.offload_window_probe_unlimited_prefetch),limits)
                session = JointOffloadSession(controller,apply_policy=engine._hyper_offload_regions.apply_policy,
                                              host_mb=args.offload_host_mb)
        else:
            session = OffloadSession(
                OffloadConfig(
                    max_resident_activation_mb=args.offload_device_mb,
                    max_offload_activation_mb=args.offload_host_mb,
                    planner=(
                        PeakResidencyPlanner(enable_communication_masking=args.offload_communication_masking)
                        if args.offload_planner == "peak"
                        else GreedyResidencyPlanner()
                    ),
                )
            )
        manager = session.residency_manager
        original_copy_d2h = manager.copy_d2h

        def counted_copy_d2h(storage_id):
            size = manager.device_resident_size(storage_id) or 0
            original_copy_d2h(storage_id)
            offload_stats["d2h_calls"] += 1
            offload_stats["d2h_gib"] += size / 2**30

        manager.copy_d2h = counted_copy_d2h
    started = time.perf_counter()
    session_setup_seconds = started - session_setup_started
    result = {
        "rank": rank,
        "world_size": 2,
        "session_setup_seconds": session_setup_seconds,
        "timing_boundary": "after shared model/data initialization; session setup reported separately",
        "parameter_count": parameter_count,
        "layers": args.layers,
        "seq_length": args.seq_length,
        "steps": 0,
        "fsdp_size": 2,
        "cpu_offload": False,
        "activation_swap": False,
        "recompute": (
            args.recompute_layers if args.recompute_layers is not None else (args.layers if args.recompute else 0)
        )
        > 0,
        "recompute_layers": args.recompute_layers
        if args.recompute_layers is not None
        else (args.layers if args.recompute else 0),
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

    observation_and_consensus_seconds = 0.0

    def observe_tuning(elapsed, peak_bytes):
        nonlocal observation_and_consensus_seconds
        if not args.hyper_offload_autotune or session.stable:
            return
        observation_started = time.perf_counter()
        from dataclasses import asdict
        from mindspeed.core.memory.hyper_offload.tuning import StepObservation
        from mindspeed.core.memory.hyper_offload.tuning.consensus import decision_fingerprint

        # Outside the measured interval; one consistent policy on every rank.
        seconds = torch.tensor(elapsed, dtype=torch.float32, device='npu')
        memory = torch.tensor(
            [peak_bytes, session.residency_manager._host_pool.total_allocated],
            dtype=torch.int64,
            device='npu',
        )
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        peak, host = memory.cpu().tolist()
        session.observe(StepObservation(seconds.cpu().item(), peak, host))
        policy = session.controller.policy
        decision = torch.tensor(
            decision_fingerprint(session.controller),
            dtype=torch.int64,
            device='npu',
        )
        minimum, maximum = decision.clone(), decision.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise RuntimeError('joint tuning decisions diverged across ranks; refusing a mismatched schedule')
        observation_and_consensus_seconds += time.perf_counter() - observation_started
        result['joint_tuning'] = dict(
            observation_and_consensus_seconds=observation_and_consensus_seconds,
            switch_seconds=session.switch_seconds,
            timing_events=session.timing_events,
            timing_event_count=len(session.timing_events),
            cache_reclaim_stats=dict(session.cache_reclaim_stats),
            cache_reclaim_observation_enabled=session.record_cache_reclaim,
            proposal_compute_seconds=getattr(session,'proposal_compute_seconds',None),
            proposal_consensus_seconds=getattr(session,'proposal_consensus_seconds',None),
            phase_sample_count=len((getattr(session,'phase_report',None) or {}).get('samples',())),
            region_observation_count=len((getattr(session,'region_report',None) or {}).get('regions',())),
            switch_drain_seconds=session.switch_drain_seconds,
            apply_regions_seconds=session.apply_regions_seconds,
            codec_prepare_seconds=session.codec_prepare_seconds,
            warmup_context_seconds=session.warmup_context_seconds,
            warmup_finish_seconds=session.warmup_finish_seconds,
            phase=session.controller.phase,
            steps=session.controller.steps,
            frozen_at_step=session.controller.frozen_at_step,
            policy=asdict(session.controller.policy),
            search_seconds=session.controller.search_seconds,
            confirmation_target=session.controller.confirmation_target,
            confirmation_fallback=session.controller.confirmation_fallback,
            selection_log=session.controller.selection_log,
            plan_seconds=session.plan_seconds,
            plan_builds=session.plan_builds,
            trace_collections=session.trace_collections,
            post_offload_trim_calls=session.post_offload_trim_calls,
            post_offload_trim_seconds=session.post_offload_trim_seconds,
            trials=[asdict(trial) for trial in session.controller.trials],
        )

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
            status="one_step_complete",
            steps=1,
            loss=float(loss.item()),
            probe_grad_max=grad_max,
            parameter_update_max=update,
            first_step_elapsed_seconds=time.perf_counter() - started,
            first_step_npu_peak_allocated_gib=torch.npu.max_memory_allocated() / 2**30,
            first_step_npu_peak_reserved_gib=torch.npu.max_memory_reserved() / 2**30,
        )
        if session is not None and session.mode == "replay":
            schedule = session.executor._schedule
            actions = [
                action for phase in (schedule.pre, schedule.post) for group in phase.values() for action in group
            ]
            result["offload_plan_actions"] = {
                kind: sum(action.kind.name == kind for action in actions)
                for kind in ("COPY_D2H", "COPY_H2D", "RELEASE_DEVICE", "RELEASE_HOST")
            }
        # Do not keep the custom ChunkLoss graph or its probe gradient alive
        # while later training steps allocate their working buffers.
        del loss, probe_grad, probe_grad_local
        observe_tuning(result['first_step_elapsed_seconds'], torch.npu.max_memory_allocated())
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
                "cache_trim_seconds": session.last_trim_seconds if session is not None else 0.0,
            }
            if args.hyper_offload_autotune:
                from dataclasses import asdict

                step_result['joint_policy'] = asdict(session.executed_policy)
                step_result['tuning_phase'] = session.controller.phase
                step_result['entry_trim_performed'] = session.last_entry_trim_performed
                step_result['post_offload_trim_calls'] = session.post_offload_trim_calls
                step_result['post_offload_trim_seconds'] = session.post_offload_trim_seconds
            result["extra_step_results"].append(step_result)
            result["steps"] += 1
            log(rank, "EXTRA_STEP " + json.dumps(step_result))
            if profiler is not None:
                profiler.step()
            # The custom ChunkLoss backward saves sizeable gradient buffers.
            # Release the step output and probe references before the next
            # forward so they cannot pin allocations across iterations.
            del step_loss, step_grad, step_grad_local
            observe_tuning(step_result['elapsed_seconds'], torch.npu.max_memory_allocated())
            step_result['cumulative_step_wall_seconds'] = time.perf_counter() - started
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
