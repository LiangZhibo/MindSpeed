#!/usr/bin/env python3
"""Phase-one NPU checks and matched raw-Offload/compressed-Offload small MLP.

No recomputation, no Qwen, no chunk pipeline. Measurements synchronize step
boundaries. Numerical snapshots are taken outside the measured interval.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa: F401

from mindspeed.core.memory.compression import SplitZipCodec
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.planning import PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


def codec_checks(library, device):
    codec = SplitZipCodec(library)
    manager = ResidencyManager(128 * 1024**2, codec=codec)
    results = []
    for sid, elements in enumerate((1, 8191, 8192, 8193, 65536, 1048576)):
        # Every BF16 bit pattern, including NaN payloads, signed zeros and Inf.
        bits = (torch.arange(elements, dtype=torch.int64) % 65536).to(torch.int16)
        tensor = bits.view(torch.bfloat16).to(device)
        buffer = manager.bind(sid, tensor)
        shadow = ShadowTensor(tensor[::2], buffer, sid)
        manager.set_transfer_node(sid, "post", "roundtrip")
        manager.copy_d2h(sid)
        packet_bytes = buffer.host_buffer.numel()
        assert buffer.host_encoding is not None
        manager.release_device(sid)
        assert buffer.device_buffer is None
        if sid % 2:
            manager.copy_h2d(sid)
        restored = shadow.resolve().view(torch.int16).cpu()
        assert torch.equal(restored, bits[::2])
        full = torch.empty(0, dtype=torch.int16, device=device).set_(buffer.device_storage(), 0, (elements,))
        assert torch.equal(full.cpu(), bits)
        # A mutation must discard the compressed snapshot and preserve new data.
        buffer.prepare_for_write()
        full.fill_(123)
        manager.copy_d2h(sid)
        manager.release_device(sid)
        manager.copy_h2d(sid)
        full = torch.empty(0, dtype=torch.int16, device=device).set_(buffer.device_storage(), 0, (elements,))
        restored_bits = full.cpu()
        assert torch.equal(restored_bits, torch.full_like(bits, 123)), (
            elements, restored_bits[:16].tolist(), int((restored_bits != 123).sum()),
        )
        results.append(dict(elements=elements, packet_bytes=packet_bytes, bit_exact=True))
        del shadow, full
        manager.clear_runtime(retained_sids=set())
    # Repeated H2D from the same Host snapshot; borrowed consumer aliases survive.
    tensor = torch.randn(8195, dtype=torch.bfloat16, device=device)
    buffer = manager.bind(99, tensor)
    shadow = ShadowTensor(tensor[2::3], buffer, 99)
    expected = tensor[2::3].cpu()
    manager.copy_d2h(99)
    for _ in range(3):
        manager.release_device(99)
        torch.testing.assert_close(shadow.resolve().cpu(), expected, rtol=0, atol=0)
    manager.clear_runtime()
    torch.testing.assert_close(shadow.resolve().cpu(), expected, rtol=0, atol=0)
    # Unsupported dtypes and excluded nodes take exactly the existing raw path.
    excluded = SplitZipCodec(library, nodes=set())
    raw_manager = ResidencyManager(1024**2, codec=excluded)
    for dtype in (torch.bfloat16, torch.float32, torch.int64):
        t = torch.ones(128, dtype=dtype, device=device)
        b = raw_manager.bind(1, t)
        raw_manager.copy_d2h(1)
        assert b.host_encoding is None
        raw_manager.release_device(1)
        raw_manager.copy_h2d(1)
        raw_manager.clear_runtime()
    return dict(roundtrips=results, stats=dict(codec.stats), raw_fallback=dict(excluded.stats))


def make_model(width):
    return torch.nn.Sequential(
        torch.nn.Linear(64, width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, 16),
    )


def run_training(args, initial, inputs, codec):
    torch.npu.empty_cache()
    model = deepcopy(initial).to(device=args.device, dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    session = OffloadSession(OffloadConfig(
        max_resident_activation_mb=args.budget_mb, max_offload_activation_mb=getattr(args, "host_mb", 1024),
        planner=PeakResidencyPlanner(), codec=codec,
    ))
    if codec:
        codec.prepare(torch.device(args.device), session.residency_manager._get_copy_stream())
    rows, snapshots = [], []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x = inputs[step].to(args.device).requires_grad_()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        before = Counter(codec.stats) if codec else Counter()
        start = time.perf_counter()
        with session:
            loss = model(x).float().square().mean()
            loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        assert torch.isfinite(loss)
        rows.append(dict(step=step, mode=session.mode, milliseconds=elapsed,
                         peak_allocated=torch.npu.max_memory_allocated(),
                         codec=dict(Counter(codec.stats) - before) if codec else {}))
        snapshots.append(dict(loss=loss.detach().cpu(), input_grad=x.grad.detach().cpu(),
                              grads=[p.grad.detach().cpu().clone() for p in model.parameters()],
                              parameters=[p.detach().cpu().clone() for p in model.parameters()]))
        del x, loss
    # Snapshot the plan for exact equality across routes: compression must not
    # change the logical storage sizes or the eviction/prefetch plan.
    schedule = session.executor._schedule
    plan = [(phase, op, a.storage_id, a.kind.name)
            for phase, mapping in (("pre", schedule.pre), ("post", schedule.post))
            for op, actions in sorted(mapping.items()) for a in actions]
    report = dict(rows=rows, median_replay_ms=statistics.median(
                      r["milliseconds"] for r in rows[getattr(args, "warmup", 2):]),
                  plan=plan, stats=dict(codec.stats) if codec else {},
                  nodes={k:dict(v) for k,v in codec.node_stats.items()} if codec else {},
                  host_pool_allocated=session.residency_manager._host_pool.total_allocated,
                  host_pool_pageable=session.residency_manager._host_pool.pageable_allocated)
    del session, optimizer, model
    gc.collect()
    torch.npu.synchronize()
    if codec is not None and hasattr(codec, "close"):
        codec.close()
    return report, snapshots


def compare(reference, actual):
    assert len(reference) == len(actual)
    for expected, got in zip(reference, actual):
        for name in ("loss", "input_grad"):
            torch.testing.assert_close(expected[name], got[name], rtol=0, atol=0)
        for name in ("grads", "parameters"):
            assert len(expected[name]) == len(got[name])
            for a, b in zip(expected[name], got[name]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--budget-mb", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--host-mb", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("at least three steps required")
    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    torch.manual_seed(20261029)
    report = dict(settings=vars(args), torch=torch.__version__, torch_npu=torch_npu.__version__)
    report["codec_checks"] = codec_checks(args.library, torch.device(args.device))
    initial = make_model(args.width).to(torch.bfloat16)
    inputs = [torch.randn(args.batch, 64, dtype=torch.bfloat16) for _ in range(args.steps)]
    report["raw"], baseline = run_training(args, initial, inputs, None)
    codec = SplitZipCodec(args.library)
    report["compressed"], snapshots = run_training(args, initial, inputs, codec)
    compare(baseline, snapshots)
    assert report["raw"]["plan"] == report["compressed"]["plan"]
    assert codec.stats["encode_calls"] and codec.stats["decode_calls"]
    # Do not charge the next route for the previous route's persistent scratch.
    del codec
    gc.collect()
    nodes = {(op, phase) for phase, op, _, kind in report["raw"]["plan"] if kind == "COPY_D2H"}
    selected = set(sorted(nodes)[::2])
    assert selected and len(selected) < len(nodes), "small case must exercise selective compression"
    partial_codec = SplitZipCodec(args.library, nodes=selected)
    report["selected_nodes"] = sorted(selected)
    report["selected"], snapshots = run_training(args, initial, inputs, partial_codec)
    compare(baseline, snapshots)
    assert report["raw"]["plan"] == report["selected"]["plan"]
    assert partial_codec.stats["encode_calls"] and partial_codec.stats["skip_node"]
    report["bit_exact_training"] = True
    report["same_offload_plan"] = True
    report["compressed_speedup_percent"] = 100*(1-report["compressed"]["median_replay_ms"]/report["raw"]["median_replay_ms"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ("codec_checks", "raw", "compressed", "selected")}, indent=2))
    for name in ("raw", "compressed", "selected"):
        print(name, report[name]["median_replay_ms"], report[name]["stats"], flush=True)


if __name__ == "__main__":
    main()
