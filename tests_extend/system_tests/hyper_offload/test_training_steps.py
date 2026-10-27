# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Repeated training verifies fresh values, optimizer updates and host reuse."""

from copy import deepcopy
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("planner_name", ["greedy", "peak"])
def test_repeated_training_matches_independent_baseline(dtype, planner_name, record_property):
    device = torch.accelerator.current_accelerator()
    torch.manual_seed(17)
    offloaded = nn.Sequential(
        nn.Linear(64, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 16),
    ).to(device=device, dtype=dtype)
    reference = deepcopy(offloaded)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(offloaded.parameters(), reference.parameters()))
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                  for model in (reference, offloaded)]
    planner = (PeakResidencyPlanner(enable_communication_masking=True)
               if planner_name == "peak" else GreedyResidencyPlanner())
    session = OffloadSession(OffloadConfig(
        max_resident_activation_mb=3, max_offload_activation_mb=64, planner=planner,
    ))
    rtol, atol = (0.02, 0.002) if dtype == torch.bfloat16 else (1e-4, 1e-6)
    host_sizes = []
    for step in range(10):
        inputs = torch.randn(1024, 64, device=device, dtype=dtype)
        snapshots = []
        for model, optimizer, context in zip(
            (reference, offloaded), optimizers, (nullcontext(), session), strict=True
        ):
            optimizer.zero_grad(set_to_none=True)
            with context:
                loss = model(inputs).float().square().mean()
                loss.backward()
            snapshots.append((loss.detach().cpu(), [p.grad.detach().cpu().clone() for p in model.parameters()]))
            optimizer.step()
        torch.testing.assert_close(snapshots[0][0], snapshots[1][0], rtol=rtol, atol=atol)
        for expected, actual in zip(snapshots[0][1], snapshots[1][1], strict=True):
            torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)
        for expected, actual in zip(reference.parameters(), offloaded.parameters(), strict=True):
            torch.testing.assert_close(expected, actual, rtol=rtol, atol=atol)
        host_sizes.append(session.residency_manager._host_pool.total_allocated)
    assert session.mode == "replay"
    assert 0 < host_sizes[-1] <= 64 * 1024**2
    assert max(host_sizes[3:]) == min(host_sizes[3:]), "host pool keeps growing after warmup"
    record_property("training_steps", 10)
    record_property("host_pool_bytes", host_sizes[-1])
