# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Real accelerator regressions for snapshot mutation and retained storage."""

import gc

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


@pytest.mark.parametrize("operation", ["add", "view", "out", "foreach"])
def test_reload_mutate_evict_roundtrip(operation):
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(1024 * 1024)
    source = torch.tensor([1., 2.], device=device)
    value = ShadowTensor(source, manager.bind(1, source), 1)
    # Keep the D2H snapshot while writing; exercise actual cross-stream copies.
    manager.copy_d2h(1)
    if operation == "add":
        value.add_(10)
    elif operation == "view":
        value.view(2).add_(10)
    elif operation == "out":
        torch.add(value, 10, out=value)
    else:
        torch._foreach_add_([value], 10)
    manager.copy_d2h(1)
    manager.release_device(1)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([11., 12.]))
    # Repeat with a real eviction and demand-page before the mutation.
    value.add_(10)
    manager.copy_d2h(1)
    manager.release_device(1)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([21., 22.]))
    manager.clear_runtime()


@pytest.mark.parametrize("keep_view", [False, True])
def test_live_shadow_survives_reset_and_host_reuse(keep_view):
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(1024 * 1024)
    executor = WarmupExecutor(manager, 1024 * 1024)
    value = executor.make_shadow(1, torch.tensor([1., 2.], device=device))
    if keep_view:
        view = value.view(2)
        del value
        gc.collect()
        value = view
    manager.copy_d2h(1)
    manager.release_device(1)
    executor.reset()
    # Acquiring the same allocation bucket must never modify a retained value.
    host = manager._host_pool.acquire(8)
    host.view(torch.float32).fill_(9)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([1., 2.]))
    manager._host_pool.release(host)


@pytest.mark.parametrize("change", ["operator", "shape", "truncated"])
def test_session_rejects_changed_execution(change):
    device = torch.accelerator.current_accelerator()
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    source = torch.ones(2, device=device)
    with session:
        torch.sin(source)
    message = {"operator": "operator differs", "shape": "input metadata", "truncated": "op count differs"}[change]
    changed_input = torch.ones(3, device=device) if change == "shape" else source
    with pytest.raises(RuntimeError, match=message):
        with session:
            if change == "operator":
                torch.cos(changed_input)
            elif change == "shape":
                torch.sin(changed_input)
    assert OffloadSession.get_active() is None


@pytest.mark.parametrize("offloaded", [False, True])
def test_final_release_keeps_output_with_changed_python_lifetime(offloaded):
    device = torch.accelerator.current_accelerator()
    manager = ResidencyManager(1024 * 1024)
    source = torch.tensor([1., 2.], device=device)
    value = ShadowTensor(source, manager.bind(1, source), 1)
    if offloaded:
        manager.copy_d2h(1)
    manager.release_device(1)
    manager.release_host(1)
    torch.testing.assert_close(value.resolve().cpu(), torch.tensor([1., 2.]))


def test_replay_can_return_loss_not_retained_in_warmup():
    device = torch.accelerator.current_accelerator()
    source = torch.ones(4, device=device, requires_grad=True)
    session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
    with session:
        source.square().sum().backward()
    source.grad = None
    with session:
        loss = source.square().sum()
        loss.backward()
    torch.testing.assert_close(loss.detach().cpu(), torch.tensor(4.))
    torch.testing.assert_close(source.grad.detach().cpu(), torch.full((4,), 2.))
