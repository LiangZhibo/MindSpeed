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
"""Regressions for stale snapshots, retained storage and invalid plan reuse."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.execution.schema import mutated_tensors
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


class BytePool:
    """Deterministic reuse of real CPU storage; no mocked tensor operations."""

    def __init__(self):
        self.free = []

    def acquire(self, size):
        return self.free.pop()[:size] if self.free else torch.empty(size, dtype=torch.uint8)

    def release(self, tensor, event=None):
        if event is not None:
            event.synchronize()
        self.free.append(tensor)


@pytest.fixture
def manager():
    result = ResidencyManager(1024)
    result._host_pool = BytePool()
    # Exercise the synchronous cross-device path without requiring an accelerator.
    result._copy_stream = SimpleNamespace(device=torch.device("meta"))
    return result


def shadow(manager, sid, value):
    return ShadowTensor(value, manager.bind(sid, value), sid)


def evict(manager, sid):
    manager.copy_d2h(sid)
    manager.release_device(sid)


@pytest.mark.parametrize("operation", ["add", "view", "out", "foreach"])
def test_mutation_after_reload_replaces_host_snapshot(manager, operation):
    value = shadow(manager, 1, torch.tensor([1., 2.]))
    evict(manager, 1)
    value.resolve()  # Keep both device data and an old host snapshot.
    if operation == "add":
        value.add_(10)
    elif operation == "view":
        value.view(2).add_(10)
    elif operation == "out":
        torch.add(value, 10, out=value)
    else:
        torch._foreach_add_([value], 10)
    evict(manager, 1)
    torch.testing.assert_close(value.resolve(), torch.tensor([11., 12.]))


def test_write_waits_for_d2h_and_keeps_h2d_lease(manager):
    value = shadow(manager, 1, torch.tensor([1., 2.]))
    manager.copy_d2h(1)
    buffer = value._buffer
    d2h, h2d = Mock(), Mock()
    buffer.host_event, buffer.device_event = d2h, h2d
    host = buffer.host_buffer
    with patch.object(buffer, "device_storage"), patch.object(manager._host_pool, "release") as release:
        buffer.prepare_for_write()
    d2h.synchronize.assert_called_once()
    release.assert_called_once_with(host, event=h2d)
    assert buffer.host_buffer is None


@pytest.mark.parametrize("keep_view", [False, True])
def test_reset_preserves_live_shadow_and_recycles_dead_storage(manager, keep_view):
    executor = WarmupExecutor(manager, 1024)
    value = executor.make_shadow(1, torch.tensor([1., 2.]))
    if keep_view:
        import gc
        view = value.view(2)
        del value
        gc.collect()
        value = view
    manager.bind(2, torch.tensor([3., 4.]))
    evict(manager, 1)
    evict(manager, 2)
    dead = manager._residency[2]
    executor.reset()
    assert dead.device_buffer is None  # Dead data must not be restored to device.
    for _ in range(2):
        manager._host_pool.acquire(8).view(torch.float32).fill_(9)
    torch.testing.assert_close(value.resolve(), torch.tensor([1., 2.]))
    assert value._buffer.host_buffer is None
    assert not manager._residency


def test_clear_without_executor_preserves_external_shadow(manager):
    value = shadow(manager, 1, torch.tensor([1., 2.]))
    evict(manager, 1)
    manager.clear_runtime()
    manager._host_pool.acquire(8).view(torch.float32).fill_(9)
    torch.testing.assert_close(value.resolve(), torch.tensor([1., 2.]))


@pytest.mark.parametrize("case", ["list", "out", "keyword"])
def test_mutable_schema_binds_all_tensor_leaves(manager, case):
    a, b = torch.ones(2), torch.ones(2)
    if case == "list":
        func, args, kwargs, written = torch.ops.aten._foreach_add_.Scalar, ([a, b], 1), {}, [a, b]
    elif case == "out":
        func, args, kwargs, written = torch.ops.aten.add.out, (a, 1), {"out": b}, [b]
    else:
        func, args, kwargs, written = torch.ops.aten.fill_.Scalar, (), {"self": a, "value": 2}, [a]
    assert [id(t) for t in mutated_tensors(func, args, kwargs)] == [id(t) for t in written]
    executor = WarmupExecutor(manager, 1024)
    executor._timer = Mock()
    executor._timer.stop.return_value = 0.
    ids = {id(a): 1, id(b): 2}
    executor._tracker = SimpleNamespace(
        get_activation_sid=lambda t: ids.get(id(t)), register_op_activations=lambda *args: None,
    )
    executor.on_op_begin(func, args, kwargs)
    # No actual op needed: we are testing a mutable op with no returned alias.
    executor.on_op_end(None)
    writes = {a.storage_id for a in executor._ops[0].accesses if a.kind.name == "WRITE"}
    assert writes == {ids[id(t)] for t in written}


def recorded(manager, func, args, kwargs=None):
    executor = WarmupExecutor(manager, 1024)
    executor._timer = Mock()
    executor._timer.stop.return_value = 0.
    executor.dispatch(func, args, kwargs or {})
    return ReplayExecutor(manager, ResidencySchedule(), executor._guide)


def test_replay_rejects_operator_before_schedule_actions(manager):
    replay = recorded(manager, torch.ops.aten.sin.default, (torch.ones(2),))
    replay._schedule.add_pre(0, ResidencyActionType.COPY_H2D, 999)
    with pytest.raises(RuntimeError, match="operator differs"):
        replay.dispatch(torch.ops.aten.cos.default, (torch.ones(2),), {})


@pytest.mark.parametrize("change", ["shape", "dtype", "stride", "scalar", "alias", "structure"])
def test_replay_rejects_changed_input_contract(manager, change):
    def op(*args):
        return torch.ones(2)
    a = torch.ones(2, 2)
    args = (a, a, 1)
    changed = {
        "shape": (torch.ones(3, 2), a, 1),
        "dtype": (a.double(), a, 1),
        "stride": (a.t(), a, 1),
        "scalar": (a, a, 2),
        "alias": (a, a.clone(), 1),
        "structure": ([a, a], 1),
    }[change]
    replay = recorded(manager, op, args)
    with pytest.raises(RuntimeError, match="input metadata or bindings differ"):
        replay.dispatch(op, changed, {})


@pytest.mark.parametrize("change", ["shape", "dtype", "alias", "structure"])
def test_replay_rejects_changed_output_contract(manager, change):
    changed = False
    def op(x):
        if not changed:
            return (x, x)
        return {
            "shape": (torch.ones(3), x), "dtype": (x.double(), x),
            "alias": (x.clone(), x), "structure": [x, x],
        }[change]
    replay = recorded(manager, op, (torch.ones(2),))
    changed = True
    with pytest.raises(RuntimeError, match="output metadata or aliases differ"):
        replay.dispatch(op, (torch.ones(2),), {})


def test_replay_rejects_wrong_logical_activation_binding(manager):
    value = shadow(manager, 1, torch.ones(2))
    replay = recorded(manager, torch.ops.aten.sin.default, (value,))
    other = shadow(manager, 2, torch.ones(2))
    with pytest.raises(RuntimeError, match="input metadata or bindings differ"):
        replay.dispatch(torch.ops.aten.sin.default, (other,), {})


def test_session_rejects_truncated_step_and_cleans_up():
    with patch("mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer"), \
         patch("mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
               return_value=(10., 10.)), \
         patch.object(ResidencyManager, "wait_for_transfers"), \
         patch.object(ResidencyManager, "sync_all_transfers"):
        session = OffloadSession(OffloadConfig(max_resident_activation_mb=1))
        with session:
            torch.ones(2)
        with pytest.raises(RuntimeError, match="op count differs"):
            with session:
                pass
        assert session.get_active() is None
        assert session.executor.op_idx == -1


def test_warmup_plan_retains_views_not_created_by_executor(manager):
    executor = WarmupExecutor(manager, 1024)
    value = executor.make_shadow(1, torch.tensor([1., 2.]))
    view = value.view(2)
    del value
    import gc
    gc.collect()
    with patch("mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth",
               return_value=(10., 10.)):
        trace, _ = executor.finish()
    assert 1 in trace.retained_sids
    torch.testing.assert_close(view.resolve(), torch.tensor([1., 2.]))


@pytest.mark.parametrize("offloaded", [False, True])
def test_last_use_release_preserves_newly_retained_output(manager, offloaded):
    value = shadow(manager, 1, torch.tensor([1., 2.]))
    if offloaded:
        manager.copy_d2h(1)
    manager.release_device(1)
    manager.release_host(1)
    if offloaded:
        manager._host_pool.acquire(8).view(torch.float32).fill_(9)
    torch.testing.assert_close(value.resolve(), torch.tensor([1., 2.]))
    assert value._buffer.host_buffer is None
