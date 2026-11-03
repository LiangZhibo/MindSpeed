"""Metadata is part of the codec selector contract, not every op's state."""
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.compression import TransferNode
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.ir.replay import OpGuide
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType as Action, ResidencySchedule
from mindspeed.core.memory.hyper_offload.runtime.residency import PhysicalBuffer
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.execution.test_replay import _RecordingFake
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.execution.test_warmup import _FakeResidencyManager


class ReplayManager(_RecordingFake):
    def __init__(self):
        super().__init__()
        self.node = None
        self.node_updates = []
        self.snapshot_contexts = []

    def set_transfer_node(self, op, phase, name):
        self.node = TransferNode(op, phase, name)
        self.node_updates.append(self.node)

    def copy_d2h(self, sid):
        super().copy_d2h(sid)
        self.snapshot_contexts.append((sid, self.node))


def test_restore_and_release_do_not_create_selection_context():
    manager = ReplayManager()
    manager._residency[9] = PhysicalBuffer(device=torch.device('cpu'))
    schedule = ResidencySchedule()
    schedule.add_pre(0, Action.COPY_H2D, 9)
    schedule.add_pre(0, Action.RELEASE_HOST, 9)
    schedule.add_post(0, Action.RELEASE_DEVICE, 9)
    executor = ReplayExecutor(manager, schedule, [OpGuide(name='<lambda>', output_leaf_count=1)])
    result = executor.dispatch(lambda value: value+1, (torch.tensor(2),), {})
    assert result.item() == 3
    assert manager.calls == ['copy_h2d(9)', 'release_host(9)', 'release_device(9)']
    assert not manager.node_updates
    executor.finish()


def test_multiple_snapshots_keep_exact_position_and_action_order():
    manager = ReplayManager()
    schedule = ResidencySchedule()
    for sid in (9, 10):
        schedule.add_post(0, Action.COPY_D2H, sid)
        schedule.add_post(0, Action.RELEASE_DEVICE, sid)
    guide = [OpGuide(name='<lambda>', output_leaf_count=2, output_bindings={0:9, 1:10})]
    executor = ReplayExecutor(manager, schedule, guide)
    executor.dispatch(lambda value: (value+1, value+2), (torch.tensor(2),), {})
    node = TransferNode(0, 'post', '<lambda>')
    assert manager.snapshot_contexts == [(9,node), (10,node)]
    assert manager.calls == ['copy_d2h(9)', 'release_device(9)', 'copy_d2h(10)', 'release_device(10)']
    executor.finish()


def test_overload_identity_is_preserved_for_selector():
    manager = ReplayManager()
    schedule = ResidencySchedule()
    schedule.add_post(0, Action.COPY_D2H, 9)
    guide = [OpGuide(name='aten.add.Tensor', output_leaf_count=1, output_bindings={0:9})]
    executor = ReplayExecutor(manager, schedule, guide)
    executor.dispatch(torch.ops.aten.add.Tensor, (torch.tensor(2), torch.tensor(3)), {})
    assert manager.snapshot_contexts == [(9,TransferNode(0,'post','aten.add.Tensor'))]
    executor.finish()


def test_invalid_operator_cannot_reach_snapshot_selection():
    manager = ReplayManager()
    schedule = ResidencySchedule()
    schedule.add_post(0, Action.COPY_D2H, 9)
    executor = ReplayExecutor(manager, schedule, [OpGuide(name='aten.mul.Tensor', output_leaf_count=1)])
    with pytest.raises(RuntimeError, match='replay operator differs'):
        executor.dispatch(torch.ops.aten.add.Tensor, (torch.tensor(2), torch.tensor(3)), {})
    assert not manager.snapshot_contexts


def test_warmup_eviction_retains_pre_boundary_context():
    class WarmupManager(_FakeResidencyManager):
        def set_transfer_node(self, op, phase, name):
            self.node = TransferNode(op, phase, name)

        def copy_d2h(self, sid):
            super().copy_d2h(sid)
            self.snapshot_context = (sid,self.node)

    manager = WarmupManager()
    manager._resident[9] = 1024
    with patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer'):
        executor = WarmupExecutor(manager, memory_limit_bytes=16)
        executor._sid_produced_at_op[9] = -1
        executor.on_op_begin(torch.ops.aten.add.Tensor, (torch.tensor(2),torch.tensor(3)), {})
    assert manager.snapshot_context == (9,TransferNode(0,'pre','aten.add.Tensor'))
    assert manager.calls == ['copy_d2h(9)', 'release_device(9)']
