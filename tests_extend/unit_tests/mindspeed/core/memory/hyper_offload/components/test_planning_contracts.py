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
"""Independent version-state oracle for planner output, not its internal model."""

import copy
import random

import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, TraceOp, StorageAccess, AccessKind
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType as Action
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner


PLANNERS = [GreedyResidencyPlanner, PeakResidencyPlanner,
            lambda: PeakResidencyPlanner(enable_communication_masking=True)]


def execute_schedule(trace, schedule):
    """Model synchronous byte versions and assert every op can read its latest value."""
    device, host, latest = {}, {}, {}
    produced = set()
    for index, op in enumerate(trace.ops):
        for action in schedule.pre_actions(index):
            sid = action.storage_id
            if action.kind == Action.COPY_H2D:
                if sid not in device:
                    assert sid in host, ('missing host', index, sid)
                    device[sid] = host[sid]
            elif action.kind == Action.RELEASE_HOST:
                host.pop(sid, None)
            else:
                pytest.fail(f'invalid pre action {action}')
        for access in op.accesses:
            sid = access.storage_id
            if sid in produced:
                assert sid in device, ('missing device', index, sid)
                assert device[sid] == latest[sid], ('stale device', index, sid)
            else:
                assert access.kind == AccessKind.WRITE
                produced.add(sid)
                latest[sid] = 0
            if access.kind == AccessKind.WRITE:
                latest[sid] += 1
                device[sid] = latest[sid]
                # Runtime invalidates old snapshots before actual writes.
                host.pop(sid, None)
        for action in schedule.post_actions(index):
            sid = action.storage_id
            if action.kind == Action.COPY_D2H:
                if sid not in host:
                    assert sid in device, ('copy after release', index, sid)
                    host[sid] = device[sid]
            elif action.kind == Action.RELEASE_DEVICE:
                device.pop(sid, None)
            elif action.kind == Action.RELEASE_HOST:
                host.pop(sid, None)
            else:
                pytest.fail(f'invalid post action {action}')
    for sid in trace.retained_sids:
        assert sid in device or sid in host


def generated_trace(seed):
    rng = random.Random(seed)
    ops = [TraceOp(f'op{i}', duration_ms=0.01) for i in range(32)]
    sizes = {}
    for sid in range(1, 9):
        sizes[sid] = rng.choice([1, 4, 16, 64, 256])
        uses = sorted(rng.sample(range(32), rng.randint(2, 9)))
        for position, op_id in enumerate(uses):
            kind = AccessKind.WRITE if position == 0 or rng.random() < .3 else AccessKind.READ
            if position and kind == AccessKind.WRITE and rng.random() < .5:
                ops[op_id].accesses.append(StorageAccess(op_id, sid, AccessKind.READ))
            ops[op_id].accesses.append(StorageAccess(op_id, sid, kind))
    return ActivationTrace(ops, sizes, set(rng.sample(range(1, 9), 2)), rng.choice([0, 16, 80, 256, 512]))


@pytest.mark.parametrize('factory', PLANNERS, ids=['greedy', 'peak', 'masked'])
@pytest.mark.parametrize('seed', range(40))
def test_generated_trace_preserves_data_and_input(factory, seed):
    trace = generated_trace(seed)
    before = copy.deepcopy(trace)
    planner = factory()
    schedule = planner.build(trace)
    assert trace == before
    assert planner.build(trace) == schedule
    execute_schedule(trace, schedule)


@pytest.mark.parametrize('factory', PLANNERS, ids=['greedy', 'peak', 'masked'])
@pytest.mark.parametrize('damage', ['negative_op', 'op_mismatch', 'missing_size', 'negative_size',
                                   'fractional_size', 'negative_budget', 'nan_duration', 'bad_kind'])
def test_invalid_trace_is_rejected_before_planning(factory, damage):
    trace = ActivationTrace([TraceOp('produce', accesses=[StorageAccess(0, 1, AccessKind.WRITE)])], {1: 8},
                            memory_limit_bytes=4)
    if damage == 'negative_op': trace.ops[0].accesses[0].op_id = -1
    elif damage == 'op_mismatch': trace.ops[0].accesses.append(StorageAccess(2, 1, AccessKind.READ))
    elif damage == 'missing_size': trace.storage_sizes.clear()
    elif damage == 'negative_size': trace.storage_sizes[1] = -1
    elif damage == 'fractional_size': trace.storage_sizes[1] = .5
    elif damage == 'negative_budget': trace.memory_limit_bytes = -1
    elif damage == 'nan_duration': trace.ops[0].duration_ms = float('nan')
    else: trace.ops[0].accesses[0].kind = 'read'
    with pytest.raises((ValueError, TypeError), match='trace|storage|budget|duration|access'):
        factory().build(trace)


@pytest.mark.parametrize('factory', PLANNERS, ids=['greedy', 'peak', 'masked'])
@pytest.mark.parametrize('budget', [None, 0, 1, 1024])
def test_empty_and_zero_storage_traces(factory, budget):
    assert not factory().build(ActivationTrace(memory_limit_bytes=budget)).pre
    trace = ActivationTrace([TraceOp('empty', accesses=[StorageAccess(0, 1, AccessKind.WRITE)])],
                            {1: 0}, {1}, memory_limit_bytes=budget)
    execute_schedule(trace, factory().build(trace))


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_peak_weight_is_rejected(weight):
    with pytest.raises(ValueError, match='finite'):
        PeakResidencyPlanner(score_weight_size=weight)


@pytest.mark.parametrize('factory', PLANNERS, ids=['greedy', 'peak', 'masked'])
def test_retained_storage_remains_in_profile_until_step_end(factory):
    trace = ActivationTrace([
        TraceOp('retained', accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
        TraceOp('temporary', accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
        TraceOp('use', accesses=[StorageAccess(2, 2, AccessKind.READ)]),
        TraceOp('tail_without_activations'),
    ], {1: 8, 2: 8}, {1}, memory_limit_bytes=8)
    planner = factory()
    grouped = planner._group_accesses_by_storage(trace)
    if isinstance(planner, GreedyResidencyPlanner):
        profile, _, _ = planner._build_resident_footprint(trace, grouped)
    else:
        profile = planner._build_profile_and_candidates(trace, grouped).resident_bytes
    assert profile == [8, 16, 16, 8]


@pytest.mark.parametrize('factory', PLANNERS, ids=['greedy', 'peak', 'masked'])
def test_sparse_and_split_op_records_normalize_without_mutating_input(factory):
    trace = ActivationTrace([
        TraceOp('produce', accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
        TraceOp('splitA', accesses=[StorageAccess(5, 1, AccessKind.READ)]),
        TraceOp('splitB', accesses=[StorageAccess(5, 2, AccessKind.WRITE)]),
    ], {1: 8, 2: 8}, memory_limit_bytes=4)
    before = copy.deepcopy(trace)
    dense = trace.validate()
    assert len(dense.ops) == 6 and len(dense.ops[5].accesses) == 2
    assert trace == before
    execute_schedule(dense, factory().build(trace))
