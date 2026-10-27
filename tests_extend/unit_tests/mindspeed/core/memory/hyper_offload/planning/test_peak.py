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
"""Unit tests for the trace-native peak residency planner."""

from __future__ import annotations

import logging
import unittest

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind, ActivationTrace, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.planning.peak import PeakResidencyPlanner


class TestPeakResidencyPlanner(unittest.TestCase):
    """Peak planner should reduce simulated peak memory and materialize the expected schedule."""

    @staticmethod
    def _action_kinds(actions):
        """Return action kinds for compact assertions."""
        return [action.kind for action in actions]

    @staticmethod
    def _has_action(actions, kind: ResidencyActionType, storage_id: int) -> bool:
        """Return whether an action list contains the expected action."""
        return any(action.kind == kind and action.storage_id == storage_id for action in actions)

    @staticmethod
    def _simulate_device_peak(trace: ActivationTrace, schedule) -> int:
        """Simulate replay-time device residency from the emitted schedule."""
        produced_sids = set()
        resident_sids = set()
        peak = 0

        for op_id, op in enumerate(trace.ops):
            for action in schedule.pre_actions(op_id):
                if action.kind == ResidencyActionType.COPY_H2D:
                    resident_sids.add(action.storage_id)

            for access in op.accesses:
                if access.storage_id not in produced_sids:
                    produced_sids.add(access.storage_id)
                    resident_sids.add(access.storage_id)

            peak = max(peak, sum(trace.storage_sizes[sid] for sid in resident_sids))

            for action in schedule.post_actions(op_id):
                if action.kind == ResidencyActionType.RELEASE_DEVICE:
                    resident_sids.discard(action.storage_id)

        return peak

    @staticmethod
    def _post_action_summary(schedule, op_id: int) -> list[tuple[ResidencyActionType, int]]:
        """Return post-action kinds and storage ids for one op."""
        return [(action.kind, action.storage_id) for action in schedule.post_actions(op_id)]

    def test_selects_peak_covering_gap(self) -> None:
        """Select an offload gap that covers the memory peak."""
        trace = ActivationTrace(
            ops=[
                TraceOp("op0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("op1", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("op2", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
            ],
            storage_sizes={1: 100, 2: 150},
            memory_limit_bytes=200,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertEqual(
            self._action_kinds(schedule.post_actions(0)),
            [ResidencyActionType.COPY_D2H, ResidencyActionType.RELEASE_DEVICE],
        )
        self.assertEqual(self._action_kinds(schedule.pre_actions(2)), [ResidencyActionType.COPY_H2D, ResidencyActionType.RELEASE_HOST])
        self.assertEqual(self._simulate_device_peak(trace, schedule), 150)

    def test_write_boundary_creates_new_version(self) -> None:
        """Treat writes to the same storage as separate offload versions."""
        trace = ActivationTrace(
            ops=[
                TraceOp("write_v0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("user_0", accesses=[StorageAccess(1, 1, AccessKind.READ)]),
                TraceOp("user_1", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
                TraceOp("inplaceadd", accesses=[StorageAccess(3, 1, AccessKind.WRITE)]),
                TraceOp("user_2", accesses=[StorageAccess(4, 1, AccessKind.READ)]),
                TraceOp("user_3", accesses=[StorageAccess(5, 1, AccessKind.READ)]),
            ],
            storage_sizes={1: 100},
            memory_limit_bytes=50,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertTrue(self._has_action(schedule.post_actions(0), ResidencyActionType.COPY_D2H, 1))
        self.assertTrue(
            self._has_action(schedule.post_actions(0), ResidencyActionType.RELEASE_DEVICE, 1)
            or self._has_action(schedule.post_actions(1), ResidencyActionType.RELEASE_DEVICE, 1),
        )
        self.assertTrue(self._has_action(schedule.post_actions(3), ResidencyActionType.COPY_D2H, 1))
        self.assertTrue(
            self._has_action(schedule.post_actions(3), ResidencyActionType.RELEASE_DEVICE, 1)
            or self._has_action(schedule.post_actions(4), ResidencyActionType.RELEASE_DEVICE, 1),
        )
        self.assertTrue(
            self._action_kinds(schedule.pre_actions(1)) == [ResidencyActionType.COPY_H2D, ResidencyActionType.RELEASE_HOST]
            or self._action_kinds(schedule.pre_actions(2)) == [ResidencyActionType.COPY_H2D, ResidencyActionType.RELEASE_HOST],
        )
        self.assertTrue(
            self._action_kinds(schedule.pre_actions(4)) == [ResidencyActionType.COPY_H2D, ResidencyActionType.RELEASE_HOST]
            or self._action_kinds(schedule.pre_actions(5)) == [ResidencyActionType.COPY_H2D, ResidencyActionType.RELEASE_HOST],
        )

    def test_retained_storage_is_not_offloaded(self) -> None:
        """Keep retained storage resident and only release normal storage."""
        trace = ActivationTrace(
            ops=[
                TraceOp("op0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("op1", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("op2", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
            ],
            storage_sizes={1: 100, 2: 150},
            retained_sids={1},
            memory_limit_bytes=200,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertEqual(schedule.pre_actions(2), [])
        self.assertEqual(self._post_action_summary(schedule, 0), [])
        self.assertEqual(self._post_action_summary(schedule, 1), [(ResidencyActionType.RELEASE_DEVICE, 2)])

    def test_alias_chain_extends_lifetime(self) -> None:
        """Extend residency lifetime across read-write alias operations."""
        trace = ActivationTrace(
            ops=[
                TraceOp("op0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp(
                    "view",
                    accesses=[
                        StorageAccess(1, 1, AccessKind.READ),
                        StorageAccess(1, 1, AccessKind.WRITE),
                    ],
                ),
                TraceOp("op2", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
            ],
            storage_sizes={1: 100},
            memory_limit_bytes=50,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertEqual(schedule.pre_actions(2), [])
        self.assertEqual(self._post_action_summary(schedule, 2), [(ResidencyActionType.RELEASE_DEVICE, 1)])

    def test_tuple_outputs_are_modeled_per_storage(self) -> None:
        """Model tuple output storages independently during selection."""
        trace = ActivationTrace(
            ops=[
                TraceOp("A", accesses=[StorageAccess(0, 1, AccessKind.WRITE), StorageAccess(0, 2, AccessKind.WRITE)]),
                TraceOp("u1", accesses=[StorageAccess(10, 1, AccessKind.READ)]),
                TraceOp("peak", accesses=[StorageAccess(50, 3, AccessKind.WRITE)]),
                TraceOp("u2", accesses=[StorageAccess(100, 2, AccessKind.READ)]),
            ],
            storage_sizes={1: 10, 2: 10, 3: 80},
            memory_limit_bytes=15,
        )

        schedule = PeakResidencyPlanner().build(trace)

        selected = [action.storage_id for action in schedule.post_actions(0) if action.kind == ResidencyActionType.COPY_D2H]
        self.assertIn(2, selected)
        self.assertNotIn(1, selected)

    def test_peak_plateau_uses_coverable_slot(self) -> None:
        """Choose a coverable slot when the peak spans a plateau."""
        trace = ActivationTrace(
            ops=[
                TraceOp("A", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("B", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("C", accesses=[StorageAccess(2, 1, AccessKind.READ), StorageAccess(2, 2, AccessKind.READ)]),
            ],
            storage_sizes={1: 100, 2: 100},
            retained_sids={2},
            memory_limit_bytes=100,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertTrue(self._has_action(schedule.post_actions(0), ResidencyActionType.COPY_D2H, 1))
        self.assertTrue(self._has_action(schedule.post_actions(0), ResidencyActionType.RELEASE_DEVICE, 1))

    def test_communication_masking_moves_release_and_prefetch_with_order(self) -> None:
        """Move releases and prefetches while preserving schedule order."""
        trace = ActivationTrace(
            ops=[
                TraceOp("produce_a", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("produce_b", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("work2"),
                TraceOp("work3"),
                TraceOp("work4"),
                TraceOp("peak", accesses=[StorageAccess(5, 3, AccessKind.WRITE)]),
                TraceOp("peak_user", accesses=[StorageAccess(6, 3, AccessKind.READ)]),
                TraceOp("work7"),
                TraceOp("use_a", accesses=[StorageAccess(8, 1, AccessKind.READ)]),
                TraceOp("use_b", accesses=[StorageAccess(9, 2, AccessKind.READ)]),
            ],
            storage_sizes={1: 60, 2: 60, 3: 130},
            memory_limit_bytes=180,
        )

        schedule = PeakResidencyPlanner(enable_communication_masking=True).build(trace)

        release_sids = [action.storage_id for action in schedule.post_actions(4) if action.kind == ResidencyActionType.RELEASE_DEVICE]
        prefetch_sids = [action.storage_id for action in schedule.pre_actions(7) if action.kind == ResidencyActionType.COPY_H2D]

        self.assertEqual(release_sids, [1, 2])
        self.assertEqual(prefetch_sids, [1, 2])
        self.assertFalse(self._has_action(schedule.post_actions(0), ResidencyActionType.RELEASE_DEVICE, 1))
        self.assertFalse(self._has_action(schedule.post_actions(1), ResidencyActionType.RELEASE_DEVICE, 2))
        self.assertEqual(schedule.pre_actions(8), [])
        self.assertEqual(schedule.pre_actions(9), [])
        self.assertLessEqual(self._simulate_device_peak(trace, schedule), trace.memory_limit_bytes)

    def test_selection_exhaustion_still_materializes_schedule(self) -> None:
        """Return a valid schedule when no candidate can reduce the peak."""
        trace = ActivationTrace(
            ops=[
                TraceOp("a0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("a1", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("a2", accesses=[StorageAccess(2, 3, AccessKind.WRITE)]),
            ],
            storage_sizes={1: 120, 2: 120, 3: 120},
            retained_sids={1, 2, 3},
            memory_limit_bytes=100,
        )

        schedule = PeakResidencyPlanner().build(trace)

        self.assertEqual(schedule.pre_actions(0), [])
        self.assertEqual(schedule.post_actions(0), [])
        self.assertEqual(schedule.post_actions(1), [])
        self.assertEqual(schedule.post_actions(2), [])

    def test_selection_exhaustion_logs_peak_position(self) -> None:
        """Log peak information when selection cannot meet the budget."""
        trace = ActivationTrace(
            ops=[
                TraceOp("a0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("a1", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("a2", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
            ],
            storage_sizes={1: 100, 2: 100},
            retained_sids={1},
            memory_limit_bytes=50,
        )

        with self.assertLogs("mindspeed.core.memory.hyper_offload.planning.peak", level=logging.INFO) as cm:
            PeakResidencyPlanner().build(trace)

        joined = "\n".join(cm.output)
        self.assertIn("PeakResidencyPlanner peak after selection", joined)
        self.assertIn("PeakResidencyPlanner final peak position", joined)

    def test_release_and_prefetch_overlap_respect_budget(self) -> None:
        """Keep masked release and prefetch overlap within the budget."""
        trace = ActivationTrace(
            ops=[
                TraceOp("p0", accesses=[StorageAccess(0, 1, AccessKind.WRITE)]),
                TraceOp("p1", accesses=[StorageAccess(1, 2, AccessKind.WRITE)]),
                TraceOp("p2", accesses=[StorageAccess(2, 1, AccessKind.READ)]),
                TraceOp("p3", accesses=[StorageAccess(3, 3, AccessKind.WRITE)]),
                TraceOp("p4", accesses=[StorageAccess(4, 2, AccessKind.READ)]),
            ],
            storage_sizes={1: 80, 2: 60, 3: 80},
            memory_limit_bytes=120,
        )

        schedule = PeakResidencyPlanner(enable_communication_masking=True).build(trace)

        self.assertLessEqual(self._simulate_device_peak(trace, schedule), trace.memory_limit_bytes)
        self.assertTrue(any(action.kind == ResidencyActionType.RELEASE_HOST for action in schedule.pre_actions(2)))


if __name__ == "__main__":
    unittest.main()
