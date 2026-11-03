"""Memory and dependency contracts for bounded transfer masking."""
import unittest
from dataclasses import replace

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType as Action
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind, StorageAccess
from mindspeed.core.memory.hyper_offload.planning.peak import PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.masking import apply_communication_masking
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import MaskingContext, PeakCandidate


class TestTransferWindows(unittest.TestCase):
    def context(self, scale):
        size = 1024**2
        start = StorageAccess(0, 1, AccessKind.WRITE)
        end = StorageAccess(10, 1, AccessKind.READ)
        candidate = PeakCandidate(1, size, start, start, end, 0, 10)
        return MaskingContext([candidate], [size] + [0] * 9 + [size], size,
                              {1: end}, set(), list(range(12)), 1, 1, scale)

    @staticmethod
    def positions(schedule, kind):
        return [a.op_id for phase in (schedule.pre, schedule.post)
                for actions in phase.values() for a in actions if a.kind == kind]

    def test_timed_windows_free_room_for_prefetch(self):
        context = self.context(1)
        schedule, resident = apply_communication_masking(context)
        self.assertEqual(self.positions(schedule, Action.COPY_D2H), [0])
        self.assertEqual(self.positions(schedule, Action.COPY_H2D), [9])
        self.assertIn(1, self.positions(schedule, Action.RELEASE_DEVICE))
        self.assertLessEqual(max(resident), context.limit)

    def test_legacy_masking_keeps_existing_positions(self):
        context = self.context(None)
        schedule, resident = apply_communication_masking(context)
        self.assertEqual(self.positions(schedule, Action.COPY_H2D), [10])
        self.assertIn(9, self.positions(schedule, Action.RELEASE_DEVICE))
        self.assertLessEqual(max(resident), context.limit)

    def test_release_window_does_not_need_to_limit_prefetch(self):
        context = replace(self.context(1), limit_prefetch_window=False)
        schedule, resident = apply_communication_masking(context)
        self.assertIn(1, self.positions(schedule, Action.RELEASE_DEVICE))
        self.assertEqual(self.positions(schedule, Action.COPY_H2D), [2])
        self.assertLessEqual(max(resident), context.limit)

    def test_zero_window_remains_dependency_safe(self):
        context = self.context(0)
        schedule, resident = apply_communication_masking(context)
        self.assertEqual(self.positions(schedule, Action.COPY_H2D), [10])
        self.assertIn(0, self.positions(schedule, Action.RELEASE_DEVICE))
        self.assertLessEqual(max(resident), context.limit)

    def test_tight_memory_overrides_transfer_window(self):
        context = self.context(1)
        context.resident_bytes[1] = context.limit
        context.resident_bytes[9] = context.limit
        schedule, resident = apply_communication_masking(context)
        self.assertIn(0, self.positions(schedule, Action.RELEASE_DEVICE))
        self.assertEqual(self.positions(schedule, Action.COPY_H2D), [10])
        self.assertLessEqual(max(resident), context.limit)

    def test_invalid_configuration_fails_before_planning(self):
        for scale in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PeakResidencyPlanner(enable_communication_masking=True, transfer_window_scale=scale)
        with self.assertRaises(ValueError):
            PeakResidencyPlanner(transfer_window_scale=1)
        context = replace(self.context(1), d2h_bandwidth_gbps=0)
        with self.assertRaises(ValueError):
            apply_communication_masking(context)


if __name__ == '__main__':
    unittest.main()


def test_measured_packet_latency_changes_window_without_relaxing_capacity():
    case=TestTransferWindows()
    original=case.context(1)
    calibrated=replace(case.context(1),transfer_cost_samples=((1024**2,.5,8.),))
    base,_=apply_communication_masking(original)
    schedule,resident=apply_communication_masking(calibrated)
    assert case.positions(schedule,Action.COPY_H2D)<case.positions(base,Action.COPY_H2D)
    assert max(resident)<=calibrated.limit
    constrained=case.context(1)
    constrained.resident_bytes[1]=constrained.limit
    constrained.resident_bytes[9]=constrained.limit
    constrained=replace(constrained,transfer_cost_samples=((1024**2,.5,8.),))
    schedule,resident=apply_communication_masking(constrained)
    assert case.positions(schedule,Action.COPY_H2D)==[10]
    assert max(resident)<=constrained.limit
