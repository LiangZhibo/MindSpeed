"""Independent discrete-slot oracles for planner interval optimizations."""
import random
import unittest

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, AccessKind, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import modeling, selection
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import PeakCandidate


class TestProfileOracles(unittest.TestCase):
    def test_profiles_and_unavoidable_floor_match_discrete_oracle(self):
        rng = random.Random(815)
        for _ in range(120):
            count = rng.randrange(4, 70)
            trace = ActivationTrace(ops=[TraceOp('random') for _ in range(count)])
            for sid in range(rng.randrange(1, 30)):
                trace.storage_sizes[sid] = rng.randrange(1, 1000)
                positions = sorted(rng.sample(range(count), rng.randrange(1, min(count, 8))))
                for pos in positions:
                    trace.ops[pos].accesses.append(StorageAccess(pos, sid, rng.choice(list(AccessKind))))
                if rng.random() < .15:
                    trace.retained_sids.add(sid)
            grouped = modeling.group_accesses_by_storage(trace)
            actual = modeling.build_profile_and_candidates(trace, grouped)
            resident, edges = [0] * count, [0] * (count - 1)
            for sid, accesses in grouped.items():
                first, last = min(a.op_id for a in accesses), max(a.op_id for a in accesses)
                if sid in trace.retained_sids:
                    last = count - 1
                for i in range(first, last + 1):
                    resident[i] += trace.storage_sizes[sid]
                for i in range(first, last):
                    edges[i] += trace.storage_sizes[sid]
            self.assertEqual(actual.resident_bytes, resident)
            self.assertEqual(actual.edge_bytes, edges)
            floor = []
            for i, size in enumerate(resident):
                ids = {c.storage_id for c in actual.candidates if c.start_interior <= i <= c.end_interior}
                reducible = min(size, sum(trace.storage_sizes[sid] for sid in ids))
                floor.append((size - reducible, i, reducible))
            unavoidable, pos, reducible = max(floor)
            self.assertEqual(selection.compute_unavoidable_peak(resident, actual.candidates),
                             (pos, unavoidable, reducible))

    def test_neighbor_union_preserves_exact_overlap(self):
        rng = random.Random(309)
        for _ in range(200):
            start, end = sorted(rng.sample(range(100), 2))
            access = StorageAccess(start, 1, AccessKind.WRITE)
            candidate = PeakCandidate(1, 10, access, access,
                                      StorageAccess(end, 1, AccessKind.READ), start, end)
            intervals = [tuple(sorted(rng.sample(range(100), 2))) for _ in range(50)]
            covered = {i for a, b in intervals for i in range(a, b)}
            expected = len(covered.intersection(range(start, end))) / (end - start)
            merged = selection.merge_lifecycles(intervals)
            self.assertEqual(selection.compute_neighbor_impact(candidate, intervals), expected)
            self.assertEqual(selection.compute_neighbor_impact(candidate, merged), expected)
