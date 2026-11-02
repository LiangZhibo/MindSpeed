"""Budget and qualification tests for the pure joint tuning controller."""
import unittest

from mindspeed.core.memory.hyper_offload.tuning import (
    JointPolicy, JointTuningController, StepObservation, TuningLimits,
)


class TestJointController(unittest.TestCase):
    def test_resident_regions_are_distinct_structures_and_cannot_overlap_checkpoints(self):
        checkpointed = JointPolicy(0, 8, recompute_regions=('0.mlp',))
        resident = JointPolicy(0, 8, recompute_regions=('0.mlp',), resident_regions=('1.mlp.gate', '0.self_attn'))
        self.assertNotEqual(checkpointed.structure_key, resident.structure_key)
        self.assertEqual(resident.resident_regions, ('0.self_attn', '1.mlp.gate'))
        for names in (('0',), ('0.mlp',), ('0.mlp.gate',), ('1', '1.self_attn'), ('1', '1')):
            with self.assertRaises(ValueError):
                JointPolicy(0, 8, recompute_regions=('0.mlp',), resident_regions=names)

    def test_region_identity_is_canonical_and_never_only_cardinality(self):
        a = JointPolicy(0, 8, recompute_regions=('1.mlp', '0.self_attn'))
        self.assertEqual(a, JointPolicy(0, 8, recompute_regions=('0.self_attn', '1.mlp')))
        self.assertNotEqual(a.structure_key, JointPolicy(0, 8, recompute_regions=('0.mlp', '1.self_attn')).structure_key)
        for names in (('0', '0.mlp'), ('0.mlp', '0.mlp'), ('0..mlp',)):
            with self.assertRaises(ValueError):
                JointPolicy(0, 8, recompute_regions=names)
        with self.assertRaises(ValueError):
            JointPolicy(1, 8, recompute_regions=('0.mlp',))

    def test_default_budget_freezes_without_continued_search(self):
        initial = JointPolicy(33, 2048)
        candidate = JointPolicy(27, 2048, transfer_window_scale=1)
        proposals = []

        def propose(trials):
            proposals.append(len(trials))
            return candidate

        tuner = JointTuningController(initial, propose)
        for _ in range(10):
            cost = 12 if tuner.policy == initial else 10
            tuner.observe(StepObservation(cost, 100, 200))
        self.assertTrue(tuner.stable)
        self.assertEqual(tuner.policy, candidate)
        self.assertEqual(tuner.frozen_at_step, 8)
        self.assertEqual(proposals, [1])
        counters = (tuner.steps, tuner.search_seconds, len(tuner.trials))
        for _ in range(100):
            tuner.observe(StepObservation(float('nan'), 1000, 2000, False))
        self.assertEqual(counters, (tuner.steps, tuner.search_seconds, len(tuner.trials)))

    def test_warmup_time_is_excluded_but_memory_is_not(self):
        initial = JointPolicy(33, 2048)
        candidate = JointPolicy(27, 2048, transfer_window_scale=1)
        tuner = JointTuningController(initial, lambda _: candidate,
                                      TuningLimits(device_peak_bytes=100))
        tuner.observe(StepObservation(999, 100, 1))
        tuner.observe(StepObservation(12, 100, 1))
        tuner.observe(StepObservation(12, 100, 1))
        self.assertEqual(tuner.trials[0].score, 12)
        tuner.observe(StepObservation(1, 101, 1))
        self.assertEqual(tuner.policy, initial)
        self.assertEqual(tuner.phase, 'restore')
        self.assertFalse(tuner.trials[1].feasible)
        for _ in range(3):
            tuner.observe(StepObservation(12, 100, 1))
        self.assertTrue(tuner.stable)

    def test_three_candidates_and_restore_fit_twelve_steps(self):
        policies = [JointPolicy(33 - i * 6, 2048, transfer_window_scale=i or None) for i in range(3)]
        tuner = JointTuningController(policies[0], lambda trials: policies[len(trials)],
                                      TuningLimits(max_steps=12))
        for step in range(12):
            elapsed = 10 if tuner.policy == policies[0] else 12
            tuner.observe(StepObservation(elapsed, 1, 1))
        self.assertEqual(len(tuner.trials), 3)
        self.assertEqual(tuner.frozen_at_step, 12)
        self.assertEqual(tuner.policy, policies[0])

    def test_proposal_exhaustion_ends_early(self):
        tuner = JointTuningController(JointPolicy(33, 2048), lambda _: None)
        for _ in range(5):
            tuner.observe(StepObservation(1, 1, 1))
        self.assertTrue(tuner.stable)
        self.assertEqual(tuner.frozen_at_step, 5)
        self.assertEqual(tuner.stop_reason, 'proposal_exhausted')

    def test_confirmation_failure_is_not_silently_accepted(self):
        tuner = JointTuningController(JointPolicy(33, 2048), lambda _: None,
                                      TuningLimits(host_peak_bytes=100))
        for _ in range(3):
            tuner.observe(StepObservation(1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, 'failed confirmation'):
            tuner.observe(StepObservation(1, 1, 101))
        self.assertFalse(tuner.stable)

    def test_invalid_measurement_never_becomes_fastest_candidate(self):
        tuner = JointTuningController(JointPolicy(33, 2048), lambda _: None)
        with self.assertRaisesRegex(RuntimeError, 'no measured policy'):
            tuner.observe(StepObservation(float('nan'), 1, 1))

    def test_budget_must_include_qualification(self):
        with self.assertRaises(ValueError):
            TuningLimits(max_steps=3)

    def test_near_tie_keeps_measured_incumbent_within_step_budget(self):
        policies = [JointPolicy(i, 2048) for i in range(3)]
        costs = {policies[0]: 12.7, policies[1]: 12.1764, policies[2]: 12.1702}
        tuner = JointTuningController(policies[0], lambda trials: policies[len(trials)],
                                      TuningLimits(max_steps=12))
        for _ in range(12):
            tuner.observe(StepObservation(costs[tuner.policy], 1, 1))
        self.assertEqual(tuner.policy, policies[1])
        self.assertTrue(tuner.stable)
        self.assertEqual(tuner.frozen_at_step, 12)

    def test_zero_margin_retains_exact_minimum_selection(self):
        initial, candidate = JointPolicy(1, 8), JointPolicy(0, 8)
        tuner = JointTuningController(initial, lambda _: candidate,
                                      TuningLimits(minimum_relative_gain=0))
        for _ in range(10):
            tuner.observe(StepObservation(10 if tuner.policy == initial else 9.999, 1, 1))
        self.assertEqual(tuner.policy, candidate)
        self.assertTrue(tuner.stable)

    def test_gain_margin_requires_finite_fraction(self):
        for margin in (-0.001, 1, float('inf'), float('nan')):
            with self.assertRaisesRegex(ValueError, 'minimum_relative_gain'):
                TuningLimits(minimum_relative_gain=margin)


if __name__ == '__main__':
    unittest.main()
