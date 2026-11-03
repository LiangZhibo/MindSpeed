"""Budget and qualification tests for the pure joint tuning controller."""
import unittest
from unittest.mock import patch

import pytest

from mindspeed.core.memory.hyper_offload.tuning import (
    JointPolicy, JointTuningController, StepObservation, TuningLimits,
)


@pytest.mark.parametrize('boundary', ['before_proposal', 'after_proposal'])
def test_shared_search_budget_keeps_straddling_rank_clocks_on_same_branch(boundary):
    initial, candidate = JointPolicy(2, 64), JointPolicy(1, 64)
    decisions = []
    for local_seconds in (9.999, 10.001):
        reduced, proposed = [], []
        def reduce_seconds(value):
            reduced.append(value)
            return 0. if boundary == 'after_proposal' and len(reduced) == 1 else 10.001
        def propose(trials):
            proposed.append(len(trials))
            tuner.search_seconds = local_seconds
            return candidate
        tuner = JointTuningController(initial, propose,
            TuningLimits(max_steps=12, max_search_seconds=10.), search_time_reducer=reduce_seconds)
        with patch('mindspeed.core.memory.hyper_offload.tuning.controller.time.perf_counter', return_value=0.):
            for _ in range(2):
                tuner.observe(StepObservation(10., 0, 0))
            tuner.search_seconds = local_seconds if boundary == 'before_proposal' else 0.
            tuner.observe(StepObservation(10., 0, 0))
            assert tuner.phase == 'confirm' and tuner.policy == initial
            assert tuner.stop_reason == 'search_time_budget'
            for _ in range(2):
                tuner.observe(StepObservation(10., 0, 0))
            assert tuner.stable
            calls = len(reduced)
            for _ in range(5):
                tuner.observe(StepObservation(10., 0, 0))
            assert len(reduced) == calls
        decisions.append((tuner.phase, tuner.policy, tuner.frozen_at_step, len(proposed), calls))
    assert decisions[0] == decisions[1]
    assert decisions[0][-2:] == ((0, 1) if boundary == 'before_proposal' else (1, 2))


@pytest.mark.parametrize('invalid', [True, None, -1., float('nan'), float('inf'), 0.5])
def test_search_budget_reducer_cannot_hide_elapsed_time(invalid):
    tuner = JointTuningController(JointPolicy(2, 64), lambda _: None,
        search_time_reducer=lambda _: invalid)
    tuner.search_seconds = 1.
    tuner.observe(StepObservation(10., 0, 0))
    tuner.observe(StepObservation(10., 0, 0))
    with pytest.raises(ValueError, match='finite upper bound'):
        tuner.observe(StepObservation(10., 0, 0))


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


def guarded(policies, max_steps=12):
    return JointTuningController(policies[0],lambda trials: policies[len(trials)] if len(trials)<len(policies) else None,
        TuningLimits(max_steps=max_steps,timing_noise_multiplier=1.0,validate_confirmation_timing=True))


def observe_costs(tuner, values):
    for value in values:
        tuner.observe(StepObservation(value,100,200))


def test_noise_guard_does_not_promote_an_overlapping_two_sample_gain():
    initial,candidate=JointPolicy(2,8),JointPolicy(1,8)
    tuner=guarded([initial,candidate],max_steps=10)
    observe_costs(tuner,[99,10,10,99,8,11,10,10,10])
    assert tuner.stable and tuner.policy==initial and tuner.frozen_at_step==9
    assert tuner.trials[1].score==9.5 and tuner.trials[1].spread==1.5


def test_independent_confirmation_can_restore_initial_within_ten_steps():
    initial,candidate=JointPolicy(2,8),JointPolicy(1,8)
    tuner=guarded([initial,candidate],max_steps=10)
    observe_costs(tuner,[99,10,10,99,8,8,11,11])
    assert tuner.phase=='fallback' and tuner.policy==initial
    observe_costs(tuner,[10])
    assert tuner.stable and tuner.frozen_at_step==9 and tuner.confirmation_fallback
    assert tuner.stop_reason=='confirmation_timing_fallback'
    assert tuner.selection_log[-1]['phase']=='fallback_validation'
    counters=(tuner.steps,tuner.search_seconds,len(tuner.selection_log))
    observe_costs(tuner,[100]*30)
    assert counters==(tuner.steps,tuner.search_seconds,len(tuner.selection_log))


def test_three_candidates_reserve_last_step_for_verified_timing_fallback():
    policies=[JointPolicy(i,8) for i in (3,2,1)]
    tuner=guarded(policies)
    observe_costs(tuner,[99,10,10,99,9,9,99,8,8,11,11,10])
    assert tuner.stable and tuner.frozen_at_step==12
    assert tuner.policy==policies[0] and tuner.confirmation_fallback
    assert len(tuner.trials)==3


def test_return_to_earlier_winner_has_explicit_one_step_confirmation_at_cap():
    policies=[JointPolicy(i,8) for i in (3,2,1)]
    tuner=guarded(policies)
    observe_costs(tuner,[99,10,10,99,8,8,99,9,9])
    assert tuner.phase=='restore' and tuner.policy==policies[1]
    assert tuner.confirmation_target==1
    observe_costs(tuner,[8,11,10])
    assert tuner.stable and tuner.frozen_at_step==12 and tuner.policy==policies[0]


def test_confirmed_gain_freezes_and_failed_fallback_capacity_never_does():
    initial,candidate=JointPolicy(2,8),JointPolicy(1,8)
    tuner=guarded([initial,candidate],max_steps=10)
    observe_costs(tuner,[99,10,10,99,8,8,8,8])
    assert tuner.stable and tuner.frozen_at_step==8 and tuner.policy==candidate
    tuner=guarded([initial,candidate],max_steps=10)
    observe_costs(tuner,[99,10,10,99,8,8,11,11])
    import pytest
    with pytest.raises(RuntimeError,match='failed confirmation'):
        tuner.observe(StepObservation(10,100,200,valid=False))
    assert not tuner.stable


def test_noise_and_confirmation_options_are_validated():
    import pytest
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='timing_noise_multiplier'):
            TuningLimits(timing_noise_multiplier=value)
    with pytest.raises(ValueError,match='validate_confirmation_timing'):
        TuningLimits(validate_confirmation_timing='yes')


def test_periodic_winner_reserves_complete_confirmation_cycle_and_rollback():
    from dataclasses import replace
    initial=JointPolicy(3,8)
    periodic=replace(initial,recompute_count=2,trim_replay_interval=2)
    further=replace(periodic,recompute_count=1)
    tuner=guarded([initial,periodic,further])
    observe_costs(tuner,[99,10,10,99,7,8])
    assert tuner.policy==periodic and tuner.phase=='confirm'
    assert len(tuner.trials)==2 and tuner.confirmation_target==2
    observe_costs(tuner,[7])
    assert not tuner.stable  # A cheap half-cycle cannot be accepted alone.
    observe_costs(tuner,[14])  # Complete-cycle mean10.5 is slower than initial10.
    assert tuner.phase=='fallback' and tuner.policy==initial
    observe_costs(tuner,[10])
    assert tuner.stable and tuner.frozen_at_step==9 and tuner.confirmation_fallback


def test_late_periodic_candidate_is_rejected_before_execution_if_cycle_cannot_fit():
    from dataclasses import replace
    initial=JointPolicy(3,8);first=JointPolicy(2,8)
    periodic=replace(first,recompute_count=1,trim_replay_interval=2)
    tuner=guarded([initial,first,periodic],max_steps=11)
    observe_costs(tuner,[99,10,10,99,8,8])
    assert tuner.policy==first and tuner.phase=='confirm' and len(tuner.trials)==2
    assert tuner.stop_reason in ('periodic_cycle_budget','tuning_budget')
    observe_costs(tuner,[8,8])
    assert tuner.stable and tuner.frozen_at_step==8


def test_periodic_measurement_and_confirmation_round_up_to_whole_cycles():
    initial=JointPolicy(0,8,trim_replay_interval=2)
    tuner=JointTuningController(initial,lambda _:None,
        TuningLimits(max_steps=12,samples_per_candidate=3,confirmation_steps=3))
    observe_costs(tuner,[99,1,3,1])
    assert not tuner.trials
    observe_costs(tuner,[3])
    assert tuner.trials[0].samples==[1,3,1,3] and tuner.confirmation_target==4
    observe_costs(tuner,[1,3,1])
    assert not tuner.stable
    observe_costs(tuner,[3])
    assert tuner.stable and tuner.frozen_at_step==9


def test_initial_periodic_policy_must_fit_complete_cycles():
    import pytest
    with pytest.raises(ValueError,match='complete initial execution cycles'):
        JointTuningController(JointPolicy(0,8,trim_replay_interval=6),lambda _:None,
                              TuningLimits(max_steps=12))


def test_same_cut_cadence_cannot_hide_behind_initial_recompute_gain():
    from dataclasses import replace
    initial,cut=JointPolicy(3,8),JointPolicy(2,8)
    cadence=replace(cut,trim_replay_interval=2)
    tuner=guarded([initial,cut,cadence])
    observe_costs(tuner,[99,12,12,99,10,10,99,11,11])
    assert tuner.policy==cut and tuner.phase=='restore' and len(tuner.trials)==3
    observe_costs(tuner,[10,10])
    assert tuner.stable and tuner.policy==cut and tuner.frozen_at_step==11


def test_periodic_confirmation_compares_and_returns_to_improved_cut():
    from dataclasses import replace
    initial,cut=JointPolicy(3,8),JointPolicy(2,8)
    cadence=replace(cut,trim_replay_interval=2)
    tuner=guarded([initial,cut,cadence])
    observe_costs(tuner,[99,12,12,99,10,10,99,8,8])
    assert tuner.phase=='confirm' and tuner.confirmation_reference_index==1
    observe_costs(tuner,[11])
    assert not tuner.stable
    observe_costs(tuner,[11])
    assert tuner.phase=='fallback' and tuner.policy==cut  # Still faster than initial12.
    observe_costs(tuner,[10])
    assert tuner.stable and tuner.policy==cut and tuner.frozen_at_step==12


def test_same_cut_periodic_gain_keeps_full_confirmation_and_freezes_at_eleven():
    from dataclasses import replace
    initial,cut=JointPolicy(3,8),JointPolicy(2,8)
    cadence=replace(cut,trim_replay_interval=2)
    tuner=guarded([initial,cut,cadence])
    observe_costs(tuner,[99,12,12,99,10,10,99,8,8,8,8])
    assert tuner.stable and tuner.policy==cadence and tuner.frozen_at_step==11
    assert len(tuner.confirmation)==2 and not tuner.confirmation_fallback


def test_periodic_search_budget_covers_winner_restore_and_fallback_branches():
    from dataclasses import replace
    from itertools import product
    for budget,cycle,candidate_cost,confirmation_cost in product((8,9,10,11,12),(1,2,3,4),(6.,9.),(6.,13.)):
        initial,cut=JointPolicy(3,8),JointPolicy(2,8)
        cadence=replace(cut,recompute_count=1,trim_replay_interval=cycle)
        tuner=guarded([initial,cut,cadence],max_steps=budget)
        while not tuner.stable:
            policy=tuner.policy
            cost=12. if policy==initial else 8. if policy==cut else candidate_cost
            if tuner.phase=='confirm' and policy!=initial:cost=confirmation_cost
            tuner.observe(StepObservation(cost,100,200))
            assert tuner.steps<=budget
        assert tuner.frozen_at_step<=budget
        if tuner.policy.trim_replay_interval>1:
            assert len(tuner.confirmation)%tuner.policy.trim_replay_interval==0


def test_periodic_phase_difference_is_not_between_cycle_noise():
    from mindspeed.core.memory.hyper_offload.tuning.controller import Trial
    trial=Trial(JointPolicy(0,8,trim_replay_interval=2),samples=[2.,12.,4.,14.])
    assert trial.balanced_samples==[7.,9.]
    assert trial.score==8. and trial.spread==1.
    trial.samples=[2.,12.,4.]
    assert trial.score==float('inf') and trial.spread==float('inf')
    triple=Trial(JointPolicy(0,8,trim_replay_interval=3),samples=[1.,1.,31.])
    assert triple.score==11.  # Median cheap phases must not hide the costly phase.


def test_alternating_periodic_gain_requires_and_passes_new_complete_cycle():
    from dataclasses import replace
    initial,cut=JointPolicy(3,8),JointPolicy(2,8)
    cadence=replace(cut,trim_replay_interval=2)
    tuner=guarded([initial,cut,cadence])
    observe_costs(tuner,[99,15,15,99,10,10,99,2,12])
    assert tuner.phase=='confirm' and tuner.policy==cadence
    observe_costs(tuner,[3])
    assert not tuner.stable
    observe_costs(tuner,[11])
    assert tuner.stable and tuner.policy==cadence and tuner.frozen_at_step==11


def test_periodic_confirmation_drift_returns_previous_winner():
    from dataclasses import replace
    initial,cut=JointPolicy(3,8),JointPolicy(2,8)
    cadence=replace(cut,trim_replay_interval=2)
    tuner=guarded([initial,cut,cadence])
    observe_costs(tuner,[99,15,15,99,10,10,99,2,12,3,19])
    assert tuner.phase=='fallback' and tuner.policy==cut
    observe_costs(tuner,[10])
    assert tuner.stable and tuner.policy==cut and tuner.frozen_at_step==12


def tentative_periodic_case(validate=True):
    policies = [JointPolicy(0, 8, recompute_regions=(name,), trim_replay_interval=cycle)
                for name, cycle in [('encoder', 1), ('projection', 1), ('activation', 2)]]
    tuner = JointTuningController(policies[0], lambda trials: policies[len(trials)],
        TuningLimits(max_steps=12, timing_noise_multiplier=1,
                     validate_confirmation_timing=validate))
    observe_costs(tuner, [99, 9.9, 10.1, 99, 8.9, 9.1, 99, 8.92, 8.98])
    return tuner, policies


def test_lower_score_near_tie_can_use_remaining_complete_confirmation_cycle():
    tuner, policies = tentative_periodic_case()
    assert tuner.phase == 'confirm' and tuner.policy == policies[2]
    assert tuner.confirmation_reference_index == 1 and tuner.confirmation_target == 2
    selection = tuner.selection_log[-1]
    assert selection['tentative_confirmation'] and selection['strict_best_index'] == 1
    assert selection['selected_index'] == 2
    observe_costs(tuner, [8.7])
    assert not tuner.stable  # Never accept a partial execution cycle.
    observe_costs(tuner, [8.8])
    assert tuner.stable and tuner.policy == policies[2] and tuner.frozen_at_step == 11
    counters = tuner.steps, tuner.search_seconds, len(tuner.selection_log)
    observe_costs(tuner, [100] * 20)
    assert counters == (tuner.steps, tuner.search_seconds, len(tuner.selection_log))


def test_tentative_confirmation_preserves_margin_and_prior_incumbent_fallback():
    for costs in ([8.9, 8.9], [9.1, 9.2]):
        tuner, policies = tentative_periodic_case()
        observe_costs(tuner, costs)
        # Even a confirmation mean below 9 must clear the unchanged margin.
        assert tuner.phase == 'fallback' and tuner.policy == policies[1]
        observe_costs(tuner, [9])
        assert tuner.stable and tuner.frozen_at_step == 12
        assert tuner.confirmation_fallback and tuner.policy == policies[1]


def test_tentative_confirmation_requires_validation_mode_and_valid_capacity():
    import pytest
    tuner, policies = tentative_periodic_case(validate=False)
    assert tuner.phase == 'restore' and tuner.policy == policies[1]
    assert not tuner.selection_log[-1]['tentative_confirmation']
    tuner, _ = tentative_periodic_case()
    with pytest.raises(RuntimeError, match='failed confirmation'):
        tuner.observe(StepObservation(8, 100, 200, False))
    assert not tuner.stable


def earlier_challenger_case(*, budget=18, enabled=True, incumbent_cycle=1, challenger_cycle=1):
    policies = [JointPolicy(0, 8, recompute_regions=(name,), trim_replay_interval=cycle)
                for name, cycle in [('encoder', 1), ('projection', incumbent_cycle),
                                    ('activation', challenger_cycle), ('other', 1)]]
    tuner = JointTuningController(policies[0], lambda trials:policies[len(trials)],
        TuningLimits(max_steps=budget,max_candidates=4,confirmation_steps=4,
                     timing_noise_multiplier=1.,validate_confirmation_timing=True,
                     confirm_best_observed=enabled))
    # An earlier challenger has a better mean but overlapping two-sample
    # variation. A later slow candidate must not erase its confirmation chance.
    observe_costs(tuner,[99.,9.9,10.1,99.,8.9,9.1,99.,8.4,9.2,99.,12.,12.])
    return tuner,policies


@pytest.mark.parametrize('incumbent_cycle',[1,2])
def test_earlier_lowest_score_receives_independent_confirmation_with_restore(incumbent_cycle):
    tuner,policies=earlier_challenger_case(incumbent_cycle=incumbent_cycle)
    assert tuner.phase=='restore' and tuner.policy==policies[2]
    assert tuner.confirmation_reference_index==1 and tuner.confirmation_target==4
    assert tuner.selection_log[-1]['strict_best_index']==1
    assert tuner.selection_log[-1]['selected_index']==2
    observe_costs(tuner,[99.])
    assert tuner.phase=='confirm' and not tuner.confirmation
    observe_costs(tuner,[8.7]*4)
    assert tuner.stable and tuner.policy==policies[2] and tuner.frozen_at_step==17
    frozen=tuner.steps,tuner.search_seconds,len(tuner.selection_log)
    observe_costs(tuner,[99.]*10)
    assert frozen==(tuner.steps,tuner.search_seconds,len(tuner.selection_log))


def test_earlier_challenger_keeps_gain_guard_and_reserved_fallback_step():
    tuner,policies=earlier_challenger_case()
    observe_costs(tuner,[99.,8.9,8.9,8.9,8.9])
    # It remains faster than the incumbent mean, but does not clear the
    # unchanged relative-gain and observed-spread margin.
    assert tuner.phase=='fallback' and tuner.policy==policies[1]
    observe_costs(tuner,[9.])
    assert tuner.stable and tuner.frozen_at_step==18 and tuner.confirmation_fallback


@pytest.mark.parametrize('budget,enabled',[(17,True),(18,False)])
def test_earlier_confirmation_requires_full_restore_confirm_fallback_budget(budget,enabled):
    tuner,policies=earlier_challenger_case(budget=budget,enabled=enabled)
    assert tuner.policy==policies[1] and tuner.phase=='restore'
    assert not tuner.selection_log[-1]['tentative_confirmation']
    while not tuner.stable:
        tuner.observe(StepObservation(9.,1,1))
    assert tuner.steps<=budget


def test_best_observed_confirmation_option_is_boolean():
    with pytest.raises(ValueError,match='confirm_best_observed'):
        TuningLimits(confirm_best_observed='yes')


def test_earlier_periodic_challenger_uses_complete_confirmation_cycles():
    policies=[JointPolicy(0,8,recompute_regions=(name,),trim_replay_interval=cycle)
              for name,cycle in [('encoder',1),('projection',1),('activation',2),('other',1)]]
    tuner=JointTuningController(policies[0],lambda trials:policies[len(trials)],
        TuningLimits(max_steps=18,max_candidates=4,confirmation_steps=3,
                     timing_noise_multiplier=1.,validate_confirmation_timing=True,
                     confirm_best_observed=True))
    observe_costs(tuner,[99.,9.9,10.1,99.,8.9,9.1,99.,8.92,8.98,99.,12.,12.])
    assert tuner.phase=='restore' and tuner.policy==policies[2]
    assert tuner.selection_log[-1]['tentative_confirmation']
    assert tuner.confirmation_target==4
    observe_costs(tuner,[99.,8.6,8.8,8.6])
    assert not tuner.stable and len(tuner.confirmation)==3
    observe_costs(tuner,[8.8])
    assert tuner.stable and tuner.frozen_at_step==17 and tuner.policy==policies[2]
