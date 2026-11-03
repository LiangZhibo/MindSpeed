"""Policy metadata must preserve distinctions smaller than float32 precision."""
from dataclasses import replace
import math

from mindspeed.core.memory.hyper_offload.tuning import JointPolicy
from mindspeed.core.memory.hyper_offload.tuning.consensus import policy_fingerprint


def test_consensus_distinguishes_every_policy_field_without_float_rounding():
    policy = JointPolicy(0, 2**45, transfer_window_scale=1.0,
                         recompute_regions=('0.mlp', '1.mlp'), resident_regions=('2.self_attn',))
    alternatives = [
        replace(policy, resident_mb=policy.resident_mb + 1),
        replace(policy, transfer_window_scale=math.nextafter(1.0, math.inf)),
        replace(policy, masking=False, transfer_window_scale=None),
        replace(policy, recompute_regions=('0.mlp', '3.mlp')),
        replace(policy, resident_regions=('3.self_attn',)),
        replace(policy, limit_prefetch_window=False),
        replace(policy, prefetch_on_copy_stream=True),
        replace(policy, trim_before_replay=False),
        JointPolicy(1, policy.resident_mb),
    ]
    expected = policy_fingerprint(policy)
    assert len(expected) == 4 and all(-2**63 <= word < 2**63 for word in expected)
    assert all(policy_fingerprint(other) != expected for other in alternatives)
    assert len({policy_fingerprint(other) for other in alternatives}) == len(alternatives)
    assert policy_fingerprint(JointPolicy(1, 64)) != policy_fingerprint(JointPolicy(2, 64))
    assert policy_fingerprint(JointPolicy(0, 64)) != policy_fingerprint(JointPolicy(0, 64, masking=False))


def test_consensus_uses_canonical_region_order_and_covers_none_window():
    first = JointPolicy(0, 64, recompute_regions=('1.mlp', '0.mlp'))
    second = JointPolicy(0, 64, recompute_regions=('0.mlp', '1.mlp'))
    assert policy_fingerprint(first) == policy_fingerprint(second)
    assert policy_fingerprint(first) != policy_fingerprint(replace(first, transfer_window_scale=0.0))


def test_full_decision_encodes_real_confirmation_fallback_and_frozen_state():
    from mindspeed.core.memory.hyper_offload.tuning.controller import JointTuningController, StepObservation, TuningLimits
    from mindspeed.core.memory.hyper_offload.tuning.consensus import decision_fingerprint
    first,second=JointPolicy(2,8),JointPolicy(1,8)
    tuner=JointTuningController(first,lambda trials:second if len(trials)==1 else None,
        TuningLimits(max_steps=10,timing_noise_multiplier=1,validate_confirmation_timing=True))
    states={}
    for value in (99,10,10,99,8,8,11,11,10):
        states[tuner.phase]=decision_fingerprint(tuner)
        tuner.observe(StepObservation(value,100,200))
    states[tuner.phase]=decision_fingerprint(tuner)
    assert 'fallback' in states and 'stable' in states
    assert states['fallback'][2:6]==states['stable'][2:6]==policy_fingerprint(first)
    assert states['fallback']!=states['stable'] and tuner.frozen_at_step==9
    frozen=decision_fingerprint(tuner)
    tuner.observe(StepObservation(1000,100,200))
    assert decision_fingerprint(tuner)==frozen
    import pytest
    tuner.phase='unsupported'
    with pytest.raises(ValueError):decision_fingerprint(tuner)


def test_confirmation_reference_and_cycle_target_are_in_consensus():
    from mindspeed.core.memory.hyper_offload.tuning.controller import JointTuningController,Trial
    from mindspeed.core.memory.hyper_offload.tuning.consensus import decision_fingerprint
    from dataclasses import replace
    initial=JointPolicy(3,8)
    tuner=JointTuningController(initial,lambda _:None)
    tuner.trials=[Trial(initial,samples=[12,12]),Trial(replace(initial,recompute_count=2),samples=[10,10])]
    first=decision_fingerprint(tuner)
    tuner.confirmation_reference_index=1
    second=decision_fingerprint(tuner)
    tuner.confirmation_target+=1
    assert len({first,second,decision_fingerprint(tuner)})==3
