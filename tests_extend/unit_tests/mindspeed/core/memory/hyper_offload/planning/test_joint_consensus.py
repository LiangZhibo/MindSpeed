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
