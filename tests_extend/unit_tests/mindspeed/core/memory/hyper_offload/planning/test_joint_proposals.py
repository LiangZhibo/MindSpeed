from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.proposals import JointNeighborhood


def test_named_region_moves_refine_or_restore_from_measured_cost():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33)))
    proposer = QwenRegionNeighborhood(48, 33, probe='mlp')
    first = Trial(initial, samples=[12, 12])
    middle = proposer((first,))
    assert '0.mlp' in middle.recompute_regions and '0.self_attn' not in middle.recompute_regions
    assert '47.input_layernorm' in middle.recompute_regions
    refined = proposer((first, Trial(middle, samples=[10, 10])))
    assert '0.mlp.gate' in refined.recompute_regions and '0.mlp' not in refined.recompute_regions
    fallback = proposer((first, Trial(middle, samples=[14, 14])))
    assert '0' in fallback.recompute_regions and '0.input_layernorm' not in fallback.recompute_regions
    assert '33.mlp.gate' in fallback.recompute_regions


def test_cheap_regions_compare_both_masking_choices_without_another_trace():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33)))
    proposer = QwenRegionNeighborhood(48, 33)
    first = Trial(initial, samples=[12, 12])
    legacy = proposer((first,))
    release_only = proposer((first, Trial(legacy, samples=[11, 11])))
    assert len(legacy.recompute_regions) == 63
    assert '33.input_layernorm' in legacy.recompute_regions
    assert '33.mlp.gate' not in legacy.recompute_regions
    assert release_only.structure_key == legacy.structure_key
    assert release_only.transfer_window_scale == 1 and legacy.transfer_window_scale is None
    assert not release_only.limit_prefetch_window
    assert proposer((first, Trial(legacy), Trial(release_only))) is None


def test_joint_neighborhood_advances_using_actual_step_cost():
    propose = JointNeighborhood(48)
    initial = Trial(JointPolicy(33, 2048), [12, 12])
    first = propose((initial,))
    assert first.recompute_count == 21 and first.transfer_window_scale == 1
    improved = Trial(first, [10, 10])
    assert propose((initial, improved)).recompute_count == 9


def test_residency_probe_only_keeps_attention_in_the_offloaded_suffix():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='attention_resident')
    norm = propose((initial,))
    resident = propose((initial, Trial(norm, [11, 11])))
    assert norm.recompute_regions == resident.recompute_regions
    assert not norm.resident_regions and len(resident.resident_regions) == 15
    assert '33.self_attn' in resident.resident_regions and '32.self_attn' not in resident.resident_regions
    assert norm.structure_key != resident.structure_key
    assert propose((initial, Trial(norm), Trial(resident))) is None


def test_suffix_residency_probe_adds_only_gate_operator_groups():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='suffix_resident')
    attention = propose((initial,))
    gate = propose((initial, Trial(attention, [11, 11])))
    assert len(attention.resident_regions) == 15 and len(gate.resident_regions) == 30
    assert '33.mlp.gate' in gate.resident_regions and '33.mlp' not in gate.resident_regions
    assert attention.recompute_regions == gate.recompute_regions
    assert propose((initial, Trial(attention), Trial(gate))) is None


def test_mlp_residency_probe_refines_individual_subregions_without_overlap():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='mlp_resident')
    gate = propose((initial,))
    mixed = propose((initial, Trial(gate, [11, 11])))
    assert '40.mlp' in mixed.resident_regions and '39.mlp.gate' in mixed.resident_regions
    assert '40.mlp.gate' not in mixed.resident_regions and '39.mlp' not in mixed.resident_regions
    assert len(gate.resident_regions) == len(mixed.resident_regions) == 30
    assert gate.structure_key != mixed.structure_key
    assert propose((initial, Trial(gate), Trial(mixed))) is None


def test_compute_probe_keeps_attention_instead_of_offloading_its_saves():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='compute_resident')
    mlp = propose((initial,))
    assert len(mlp.resident_regions) == 48 and '0.mlp' in mlp.recompute_regions
    gate = propose((initial, Trial(mlp, [11, 11])))
    assert '0.mlp.gate' in gate.recompute_regions and not gate.limit_prefetch_window
    restored = propose((initial, Trial(mlp, [13, 13])))
    assert '0' in restored.recompute_regions and len(restored.resident_regions) == 15
    assert '33.input_layernorm' in restored.recompute_regions


def test_budget_probe_reuses_structure_and_only_reduces_managed_residency():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='budget_resident')
    fine = propose((initial,))
    smaller = propose((initial, Trial(fine, [11, 11])))
    assert smaller.structure_key == fine.structure_key and smaller.resident_mb == 1024
    assert len(smaller.resident_regions) == 15 and not smaller.limit_prefetch_window
    assert propose((initial, Trial(fine), Trial(smaller))) is None


def test_joint_neighborhood_reverses_direction_after_regression():
    propose = JointNeighborhood(48)
    initial = Trial(JointPolicy(33, 2048), [12, 12])
    worse = Trial(propose((initial,)), [14, 14])
    next_policy = propose((initial, worse))
    assert next_policy.recompute_count == 45 and next_policy.transfer_window_scale == 1
    assert propose((initial, worse, Trial(next_policy, [13, 13]))) is None


def test_stream_probe_changes_execution_without_invalidating_structure():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='stream_resident')
    fine = propose((initial,))
    stream = propose((initial, Trial(fine, [11, 11])))
    assert fine.structure_key == stream.structure_key and fine != stream
    assert not fine.prefetch_on_copy_stream and stream.prefetch_on_copy_stream
    assert fine.resident_mb == stream.resident_mb == 2048
    assert propose((initial, Trial(fine), Trial(stream))) is None


def test_cache_probe_keeps_region_and_memory_constraints_fixed():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='cache_resident')
    fine = propose((initial,))
    cached = propose((initial, Trial(fine, [11, 11])))
    assert fine.structure_key == cached.structure_key and fine != cached
    assert fine.trim_before_replay and not cached.trim_before_replay
    assert fine.resident_mb == cached.resident_mb == 2048
    assert fine.prefetch_on_copy_stream == cached.prefetch_on_copy_stream
    assert propose((initial, Trial(fine), Trial(cached))) is None


def test_pointwise_probe_adds_only_cheap_mlp_checkpoint_group():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='pointwise_resident')
    fine = propose((initial,))
    pointwise = propose((initial, Trial(fine, [11, 11])))
    assert set(pointwise.recompute_regions) - set(fine.recompute_regions) == {
        f'{i}.mlp.pointwise' for i in range(33, 48)}
    assert pointwise.resident_regions == fine.resident_regions
    assert pointwise.structure_key != fine.structure_key
    assert not any(name.endswith('.mlp.gate') for name in pointwise.recompute_regions)
    assert propose((initial, Trial(fine), Trial(pointwise))) is None


def test_gate_checkpoint_probe_recomputes_projection_without_limiting_prefetch():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='gate_recompute_resident')
    fine = propose((initial,))
    gate = propose((initial, Trial(fine, [11, 11])))
    assert set(gate.recompute_regions) - set(fine.recompute_regions) == {
        f'{i}.mlp.gate' for i in range(33, 48)}
    assert gate.resident_regions == fine.resident_regions
    assert not gate.limit_prefetch_window and gate.transfer_window_scale == 1
    assert not any(name.endswith('.mlp.pointwise') for name in gate.recompute_regions)
    assert propose((initial, Trial(fine), Trial(gate))) is None


def test_placement_probe_preserves_region_counts_and_moves_offload_to_early_forward():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='placement_resident')
    fine = propose((initial,))
    early = propose((initial, Trial(fine, [11, 11])))
    assert len(fine.recompute_regions) == len(early.recompute_regions) == 63
    assert len(fine.resident_regions) == len(early.resident_regions) == 15
    assert '0' not in early.recompute_regions and '15' in early.recompute_regions
    assert '0.input_layernorm' in early.recompute_regions and '0.self_attn' in early.resident_regions
    assert not early.limit_prefetch_window and early.transfer_window_scale == fine.transfer_window_scale
    assert early.structure_key != fine.structure_key
    assert propose((initial, Trial(fine), Trial(early))) is None


def test_window_probe_reuses_trace_and_budget_with_bounded_lookahead():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='window_resident')
    fine = propose((initial,))
    bounded = propose((initial, Trial(fine, [11, 11])))
    assert fine.structure_key == bounded.structure_key and fine != bounded
    assert fine.resident_mb == bounded.resident_mb == 2048
    assert not fine.limit_prefetch_window and bounded.limit_prefetch_window
    assert bounded.transfer_window_scale == 8 * fine.transfer_window_scale
    assert propose((initial, Trial(fine), Trial(bounded))) is None


def test_release_probe_changes_only_allocator_release_wait_policy():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='release_resident')
    fine = propose((initial,))
    deferred = propose((initial, Trial(fine, [11, 11])))
    assert fine.structure_key == deferred.structure_key and fine != deferred
    assert fine.resident_mb == deferred.resident_mb == 2048
    assert not fine.defer_h2d_release and deferred.defer_h2d_release
    assert fine.transfer_window_scale == deferred.transfer_window_scale
    assert fine.limit_prefetch_window == deferred.limit_prefetch_window
    assert propose((initial, Trial(fine), Trial(deferred))) is None


def test_configured_window_probe_preserves_control_and_deduplicates_same_schedule():
    from dataclasses import replace
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    control = QwenRegionNeighborhood(48, 33, probe='window_resident')((initial,))
    for multiplier, bounded in ((4, True), (16, True), (8, False)):
        propose = QwenRegionNeighborhood(48, 33, probe='window_resident',
                                        window_probe_multiplier=multiplier, window_probe_limit_prefetch=bounded)
        assert propose((initial,)) == control
        changed = propose((initial, Trial(control, [11, 11])))
        assert changed == replace(control, transfer_window_scale=multiplier, limit_prefetch_window=bounded)
        assert changed.structure_key == control.structure_key
        assert propose((initial, Trial(control), Trial(changed))) is None
    unchanged = QwenRegionNeighborhood(48, 33, probe='window_resident',
                                       window_probe_multiplier=1, window_probe_limit_prefetch=False)
    assert unchanged((initial, Trial(control))) is None


def test_window_probe_rejects_invalid_timing_controls():
    import unittest
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    check = unittest.TestCase()
    for multiplier in (-1, float('inf'), float('nan')):
        with check.assertRaisesRegex(ValueError, 'window_probe_multiplier'):
            QwenRegionNeighborhood(48, 33, window_probe_multiplier=multiplier)
    with check.assertRaisesRegex(ValueError, 'window_probe_limit_prefetch'):
        QwenRegionNeighborhood(48, 33, window_probe_limit_prefetch=1)


def test_phase_trim_probe_keeps_fine_window_control_and_changes_only_cleanup():
    from dataclasses import replace
    import unittest
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='phase_trim_resident')
    fine = propose((initial,))
    changed = propose((initial, Trial(fine, [11, 11])))
    assert not fine.trim_after_offload and fine.transfer_window_scale == 8 and fine.limit_prefetch_window
    assert changed == replace(fine, trim_after_offload=True)
    assert changed.structure_key == fine.structure_key and len(fine.resident_regions) == 15
    assert propose((initial, Trial(fine), Trial(changed))) is None
    with unittest.TestCase().assertRaisesRegex(ValueError, 'trim_after_offload'):
        JointPolicy(0, 16, trim_after_offload=1)


def test_pointwise_window_probe_changes_only_pointwise_regions_at_the_same_finite_window():
    from dataclasses import replace
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='pointwise_window_resident')
    fine = propose((initial,))
    pointwise = propose((initial, Trial(fine, [11, 11])))
    assert fine.transfer_window_scale == 8 and fine.limit_prefetch_window
    assert not fine.trim_after_offload and fine.trim_before_replay
    assert pointwise == replace(fine, recompute_regions=fine.recompute_regions +
                                tuple(f'{i}.mlp.pointwise' for i in range(33, 48)))
    assert pointwise.structure_key != fine.structure_key
    assert propose((initial, Trial(fine), Trial(pointwise))) is None


def test_phase_cache_probe_compares_cleanup_placement_with_the_same_regions_and_window():
    from dataclasses import replace
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='phase_cache_resident')
    both = propose((initial,))
    middle_only = propose((initial, Trial(both, [11, 11])))
    assert both.trim_before_replay and both.trim_after_offload
    assert both.transfer_window_scale == 8 and both.limit_prefetch_window
    assert middle_only == replace(both, trim_before_replay=False)
    assert both.structure_key == middle_only.structure_key and len(both.resident_regions) == 15
    assert propose((initial, Trial(both), Trial(middle_only))) is None
    assert initial.policy.trim_before_replay and not initial.policy.trim_after_offload


def test_partial_compute_probe_keeps_mlp_checkpoints_and_changes_only_bounded_attention_group():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='partial_compute_resident')
    fine = propose((initial,))
    partial = propose((initial, Trial(fine, [11, 11])))
    assert {name for name in partial.recompute_regions if '.' not in name} == {str(i) for i in range(25)}
    assert {name for name in partial.recompute_regions if name.endswith('.mlp')} == {
        f'{i}.mlp' for i in range(25, 33)}
    assert set(partial.resident_regions) - set(fine.resident_regions) == {f'{i}.self_attn' for i in range(25, 33)}
    assert not any(f'{i}.mlp' in partial.recompute_regions for i in range(33, 48))
    assert partial.resident_mb == fine.resident_mb and not partial.defer_h2d_release
    assert partial.transfer_window_scale == fine.transfer_window_scale and not partial.limit_prefetch_window
    assert propose((initial, Trial(fine), Trial(partial))) is None


def test_gate_compute_probe_directly_qualifies_prefix_operator_group_tradeoff():
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
    initial = Trial(JointPolicy(0, 2048, recompute_regions=tuple(str(i) for i in range(33))), [12, 12])
    propose = QwenRegionNeighborhood(48, 33, probe='gate_compute_resident')
    fine = propose((initial,))
    gate = propose((initial, Trial(fine, [11, 11])))
    assert {name for name in gate.recompute_regions if name.endswith('.mlp.gate')} == {
        f'{i}.mlp.gate' for i in range(33)}
    assert not any('.' not in name or name.endswith('.mlp') for name in gate.recompute_regions)
    assert len(gate.recompute_regions) == 33 + 2 * 48
    assert set(gate.resident_regions) == {f'{i}.self_attn' for i in range(48)}
    assert gate.resident_mb == fine.resident_mb and not gate.defer_h2d_release
    assert not gate.limit_prefetch_window and gate.transfer_window_scale == fine.transfer_window_scale
    assert propose((initial, Trial(fine), Trial(gate))) is None
