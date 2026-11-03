"""Global byte capacity must conserve shared time before node selection."""
import pytest

from mindspeed.core.memory.hyper_offload.tuning.global_capacity import (
    estimate_global_copy_capacity, estimate_forward_backward_capacity,
)
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace, TraceOp


def capacity(down, up):
    return estimate_global_copy_capacity(down, up,
        d2h_gibps=1., h2d_gibps=1., host_limit_bytes=2**40)


def test_overlapping_directions_cannot_each_spend_the_whole_step():
    result = capacity([(0, 1000)], [(0, 1000)])
    assert result.roundtrip_original_bytes == 2**29
    assert result.shared_server_ms == 1000
    assert result.limiting_constraints == ('shared_copy_server',)


def test_duplicate_local_windows_never_increase_global_capacity():
    result = capacity([(0, 400), (200, 700), (0, 400)], [(300, 1000), (400, 800)])
    assert result.roundtrip_original_bytes == 2**29
    assert result.d2h_available_ms == result.h2d_available_ms == 700
    assert result.shared_server_ms == 1000
    assert [(s.start_ms, s.finish_ms) for s in result.segments] == [(0, 300), (300, 700), (700, 1000)]


def test_phase_capacity_can_be_tighter_than_total_step_capacity():
    result = estimate_global_copy_capacity([(0, 400)], [(400, 1000)],
        d2h_gibps=2., h2d_gibps=1., host_limit_bytes=2**40)
    assert result.roundtrip_original_bytes == int(.6 * 2**30)
    assert result.limiting_constraints == ('h2d_windows',)


def test_host_limit_and_missing_direction_bound_the_same_global_budget():
    result = estimate_global_copy_capacity([(0, 1000)], [(0, 1000)],
        d2h_gibps=1., h2d_gibps=1., host_limit_bytes=123)
    assert result.roundtrip_original_bytes == 123
    assert result.limiting_constraints == ('host_capacity',)
    assert capacity([], [(0, 1000)]).roundtrip_original_bytes == 0


def test_removing_recomputation_shrinks_capacity_before_node_selection():
    trace = ActivationTrace([TraceOp('any_forward', 1000.), TraceOp('any_backward', 1000.)], {},
                            d2h_bandwidth_gbps=1., h2d_bandwidth_gbps=1., memory_limit_bytes=1)
    full = estimate_forward_backward_capacity(trace, 0, host_limit_bytes=2**40)
    reduced = estimate_forward_backward_capacity(trace, 0, host_limit_bytes=2**40,
                                                  removed_backward_compute_ms=500.)
    assert full.roundtrip_original_bytes == 2**30
    assert reduced.roundtrip_original_bytes == 2**29
    assert reduced.h2d_available_ms == 500.
    assert reduced.limiting_constraints == ('h2d_windows',)
    with pytest.raises(ValueError, match='removed recomputation'):
        estimate_forward_backward_capacity(trace, 0, host_limit_bytes=2**40,
                                            removed_backward_compute_ms=1001.)


def test_global_histogram_prices_packet_mix_instead_of_all_bytes_at_slowest_rate():
    trace = ActivationTrace([TraceOp('forward', 1000.), TraceOp('backward', 1000.)], {},
        d2h_bandwidth_gbps=1., h2d_bandwidth_gbps=1., memory_limit_bytes=1,
        transfer_cost_samples=((2**20, 10., 10.), (2**24, 20., 20.)))
    result = estimate_forward_backward_capacity(trace, 0, host_limit_bytes=2**40,
                                                size_histogram=((2**20, 1), (2**24, 100)))
    expected = (2**20 + 100*2**24) / 2**30 / 2.01
    assert result.d2h_gibps == pytest.approx(expected)
    assert result.roundtrip_original_bytes > 5*(2**20/.01)


@pytest.mark.parametrize('windows', [[(2, 1)], [(0, float('nan'))], [(True, 2)], [(-1, 2)]])
def test_invalid_timeline_is_not_treated_as_free_capacity(windows):
    with pytest.raises(ValueError):
        capacity(windows, [(0, 10)])
