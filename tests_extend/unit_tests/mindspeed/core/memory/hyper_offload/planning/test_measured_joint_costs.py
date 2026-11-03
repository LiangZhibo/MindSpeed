"""Counterexamples for compression/recomputation proposal economics."""
from dataclasses import replace

import pytest

from mindspeed.core.memory.hyper_offload.tuning.controller import JointPolicy, Trial
from mindspeed.core.memory.hyper_offload.tuning.costs import (
    MeasuredJointNeighborhood, RegionAlternative, TransferDemand, TransportMeasurement,
)

MIB = 1024**2


def evidence():
    return dict(raw=TransportMeasurement("raw", 1., 1.), codecs=(), regions=(), current_transfers=(),
                d2h_overlap_ms=0., h2d_overlap_ms=0.,
                device_headroom_bytes=1024*MIB, host_headroom_bytes=1024*MIB)


def initial(names=("stage",)):
    return Trial(JointPolicy(0, 64, recompute_regions=names), samples=[100.])


def codec(**changes):
    return replace(TransportMeasurement("lossless", 1., 1., packet_ratio=.25,
                                       encode_ms_per_mib=.05, decode_ms_per_mib=.05,
                                       lossless_verified=True), **changes)


def test_codec_can_make_more_offload_less_recompute_profitable():
    data = evidence()
    data['codecs'] = (codec(),)
    data['regions'] = (RegionAlternative(("stage",), ("stage.cheap",), 100.,
                                        (TransferDemand(100*MIB, "bfloat16"),)),)
    proposal = MeasuredJointNeighborhood(lambda: data)
    selected = proposal((initial(),))
    assert selected.codec_key == 'lossless' and selected.recompute_regions == ('stage.cheap',)
    assert proposal.estimates[0]['gain_ms'] == 0  # Raw keeps the checkpoint.
    assert proposal.estimates[1]['gain_ms'] == pytest.approx(40.)


def test_already_hidden_traffic_can_make_compression_unprofitable():
    data = evidence()
    data['codecs'] = (codec(),)
    data['current_transfers'] = (TransferDemand(1000*MIB, 'bfloat16', 2000., 2000.),)
    data['d2h_overlap_ms'] = data['h2d_overlap_ms'] = 2000.
    data['regions'] = (RegionAlternative(('stage',), (), 100., (TransferDemand(100*MIB, 'bfloat16'),)),)
    assert MeasuredJointNeighborhood(lambda: data)((initial(),)) is None


def test_ratio_does_not_shrink_packet_or_host_capacity():
    data = evidence()
    data['codecs'] = (codec(packet_ratio=.01, temporary_device_bytes=1025*MIB),)
    data['regions'] = (RegionAlternative(('stage',), (), 100., (TransferDemand(100*MIB, 'bfloat16'),)),)
    assert MeasuredJointNeighborhood(lambda: data)((initial(),)) is None
    data['codecs'] = (codec(packet_ratio=.01, additional_host_reservation_bytes=1025*MIB),)
    assert MeasuredJointNeighborhood(lambda: data)((initial(),)) is None


@pytest.mark.parametrize('change,dtype', [({'max_bytes':10*MIB}, 'bfloat16'),
                                          ({}, 'float32'), ({}, 'unknown'),
                                          ({'lossless_verified':False}, 'bfloat16')])
def test_no_extrapolated_codec_savings_outside_verified_support(change, dtype):
    data = evidence()
    data['codecs'] = (codec(**change),)
    data['regions'] = (RegionAlternative(('stage',), (), 100., (TransferDemand(100*MIB, dtype),)),)
    assert MeasuredJointNeighborhood(lambda: data)((initial(),)) is None


def test_opaque_region_renaming_does_not_change_the_decision():
    choices = []
    for prefix in ('encoder.block', 'unfamiliar.network.branch'):
        data = evidence()
        data['regions'] = (RegionAlternative((prefix,), (prefix+'.cheap',), 300.,
                                            (TransferDemand(100*MIB),)),)
        selected = MeasuredJointNeighborhood(lambda: data)((initial((prefix,)),))
        choices.append((selected.codec_key, selected.recompute_regions[0].removeprefix(prefix)))
    assert choices == [('raw', '.cheap'), ('raw', '.cheap')]


def test_combined_memory_bounds_and_semantic_conflicts_are_checked():
    data = evidence()
    data['regions'] = tuple(RegionAlternative((name,), (name+'.point',), 1., (),
                                              additional_device_bytes=700*MIB) for name in ('a', 'b'))
    trial = initial(('a','b'))
    selected = MeasuredJointNeighborhood(lambda: data)((trial,))
    assert selected.recompute_regions == ('a.point','b')
    data['device_headroom_bytes'] = 2048*MIB
    data['conflicts'] = (('a.point','b.point'),)
    selected = MeasuredJointNeighborhood(lambda: data)((trial,))
    assert selected.recompute_regions == ('a.point','b')


def test_evidence_is_consumed_once_and_never_repeated_after_candidates():
    data = evidence()
    data['regions'] = (RegionAlternative(('stage',), (), 1., ()),)
    calls = []
    def provide():
        calls.append(1)
        return data
    propose = MeasuredJointNeighborhood(provide)
    trial = initial()
    selected = propose((trial,))
    assert propose((trial,Trial(selected,samples=[90.]))) is None
    assert calls == [1]


def test_missing_evidence_and_oversized_search_keep_bounded_behavior():
    assert MeasuredJointNeighborhood(lambda: None)((initial(),)) is None
    data = evidence()
    data['codecs'] = (codec(),)
    with pytest.raises(ValueError, match='capacity'):
        MeasuredJointNeighborhood(lambda: data, max_profiles=1)((initial(),))


def test_many_packets_cannot_each_claim_the_same_overlap_window():
    data = evidence()
    data['d2h_overlap_ms'] = data['h2d_overlap_ms'] = 200.
    data['codecs'] = (codec(),)
    # Each packet individually fits, but their serial DMA exceeds the window.
    demands = tuple(TransferDemand(100*MIB, 'bfloat16', 200., 200.) for _ in range(10))
    data['regions'] = (RegionAlternative(('stage',), (), 100., demands),)
    assert MeasuredJointNeighborhood(lambda: data)((initial(),)) is None
