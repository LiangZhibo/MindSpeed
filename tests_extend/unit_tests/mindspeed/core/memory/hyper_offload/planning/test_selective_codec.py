from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mindspeed.core.memory.compression.transport import TransferContext, TransferNode
from mindspeed.core.memory.hyper_offload.tuning.calibrated_transport import CalibratedPacketCost
from mindspeed.core.memory.hyper_offload.tuning.peak_windows import PeakWindowModel
from mindspeed.core.memory.hyper_offload.tuning.selective_codec import FrozenNodeCodec, codec_scopes
from .test_peak_windows import trace


def profile(size=100, ratio=.5, codec_ms=3.):
    return CalibratedPacketCost(size,1.,1.,codec_ms,codec_ms,ratio,1000,1000)


def test_a_hidden_packet_can_offer_capacity_for_more_offload_despite_inclusive_codec_cost():
    model=PeakWindowModel(trace())
    scopes=codec_scopes(model,(profile(),),('anonymous',),device_headroom=64*1024**2,host_headroom=64*1024**2)
    assert len(scopes)==1 and scopes[0].purpose=='transfer_capacity'
    assert scopes[0].nodes==((0,'post',0,100),)
    assert scopes[0].estimated_link_bytes_saved==100
    # A candidate is not an asserted gain: the inclusive3ms cost may worsen
    # pressure despite a useful ratio. A complete training trial decides.
    assert scopes[0].codec_peak_bytes>=scopes[0].raw_peak_bytes


def test_only_peak_crossing_supported_packet_sizes_can_enter_scope():
    assert not codec_scopes(PeakWindowModel(trace(limit=1000)),(profile(),),(),
                           device_headroom=10**8,host_headroom=10**8)
    model=PeakWindowModel(trace())
    for profiles in ((),(profile(size=200),),(profile(ratio=1.1),)):
        assert not codec_scopes(model,profiles,(),device_headroom=10**8,host_headroom=10**8)


def test_capacity_uses_worst_case_device_and_full_new_host_bucket_not_ratio():
    model=PeakWindowModel(trace())
    wide=codec_scopes(model,(profile(ratio=.01),),(),device_headroom=10**8,host_headroom=10**8)[0]
    assert wide.extra_device_bytes>3*16384
    assert wide.extra_host_bytes>100
    for device,host in ((wide.extra_device_bytes-1,10**8),(10**8,wide.extra_host_bytes-1)):
        assert not codec_scopes(model,(profile(ratio=.01),),(),device_headroom=device,host_headroom=host)


def test_snapshot_selector_is_exact_and_unknown_direction_storage_or_dtype_falls_back():
    scope=codec_scopes(PeakWindowModel(trace()),(profile(),),(),device_headroom=10**8,host_headroom=10**8)[0]
    backend=SimpleNamespace(accepts=lambda c:c.dtype==torch.bfloat16)
    codec=FrozenNodeCodec(backend,scope)
    request=TransferContext(TransferNode(0,'post'),0,100,torch.bfloat16,torch.device('cpu'))
    assert codec.accepts(request)
    for other in (replace(request,node=None),replace(request,node=TransferNode(0,'pre')),
                  replace(request,storage_id=1),replace(request,original_bytes=200),
                  replace(request,dtype=torch.float32)):
        assert not codec.accepts(other)


def test_same_numeric_node_in_different_graph_has_a_different_profile_key():
    model=PeakWindowModel(trace())
    a=codec_scopes(model,(profile(),),('cut_a',),device_headroom=10**8,host_headroom=10**8)[0]
    b=codec_scopes(model,(profile(),),('cut_b',),device_headroom=10**8,host_headroom=10**8)[0]
    assert a.key!=b.key and a.nodes==b.nodes


def test_cost_override_shares_queues_and_never_mutates_raw_model():
    model=PeakWindowModel(trace(packets=2,latency=3.,pressure_at=3,consumer=7,limit=100))
    raw=model.forecast(model.opportunities)
    compressed=model.forecast(model.opportunities,transfer_costs={o.key:(1.,1.) for o in model.opportunities})
    assert compressed.max_h2d_lateness_ms<raw.max_h2d_lateness_ms
    assert model.forecast(model.opportunities)==raw
    for invalid in ({(99,):(.1,.1)},{model.opportunities[0].key:(0.,1.)},
                    {model.opportunities[0].key:(float('nan'),1.)}):
        with pytest.raises(ValueError):model.forecast(model.opportunities,transfer_costs=invalid)
