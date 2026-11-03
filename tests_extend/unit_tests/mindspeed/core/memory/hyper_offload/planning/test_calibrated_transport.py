"""Real-data calibration must not reward hidden traffic or discount capacity."""
import pytest
from mindspeed.core.memory.hyper_offload.tuning.calibrated_transport import common_profiles, profiles_from_calibration


def row(size=1024, raw=.01, codec=.007, ratio=.5, name='sample'):
    result=lambda cost,packet:dict(d2h_inclusive_seconds=cost,h2d_inclusive_seconds=cost,
                                  packet_bytes=packet,lossless_exact=True)
    return dict(status='complete',bytes=size,source_sha256=name,required_extra_device_bytes=6*size,
                required_extra_host_bytes=8*size,measurements=[dict(results={
                    'raw':result(raw,size),'codec':result(codec,int(size*ratio))}) for _ in range(3)])


def test_ratio_does_not_reward_hidden_transfer_or_discount_capacity():
    cost=profiles_from_calibration(dict(status='complete',rows=[row()]))[0]
    assert cost.exposed_costs(0,0)==(20.,14.)
    assert cost.exposed_costs(20,20)==(0.,4.)
    assert not cost.capacity_fits(6*1024-1,8*1024)
    assert cost.capacity_fits(6*1024,8*1024)
    assert cost.supports(1024,'bfloat16')
    assert not cost.supports(2048,'bfloat16') and not cost.supports(1024,'float32')


def test_adverse_snapshot_and_packet_expansion_remain_in_profile():
    fast=row(name='a');slow=row(raw=.008,codec=.014,ratio=1.1,name='b')
    cost=profiles_from_calibration(dict(status='complete',rows=[fast,slow]))[0]
    assert cost.raw_d2h_ms==8 and cost.codec_d2h_ms==14
    assert cost.packet_ratio>1 and cost.sample_hashes==('a','b')
    raw,codec=cost.exposed_costs(0,0)
    assert codec>raw


def test_unverified_or_invalid_measurements_never_form_eligible_profiles():
    assert profiles_from_calibration(dict(status='skipped_capacity',rows=[row()]))==()
    skipped=row();skipped['status']='skipped_capacity'
    assert profiles_from_calibration(dict(status='complete',rows=[skipped]))==()
    for value in (0,float('nan'),float('inf'),True):
        invalid=row();invalid['measurements'][0]['results']['codec']['h2d_inclusive_seconds']=value
        with pytest.raises(ValueError):profiles_from_calibration(dict(status='complete',rows=[invalid]))
    invalid=row();invalid['measurements'][0]['results']['codec']['lossless_exact']=False
    with pytest.raises(ValueError,match='losslessly'):profiles_from_calibration(dict(status='complete',rows=[invalid]))


def test_profile_count_is_bounded_and_aliases_cannot_mutate_frozen_evidence():
    rows=[row(size=2**i,name=str(i)) for i in range(8,13)]
    costs=profiles_from_calibration(dict(status='complete',rows=rows))
    assert [x.nbytes for x in costs]==[1024,2048,4096]
    rows[-1]['required_extra_device_bytes']=0
    assert costs[-1].extra_device_bytes==6*4096
    for slack in (-1,float('nan'),True):
        with pytest.raises(ValueError):costs[0].exposed_costs(slack,0)


def test_rank_consensus_requires_common_support_and_retains_adverse_costs():
    first=profiles_from_calibration(dict(status='complete',rows=[row(1024),row(2048)]))
    second=profiles_from_calibration(dict(status='complete',rows=[row(1024,raw=.006,codec=.02,name='rank1')]))
    common=common_profiles([first,second])
    assert len(common)==1 and common[0].nbytes==1024
    assert common[0].raw_d2h_ms==6 and common[0].codec_h2d_ms==20
    assert common[0].sample_hashes==('rank1','sample')
    assert common_profiles([first,()])==() and common_profiles([])==()
    with pytest.raises(ValueError,match='duplicate'):
        common_profiles([(first[0],first[0])])


@pytest.mark.parametrize('ratio', [.25,.75,1.,1.1])
@pytest.mark.parametrize('codec', [(12.,22.),(3.,40.),(30.,4.)])
def test_nonpositive_unhidden_gain_never_becomes_positive_at_other_slack(ratio,codec):
    from mindspeed.core.memory.hyper_offload.tuning.calibrated_transport import CalibratedPacketCost
    cost=CalibratedPacketCost(1024,10.,20.,*codec,ratio,0,0)
    raw,packed=cost.exposed_costs(0,0)
    assert packed>=raw
    # These are all breakpoints of the piecewise-linear exposed-cost model.
    for outbound in (0.,10.,10.*ratio,100.):
        for inbound in (0.,20.,20.*ratio,100.):
            raw,packed=cost.exposed_costs(outbound,inbound)
            assert packed>=raw
