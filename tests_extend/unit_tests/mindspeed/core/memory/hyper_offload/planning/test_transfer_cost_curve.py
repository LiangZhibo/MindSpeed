"""Bounded calibration, latency interpolation, conservative unsupported sizes."""
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import pytest

from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace
from mindspeed.core.memory.hyper_offload.ir.transfer import transfer_duration_ms
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.tuning.session import JointOffloadSession

MIB=1024**2
PROBE='mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth'


def test_interpolates_measured_latency_and_keeps_legacy_outside_support():
    points=((16*MIB,2.,3.),(64*MIB,5.,6.))
    assert transfer_duration_ms(40*MIB,16,points)==3.5
    assert transfer_duration_ms(40*MIB,16,points,'h2d')==4.5
    assert transfer_duration_ms(16*MIB,16,points)==2.
    for size in (MIB,128*MIB):
        assert transfer_duration_ms(size,16,points)==transfer_duration_ms(size,16)
    noisy=((16*MIB,9.,8.),(64*MIB,5.,6.))
    assert transfer_duration_ms(64*MIB,16,noisy)==9.  # Slow observations never silently discarded.


@pytest.mark.parametrize('points',[
    ((1,1,1),(1,2,2)),((2,1,1),(1,2,2)),((True,1,1),),((1,0,1),),
    ((1,float('nan'),1),),((1,1,float('inf')),),((1,2),),tuple((i+1,1,1) for i in range(5))])
def test_trace_rejects_invalid_calibration(points):
    with pytest.raises(ValueError,match='transfer calibration'):
        ActivationTrace(transfer_cost_samples=points).validate()


def test_actual_traffic_ranking_capacity_and_four_point_limit():
    manager=MagicMock();executor=WarmupExecutor(manager,128*MIB,transfer_probe_budget=lambda:128*MIB)
    executor._transfer_size_counts={2*MIB:20,8*MIB:20,32*MIB:20,64*MIB:20,256*MIB:100}
    seen=[]
    def probe(*,size_bytes=16*MIB,diagnostics):
        seen.append(size_bytes);diagnostics.update(status='measured',bytes=size_bytes,
            d2h_samples_ms=(1,1,99,1,1),h2d_samples_ms=(2,2,2,2,2))
        return 16.,8.
    with patch(PROBE,side_effect=probe):trace,_=executor.finish()
    assert seen==[16*MIB,64*MIB,32*MIB,8*MIB]
    assert trace.transfer_cost_samples==tuple((n,1,2) for n in sorted(seen))
    assert not executor._transfer_size_counts
    trace.validate()


def test_all_probes_respect_small_headroom_and_failed_allocations_are_not_measurements():
    executor=WarmupExecutor(MagicMock(),MIB,transfer_probe_budget=lambda:MIB)
    executor._transfer_size_counts={MIB:10,2*MIB:100}
    seen=[]
    def probe(*,size_bytes=16*MIB,diagnostics):
        seen.append(size_bytes);diagnostics.update(status='allocation_fallback',bytes=size_bytes)
        return 1.,1.
    with patch(PROBE,side_effect=probe):trace,_=executor.finish()
    assert seen==[MIB] and not trace.transfer_cost_samples
    assert trace.d2h_bandwidth_gbps==trace.h2d_bandwidth_gbps==1.


def test_probe_capacity_intersects_device_host_and_managed_pool_limits():
    session=object.__new__(JointOffloadSession)
    session.controller=SimpleNamespace(limits=SimpleNamespace(device_peak_bytes=100*MIB),
                                       policy=SimpleNamespace(resident_mb=80))
    pool=SimpleNamespace(max_host_bytes=100*MIB,total_allocated=70*MIB)
    session.residency_manager=SimpleNamespace(_host_pool=pool)
    backend=SimpleNamespace(memory_stats=lambda:{'allocated_bytes.all.current':60*MIB})
    with patch('torch.accelerator.is_available',return_value=True), \
         patch('torch.accelerator.current_accelerator',return_value='cpu'), \
         patch('torch.get_device_module',return_value=backend):
        assert session._transfer_probe_budget()==30*MIB
        pool.total_allocated=0
        assert session._transfer_probe_budget()==40*MIB
        session.controller.limits.device_peak_bytes=50*MIB
        assert session._transfer_probe_budget()==0
