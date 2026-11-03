from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning.codec_calibration import ActivationSamples, CalibrationLimits


def manager_for(sizes):
    buffers = {sid:SimpleNamespace(host_buffer=None,logical_nbytes=size,storage_dtype=torch.bfloat16,
                                  host_encoding=None,host_event=Mock()) for sid,size in enumerate(sizes)}
    manager = SimpleNamespace(_residency=buffers,_host_pool=SimpleNamespace(total_allocated=sum(sizes)))
    def copy(sid):
        buffers[sid].host_buffer=torch.arange(buffers[sid].logical_nbytes,dtype=torch.uint8)
    manager.copy_d2h=copy
    return manager


def test_owned_snapshots_do_not_alias_recycled_pool_and_instance_hook_is_retired():
    manager=manager_for((16,32));original=manager.copy_d2h
    untouched=manager_for((16,));other=untouched.copy_d2h
    samples=ActivationSamples(CalibrationLimits(sample_bytes=64))
    samples.install(manager,1000)
    manager.copy_d2h(0)
    expected=samples.items[0]['data'].clone()
    manager._residency[0].host_buffer.zero_()
    assert torch.equal(samples.items[0]['data'],expected)
    manager._residency[0].host_event.synchronize.assert_called_once()
    assert untouched.copy_d2h is other
    samples.close()
    assert manager.copy_d2h is original and samples.closed
    manager.copy_d2h(1)
    assert len(samples.items)==1
    samples.release()
    assert samples.bytes==0 and not samples.items and samples.peak_bytes==16


def test_sample_cap_evicts_smaller_strata_before_allocating_new_snapshot():
    manager=manager_for((8,16,32))
    samples=ActivationSamples(CalibrationLimits(sample_bytes=40,size_groups=2))
    samples.install(manager,1000)
    for sid in range(3):manager.copy_d2h(sid)
    assert samples.bytes<=40 and len({row['group'] for row in samples.items})<=2
    assert samples.items[-1]['bytes']==32
    assert all(row['bytes']!=16 for row in samples.items)
    samples.close()


def test_host_bound_unknown_dtype_and_preexisting_host_data_are_excluded():
    manager=manager_for((16,16,16))
    manager._residency[1].storage_dtype=None
    manager.copy_d2h(2)
    samples=ActivationSamples(CalibrationLimits(sample_bytes=64))
    samples.install(manager,63)  # Existing48B plus16B snapshot would exceed.
    for sid in range(3):manager.copy_d2h(sid)
    assert not samples.items
    samples.close()


def test_capture_failure_still_allows_exact_instance_cleanup():
    manager=manager_for((16,));original=manager.copy_d2h
    manager._residency[0].host_event.synchronize.side_effect=RuntimeError('transfer failed')
    samples=ActivationSamples();samples.install(manager,1000)
    with pytest.raises(RuntimeError,match='transfer failed'):manager.copy_d2h(0)
    samples.close()
    assert manager.copy_d2h is original and not samples.items


def test_calibration_limits_are_finite_and_require_repeated_measurements():
    for kwargs in ({'sample_bytes':0},{'size_groups':4},{'samples_per_group':3},
                   {'repetitions':1},{'repetitions':6},{'repetitions':True}):
        with pytest.raises(ValueError):CalibrationLimits(**kwargs)
