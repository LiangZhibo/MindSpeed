"""Real-sample diagnostic ownership and hard bounds, independent of hardware."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch


def harness():
    path=next(p for p in Path(__file__).resolve().parents if (p/'tools/hyper_offload').is_dir())/'tools/hyper_offload/compression_applicability.py'
    spec=importlib.util.spec_from_file_location('sample_harness',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def add(budget,size,value=0):
    group=budget.admit(size)
    if group is not None:budget.retain(group,torch.full((size,),value,dtype=torch.uint8),{})
    return group


def test_bounded_owned_strata_evict_smaller_without_tensor_comparison():
    budget=harness().SampleBudget(max_bytes=64,max_groups=2,per_group=2)
    assert add(budget,8,1) is not None
    assert add(budget,8,2) is not None
    assert add(budget,16) is not None
    assert add(budget,32) is not None
    assert [x['bytes'] for x in budget.items]==[16,32]
    assert add(budget,32,3) is not None
    assert [x['bytes'] for x in budget.items]==[32,32]
    assert add(budget,32,4) is None
    assert budget.bytes==budget.peak_bytes==64
    assert add(budget,66) is None


@pytest.mark.parametrize('size',[True,0,-2,3,1.5])
def test_invalid_sample_never_consumes_memory(size):
    budget=harness().SampleBudget(max_bytes=64)
    assert budget.admit(size) is None
    assert not budget.items and not budget.bytes


def test_capture_waits_for_host_writer_and_owns_snapshot(monkeypatch):
    from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
    log=[];host=torch.zeros(16,dtype=torch.uint8)
    class Fence:
        def synchronize(self):log.append('wait');host.fill_(27)
    buffer=SimpleNamespace(host_buffer=None,storage_dtype=torch.bfloat16,host_encoding=None,
                           logical_nbytes=16,host_event=Fence())
    manager=SimpleNamespace(_residency={1:buffer},_transfer_node=None,
                            _host_pool=SimpleNamespace(total_allocated=16,max_host_bytes=128))
    def original(self,sid):
        log.append('copy');self._residency[sid].host_buffer=host
    monkeypatch.setattr(ResidencyManager,'copy_d2h',original)
    sampler=harness().WarmupSamples(max_bytes=64);sampler.install()
    try:
        ResidencyManager.copy_d2h(manager,1)
        assert log==['copy','wait']
        owned=sampler.budget.items[0]['data'];host.fill_(99)
        assert torch.equal(owned,torch.full((16,),27,dtype=torch.uint8))
        ResidencyManager.copy_d2h(manager,1)
        assert sampler.copy_observations==1 and len(sampler.budget.items)==1
    finally:sampler.close()
    assert ResidencyManager.copy_d2h is original
    sampler.close()


def test_capture_respects_training_host_capacity_and_non_bf16(monkeypatch):
    from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
    host=torch.ones(16,dtype=torch.uint8)
    buffer=SimpleNamespace(host_buffer=None,storage_dtype=torch.bfloat16,host_encoding=None,
                           logical_nbytes=16,host_event=None)
    manager=SimpleNamespace(_residency={1:buffer},_transfer_node=None,
                            _host_pool=SimpleNamespace(total_allocated=120,max_host_bytes=128))
    def original(self,sid):self._residency[sid].host_buffer=host
    monkeypatch.setattr(ResidencyManager,'copy_d2h',original)
    sampler=harness().WarmupSamples(max_bytes=64);sampler.install()
    try:
        ResidencyManager.copy_d2h(manager,1)
        assert not sampler.budget.items
        manager._host_pool.total_allocated=0;buffer.host_buffer=None;buffer.storage_dtype=torch.float32
        ResidencyManager.copy_d2h(manager,1)
        assert not sampler.budget.items
    finally:sampler.close()
