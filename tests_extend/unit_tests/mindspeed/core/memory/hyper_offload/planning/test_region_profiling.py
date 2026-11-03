"""First-forward observer must not alter training or survive into replay."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning.profiling import RegionProfiler, _boundary


class Event:
    created = 0
    def __init__(self):
        type(self).created += 1
        self.index = type(self).created
    def record(self, stream):
        pass
    def synchronize(self):
        pass
    def elapsed_time(self, other):
        return float(other.index-self.index)


def registry(model):
    return SimpleNamespace(_modules={'unknown.branch':(None,None,model)},_callables={},observer=None)


def profiler(model, **kwargs):
    session = SimpleNamespace(mode='warmup',executor=SimpleNamespace(op_idx=0,in_opaque_region=False))
    return RegionProfiler(registry(model),event_factory=Event,stream_provider=lambda:None,
                          session_provider=lambda:session,**kwargs)


def test_hooks_preserve_rng_outputs_gradients_and_are_removed_before_backward():
    torch.manual_seed(17)
    reference = torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.Dropout(.2),torch.nn.Linear(16,8))
    candidate = deepcopy(reference)
    x=torch.randn(3,8)
    torch.manual_seed(31)
    expected=reference(x).square().sum();expected.backward();expected_rng=torch.get_rng_state().clone()
    observer=profiler(candidate).start()
    torch.manual_seed(31)
    actual=candidate(x).square().sum()
    report=observer.finish_forward()
    assert observer.closed and observer.regions.observer is None
    assert not candidate._forward_hooks and not candidate._forward_pre_hooks
    actual.backward()
    assert torch.equal(torch.get_rng_state(),expected_rng)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(candidate.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert len(report['regions'])==1 and report['regions'][0]['complete']
    before=Event.created
    candidate(x)
    assert Event.created==before and len(observer.rows)==1
    with pytest.raises(RuntimeError,match='one-shot'):
        observer.start()


def test_boundary_bytes_count_alias_storage_once_and_are_not_saved_bytes():
    x=torch.empty(32)
    shapes, size=_boundary((x,x[3:]))
    assert size==x.untyped_storage().nbytes() and len(shapes)==2


def test_record_cap_keeps_running_model_and_marks_missing_evidence():
    model=torch.nn.Linear(8,8)
    observer=profiler(model,max_records=1).start()
    model(torch.ones(1,8));model(torch.ones(1,8))
    report=observer.finish_forward()
    assert report['truncated'] and len(report['regions'])==1
    assert report['memory_scope']=='unique boundary storage bytes, not saved activations'


def test_opaque_and_incomplete_regions_are_not_claimed_as_visible_evidence():
    model=torch.nn.Linear(8,8)
    session=SimpleNamespace(mode='warmup',executor=SimpleNamespace(op_idx=4,in_opaque_region=True))
    observer=RegionProfiler(registry(model),event_factory=Event,stream_provider=lambda:None,
                            session_provider=lambda:session).start()
    token=observer.begin('unknown.branch',(torch.ones(1,8),))
    observer.abort(token)
    report=observer.finish_forward()
    assert report['regions'][0]['opaque'] and not report['regions'][0]['complete']
