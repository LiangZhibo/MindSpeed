from types import SimpleNamespace
from copy import deepcopy

import torch

from mindspeed.core.memory.hyper_offload.tuning.phase_memory import PhaseMemoryObserver


class Executor:
    op_idx=0
    in_opaque_region=True
    def on_op_end(self,result):return result
    def finish(self):return 'trace','guide'


def test_phase_observer_preserves_gradient_and_restores_executor_and_hooks():
    torch.manual_seed(71)
    reference=torch.nn.Linear(8,8);candidate=deepcopy(reference)
    x=torch.randn(3,8)
    expected=reference(x).square().sum();expected.backward()
    executor=Executor();end=executor.on_op_end;finish=executor.finish
    session=SimpleNamespace(mode='warmup',executor=executor)
    regions=SimpleNamespace(_modules={'arbitrary':(None,None,candidate)},selected={'arbitrary'})
    observer=PhaseMemoryObserver(session,regions,reader=lambda:(10,20,30,40))
    value=executor.on_op_end(candidate(x))
    observer.finish_forward()
    executor.op_idx=9
    value.square().sum().backward()
    assert executor.finish()==('trace','guide')
    assert observer.closed and executor.on_op_end==end and executor.finish==finish
    assert not candidate._forward_hooks
    assert observer.report['samples']==[(0,10,20,30,40)]
    assert observer.report['regions']['arbitrary'][0]['backward_ops']==[9]
    for left,right in zip(candidate.parameters(),reference.parameters()):
        torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
    executor.on_op_end(candidate(x))
    assert len(observer.samples)==1


def test_bounded_observation_and_exception_cleanup_are_explicit():
    executor=Executor();end=executor.on_op_end
    session=SimpleNamespace(mode='warmup',executor=executor)
    regions=SimpleNamespace(_modules={},selected=set())
    observer=PhaseMemoryObserver(session,regions,reader=lambda:(1,2,3,4),max_ops=1)
    executor.on_op_end(None);executor.on_op_end(None)
    assert observer.truncated and len(observer.samples)==1
    observer.close();observer.close()
    assert executor.on_op_end==end


def test_allocator_counters_share_one_snapshot():
    from types import SimpleNamespace
    from mindspeed.core.memory.hyper_offload.tuning.phase_memory import _memory
    calls=[]
    def stats():
        calls.append(1)
        return {'allocated_bytes.all.current':10,'allocated_bytes.all.peak':20,
                'reserved_bytes.all.current':30,'reserved_bytes.all.peak':40}
    assert _memory(SimpleNamespace(memory_stats=stats))==(10,20,30,40)
    assert calls==[1]
