"""External correctness/performance harness for the frozen native Qwen entry.

Full snapshots in correctness mode intentionally perturb timings; those runs
are never accepted as performance evidence. Runtime source stays unmodified.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

config=json.loads(Path(sys.argv[1]).read_text())
root=Path(config['source'])
output=Path(config['output'])
output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK'])
import torch
import torch_npu
from torch.distributed.tensor import DTensor
torch.npu.set_device(rank)
torch.use_deterministic_algorithms(config['deterministic'])
torch.npu.set_per_process_memory_fraction(config['allocator_fraction'],rank)
entry=root/'examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_qwen25_native_mindspeed_fsdp2.py'
sys.argv=[str(entry),'--model-path',config['model'],'--seq-length','4096',
          '--layers','48','--profile-path',str(output),'--recompute-layers',str(config['recompute_layers']),
          '--extra-steps',str(config['steps']-1),'--use-torch-fsdp2',
          '--fsdp2-config-path',str(entry.parent/'native_fsdp2_config.yaml'),'--no-profile']
if config['offload']:
    assert config['mask_communication']
    sys.argv += ['--hyper-offload','--offload-device-mb','2048','--offload-host-mb','131072',
                 '--offload-planner','peak','--offload-communication-masking']
sys.path.insert(0,str(entry.parent))
spec=importlib.util.spec_from_file_location('qualification_entry',entry)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured={}
diagnostic=config['mode']=='correctness'

def write(name,value):
    (output/f'{name}_rank{rank}.json').write_text(json.dumps(value,indent=2)+'\n')

def tensor_digest(value):
    if value is None:
        return None
    local=value.to_local() if isinstance(value,DTensor) else value
    cpu=local.detach().cpu().contiguous()
    raw=cpu.view(torch.uint8).numpy()
    return dict(shape=list(cpu.shape),dtype=str(cpu.dtype),bytes=cpu.numel()*cpu.element_size(),
                finite=bool(torch.isfinite(cpu).all()),
                sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest())

def full_state(engine,initial=False):
    rows=[]
    for name,parameter in engine.named_parameters():
        canonical=name.replace('._checkpoint_wrapped_module.','.')
        row=dict(name=canonical,source_name=name,parameter=tensor_digest(parameter))
        if not initial:
            row['gradient']=tensor_digest(parameter.grad)
            assert row['gradient'] is not None and row['gradient']['finite'],canonical
        assert row['parameter']['finite'],canonical
        rows.append(row)
    assert len({r['name'] for r in rows})==len(rows)
    return rows

original_make=module.make_engine
def capture_engine(*args,**kwargs):
    value=original_make(*args,**kwargs)
    captured['engine']=value[0]
    if diagnostic:
        began=time.time()
        rows=full_state(value[0],initial=True)
        write('initial_state',dict(rank=rank,status='complete',tensors=rows,started=began,finished=time.time()))
    return value
module.make_engine=capture_engine

original_init=module.OffloadSession.__init__
def capture_session(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    captured['session']=self
module.OffloadSession.__init__=capture_session

if diagnostic:
    original_randint=torch.randint
    def capture_input(*args,**kwargs):
        value=original_randint(*args,**kwargs)
        if tuple(value.shape)==(1,4096) and value.device.type=='npu':
            assert 'input' not in captured
            captured['input']=tensor_digest(value)
            write('input',dict(rank=rank,status='complete',tensor=captured['input'],
                               seed=1234+rank,scope='fixed input_ids reused by unmodified entry each step'))
            torch.randint=original_randint
        return value
    torch.randint=capture_input
    original_step=torch.optim.SGD.step
    selected=set(config['audit_steps'])
    captured['step']=0
    def audited_step(self,*args,**kwargs):
        result=original_step(self,*args,**kwargs)
        captured['step']+=1
        step=captured['step']
        if step in selected:
            torch.npu.synchronize()
            began=time.time()
            rows=full_state(captured['engine'])
            write(f'full_step{step}',dict(rank=rank,status='complete',step=step,tensors=rows,
                                        started=began,finished=time.time(),scope='all local gradients and updated parameters'))
            if step==config['steps']:
                captured['last_rows']=rows
            print(f'FULL_STEP_AUDIT_COMPLETE rank={rank} step={step} tensors={len(rows)}',flush=True)
        return result
    torch.optim.SGD.step=audited_step

write('launch',dict(time=time.time(),rank=rank,executable=sys.executable,argv=sys.argv,
                   torch=torch.__version__,torch_npu=torch_npu.__version__,source=str(root),config=config,
                   cpu_affinity=sorted(os.sched_getaffinity(0)),deterministic=torch.are_deterministic_algorithms_enabled(),
                   performance_usable=not diagnostic))
module.main()
assert torch.are_deterministic_algorithms_enabled()==config['deterministic']
torch.npu.synchronize()
began=time.time()
if diagnostic:
    assert captured['step']==config['steps'] and 'input' in captured and 'last_rows' in captured
rows=captured['last_rows'] if diagnostic else full_state(captured['engine'])
session=captured.get('session')
actions=[]
guide=[]
binding_policy=None
if session is not None:
    schedule=session.executor._schedule
    actions=[dict(phase=phase,op=op,storage_id=a.storage_id,kind=a.kind.name)
             for phase in ('pre','post') for op,group in sorted(getattr(schedule,phase).items()) for a in group]
    guide=[dict(name=g.name,input_bindings=g.input_bindings,output_bindings=g.output_bindings)
           for g in session.executor._guide]
    managed=getattr(session.executor,'_managed_sids',None)
    binding_policy=dict(original_positions=sum(len(g['output_bindings']) for g in guide),
                        replay_positions=sum(len(g['output_bindings']) if managed is None else
                                             sum(sid in managed for sid in g['output_bindings'].values()) for g in guide))
assert bool(session)==config['offload']
write('full_audit',dict(status='complete',rank=rank,started=began,finished=time.time(),
                       scope='all final local parameter and gradient shards',tensors=rows,
                       schedule=actions,guide=guide,binding_policy=binding_policy,
                       allocator_stats=torch.npu.memory_stats(),deterministic=torch.are_deterministic_algorithms_enabled()))
print(f'FULL_AUDIT_COMPLETE rank={rank} tensors={len(rows)}',flush=True)
