"""Two-rank native-FSDP numerical qualification, including real raw transfers."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--dropout-probability', type=float, default=0.0)
args = parser.parse_args()
assert 0 <= args.dropout_probability < 1
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
entry = root / 'examples/qwen25_32b_fsdp2_a2/native_fsdp2/run_qwen25_native_mindspeed_fsdp2.py'
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), '--model-path', '/data/disk1/machangwei/Qwen2.5-32B',
            '--profile-path', str(output), '--use-torch-fsdp2',
            '--fsdp2-config-path', str(entry.parent / 'native_fsdp2_config.yaml'), '--no-profile']
spec = importlib.util.spec_from_file_location('native_entry', entry)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from mindspeed.args_utils import get_mindspeed_args, set_full_args
from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, JointTuningController, StepObservation, TuningLimits
from mindspeed.core.memory.hyper_offload.tuning.regions import qwen_recompute_regions
from mindspeed.core.memory.hyper_offload.tuning.session import JointOffloadSession

rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(rank)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
full = vars(get_mindspeed_args(get_defaults=True)).copy()
full.update(vars(module.parse_args()))
full['init_model_with_meta_device'] = False
set_full_args(argparse.Namespace(**full))
dist.init_process_group('hccl')
module.apply_hccl_premul_sum_patch()

# Metadata must not lose the low bits of byte counts or policy digests.
from mindspeed.core.memory.hyper_offload.tuning.consensus import policy_fingerprint
metadata = torch.tensor([2**45 + 1 + rank, 2**45 + 100 - rank], dtype=torch.int64, device='npu')
dist.all_reduce(metadata, op=dist.ReduceOp.MAX)
assert metadata.cpu().tolist() == [2**45 + 2, 2**45 + 100]
for diverge in (False, True):
    policy = JointPolicy(0, 16 + rank if diverge else 16, recompute_regions=('0.mlp',))
    words = torch.tensor(policy_fingerprint(policy), dtype=torch.int64, device='npu')
    minimum, maximum = words.clone(), words.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert bool(torch.equal(minimum, maximum)) != diverge


class GatedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(512, 2048)
        self.up_proj = torch.nn.Linear(512, 2048)
        self.down_proj = torch.nn.Linear(2048, 512)
        self.act_fn = (torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Dropout(args.dropout_probability))
                       if args.dropout_probability else torch.nn.SiLU())

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = GatedMLP()
        self.self_attn = torch.nn.Linear(512, 512)
        if args.dropout_probability:
            self.self_attn = torch.nn.Sequential(self.self_attn, torch.nn.Dropout(args.dropout_probability))
        self.input_layernorm = torch.nn.LayerNorm(512)
        self.post_attention_layernorm = torch.nn.LayerNorm(512)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(6)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.float().square().mean()


def make_engine():
    torch.manual_seed(781)
    model = Model().to(dtype=torch.bfloat16)
    config = module.TransformerConfig(num_layers=6, hidden_size=512, num_attention_heads=8,
                                      bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=True)
    return module.TorchFullyShardedDataParallel(config, module.DistributedDataParallelConfig(), model,
                                                sub_modules_to_wrap={Block}, process_group=dist.group.WORLD)


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


reference = make_engine()
candidate = make_engine()
opt_ref = torch.optim.SGD(reference.parameters(), lr=.001)
opt = torch.optim.SGD(candidate.parameters(), lr=.001)
regions = qwen_recompute_regions(candidate.module.layers)
initial = JointPolicy(0, 16, recompute_regions=('0', '1', '2.mlp', '3.self_attn', '4.input_layernorm'))
proposal = JointPolicy(0, 16, transfer_window_scale=1,
                       recompute_regions=('0.mlp.pointwise', '1.mlp.gate', '2.self_attn', '3.mlp', '5.post_attention_layernorm'),
                       limit_prefetch_window=False, resident_regions=('4.self_attn', '4.mlp.gate', '5.mlp'))
budget_proposal = replace(proposal, resident_mb=12, prefetch_on_copy_stream=True, trim_before_replay=False,
                          defer_h2d_release=True, trim_after_offload=True)
tuner = JointTuningController(initial, lambda trials: proposal if len(trials) == 1 else budget_proposal,
                              TuningLimits(max_steps=12))
session = JointOffloadSession(tuner, apply_policy=regions.apply_policy, host_mb=512)
manager = session.residency_manager
copy_count = [0, 0]
deferred_release_count = [0]
copy_d2h, copy_h2d = manager.copy_d2h, manager.copy_h2d
release_device = manager.release_device


def counted_d2h(sid):
    copy_count[0] += 1
    return copy_d2h(sid)


def counted_h2d(sid):
    copy_count[1] += 1
    return copy_h2d(sid)


def counted_release(sid):
    buffer = manager._residency.get(sid)
    if (manager.defer_h2d_release and buffer is not None and buffer.device_buffer is not None
            and buffer.device_event is not None and buffer.device_copy_stream_recorded):
        deferred_release_count[0] += 1
    return release_device(sid)


manager.copy_d2h, manager.copy_h2d = counted_d2h, counted_h2d
manager.release_device = counted_release
torch.manual_seed(90 + rank)
inputs = torch.randn(2, 512, 512, dtype=torch.bfloat16, device='npu')
records = []
for step in range(1, 17):
    opt_ref.zero_grad(set_to_none=True)
    opt.zero_grad(set_to_none=True)
    x_ref, x = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
    cpu_rng, npu_rng = torch.get_rng_state().clone(), torch.npu.get_rng_state().clone()
    expected = reference(x_ref)
    expected.backward()
    expected_cpu_rng, expected_npu_rng = torch.get_rng_state().clone(), torch.npu.get_rng_state().clone()
    expected_loss = expected.detach().cpu()
    expected_input = x_ref.grad.detach().cpu()
    expected_gradients = [local(p.grad).detach().cpu() for p in reference.parameters()]
    opt_ref.step()
    expected_parameters = [local(p).detach().cpu() for p in reference.parameters()]
    del expected, x_ref
    torch.set_rng_state(cpu_rng)
    torch.npu.set_rng_state(npu_rng)
    with session:
        actual = candidate(x)
        actual.backward()
    assert torch.equal(torch.get_rng_state(), expected_cpu_rng), 'checkpoint changed CPU RNG progression'
    assert torch.equal(torch.npu.get_rng_state(), expected_npu_rng), 'checkpoint changed NPU RNG progression'
    torch.testing.assert_close(actual.detach().cpu(), expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(x.grad.detach().cpu(), expected_input, rtol=0, atol=0)
    for p, expected_gradient in zip(candidate.parameters(), expected_gradients):
        torch.testing.assert_close(local(p.grad).detach().cpu(), expected_gradient, rtol=0, atol=0)
    opt.step()
    for p, expected_parameter in zip(candidate.parameters(), expected_parameters):
        torch.testing.assert_close(local(p).detach().cpu(), expected_parameter, rtol=0, atol=0)
    torch.npu.synchronize()
    records.append(dict(step=step, loss=float(actual.item()), policy=asdict(session.executed_policy),
                        copies=list(copy_count), phase=tuner.phase, rng_state_exact=True))
    del actual, x, expected_gradients, expected_parameters
    if not session.stable:
        # Force a return to the initial policy to exercise cached-plan reuse.
        session.observe(StepObservation(1 if session.executed_policy == initial else 2,
                                        torch.npu.max_memory_allocated(), manager._host_pool.total_allocated))
    print('QUALIFIED', rank, step, records[-1], flush=True)
assert session.stable and tuner.frozen_at_step <= 12
assert session.trace_collections == 2 and session.policy_switches == 3 and session.plan_builds == 3
assert session.executed_policy == initial
assert copy_count[0] > 0 and copy_count[1] > 0, 'qualification must include actual offload'
assert deferred_release_count[0] > 0, 'qualification must exercise allocator-fenced H2D release'
assert not manager.defer_h2d_release, 'restoring the initial policy must restore release behavior'
assert session.post_offload_trim_calls > 0, 'qualification must execute the post-offload cache trim'
assert session.executor._after_offload is None, 'restoring the initial policy must disable the trim callback'
result = dict(status='passed', rank=rank, steps=16, records=records,
              dropout_probability=args.dropout_probability, rng_state_exact_steps=len(records),
              integer_metadata='exact int64 max above2**45; equal and deliberately different policy digests checked',
              frozen_at_step=tuner.frozen_at_step, copies=copy_count,
              deferred_h2d_releases=deferred_release_count[0],
              post_offload_trim_calls=session.post_offload_trim_calls,
              post_offload_trim_seconds=session.post_offload_trim_seconds,
              trace_collections=session.trace_collections, plan_seconds=session.plan_seconds,
              scope='loss, all local parameter gradients, all updated parameters, input gradients; exact comparison',
              host_peak_bytes=manager._host_pool.total_allocated)
(output / f'qualification_rank{rank}.json').write_text(json.dumps(result, indent=2))
dist.barrier()
dist.destroy_process_group()
