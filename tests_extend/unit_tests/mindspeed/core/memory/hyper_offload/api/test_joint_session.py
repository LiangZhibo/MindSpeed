"""Numerical and lifetime checks across complete joint-policy transitions."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, JointTuningController, StepObservation, TuningLimits
from mindspeed.core.memory.hyper_offload.tuning.regions import LayerRecomputeRegions
from mindspeed.core.memory.hyper_offload.tuning.regions import qwen_recompute_regions
from mindspeed.core.memory.hyper_offload.tuning.session import JointOffloadSession


@contextmanager
def cpu_runtime():
    with ExitStack() as stack:
        timer = stack.enter_context(patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.DeviceTimer'))
        timer.return_value.stop.return_value = 1.0
        stack.enter_context(patch('mindspeed.core.memory.hyper_offload.execution.warmup.executor.profile_transfer_bandwidth',
                                  return_value=(16.0, 16.0)))
        stack.enter_context(patch('torch.Stream'))
        stack.enter_context(patch('torch.accelerator.current_stream'))
        yield


class Block(torch.nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, x):
        return self.dropout(torch.sin(self.linear(x))) + x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.square().sum()


@pytest.mark.parametrize('candidate,best_count,collections', [
    (JointPolicy(1, 64, transfer_window_scale=1), 1, 2),
    (JointPolicy(1, 64, transfer_window_scale=1), 3, 2),
    (JointPolicy(3, 64, transfer_window_scale=1), 3, 1),
    (JointPolicy(3, 32, transfer_window_scale=1), 3, 1),
    (JointPolicy(3, 64, prefetch_on_copy_stream=True), 3, 1),
    (JointPolicy(3, 64, trim_before_replay=False), 3, 1),
    (JointPolicy(3, 64, trim_after_offload=True), 3, 1),
    (JointPolicy(3, 64, trim_replay_interval=2), 3, 1),
])
def test_joint_switches_preserve_rng_gradients_and_updates(candidate, best_count, collections):
    torch.manual_seed(2)
    reference = Model()
    model = deepcopy(reference)
    regions = LayerRecomputeRegions(model.layers)
    original_parameters = tuple(model.parameters())
    tuner = JointTuningController(JointPolicy(3, 64), lambda _: candidate)
    opt_ref = torch.optim.SGD(reference.parameters(), lr=.001)
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    inputs = torch.randn(3, 16)
    with cpu_runtime():
        session = JointOffloadSession(tuner, regions.apply, host_mb=64)
        manager = session.residency_manager
        pool = manager._host_pool
        for step in range(12):
            opt_ref.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            x_ref = inputs.clone().requires_grad_()
            x = inputs.clone().requires_grad_()
            torch.manual_seed(100 + step)
            expected = reference(x_ref)
            expected.backward()
            torch.manual_seed(100 + step)
            with session:
                actual = model(x)
                actual.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
            for actual_p, expected_p in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=0, atol=0)
            opt.step()
            opt_ref.step()
            for actual_p, expected_p in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual_p, expected_p, rtol=0, atol=0)
            del actual, expected, x, x_ref
            if not session.stable:
                cost = 1 if session.executed_policy.recompute_count == best_count else 2
                session.observe(StepObservation(cost, 0, 0))
            assert manager.prefetch_on_copy_stream == session.executed_policy.prefetch_on_copy_stream
            assert session.trim_before_replay == session.executed_policy.trim_before_replay
            assert (session.executor._after_offload is not None) == session.executed_policy.trim_after_offload
            assert session.residency_manager is manager
            assert manager._host_pool is pool
        assert session.stable
        assert tuner.frozen_at_step <= 10
        assert regions.count == best_count
        assert session.trace_collections == collections
        assert session.plan_builds == 2
        assert not session._plans and not session._traces
        assert all(a is b for a, b in zip(original_parameters, model.parameters()))
        # Stable observations are optional and cannot restart sampling.
        session.observe(StepObservation(1, 0, 0))


def test_observation_and_switch_are_step_boundary_only():
    model = Model()
    regions = LayerRecomputeRegions(model.layers)
    tuner = JointTuningController(JointPolicy(0, 64), lambda _: None)
    with cpu_runtime():
        session = JointOffloadSession(tuner, regions.apply)
        with pytest.raises(RuntimeError, match='no completed'):
            session.observe(StepObservation(1, 0, 0))
        with session:
            with pytest.raises(RuntimeError, match='active offload'):
                regions.apply(1)
            with pytest.raises(RuntimeError, match='context has exited'):
                session.observe(StepObservation(1, 0, 0))
        with pytest.raises(RuntimeError, match='observe the completed'):
            with session:
                pass
        session.observe(StepObservation(1, 0, 0))


def test_initial_post_offload_trim_installs_callback_after_warmup_without_inventing_transfers():
    model = Model()
    regions = LayerRecomputeRegions(model.layers)
    tuner = JointTuningController(JointPolicy(3, 64, trim_after_offload=True), lambda _: None)
    with cpu_runtime():
        session = JointOffloadSession(tuner, regions.apply)
        with session:
            model(torch.randn(3, 16, requires_grad=True)).backward()
        assert session.mode == 'replay' and session.trace_collections == 1
        assert session.executor._after_offload is not None
        assert session.executor._last_offload_release_op is None
        assert session.post_offload_trim_calls == 0


class GatedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(16, 32)
        self.up_proj = torch.nn.Linear(16, 32)
        self.down_proj = torch.nn.Linear(32, 16)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FineBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = GatedMLP()
        self.input_layernorm = torch.nn.LayerNorm(16)
        self.post_attention_layernorm = torch.nn.LayerNorm(16)
        self.self_attn = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Dropout(.2))

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


@pytest.mark.parametrize('candidate_names', [('0.self_attn', '1.mlp'), ('0.mlp.gate', '1.input_layernorm'), ('0.mlp.pointwise', '1.input_layernorm')])
@pytest.mark.parametrize('resident_names', [(), ('1.self_attn', '2.mlp.gate'), ('1.self_attn', '2.mlp')])
def test_named_regions_preserve_rng_and_numerics_with_equal_size_trace_keys(candidate_names, resident_names):
    torch.manual_seed(18)
    reference = Model()
    reference.layers = torch.nn.ModuleList([FineBlock() for _ in range(3)])
    model = deepcopy(reference)
    regions = qwen_recompute_regions(model.layers)
    parameters = tuple(model.parameters())
    initial = JointPolicy(0, 64, recompute_regions=('0.mlp', '1.self_attn'))
    proposal = JointPolicy(0, 64, recompute_regions=candidate_names, transfer_window_scale=1,
                           resident_regions=resident_names)
    tuner = JointTuningController(initial, lambda _: proposal)
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.0001)
    with cpu_runtime():
        session = JointOffloadSession(tuner, apply_policy=regions.apply_policy, host_mb=64)
        for step in range(12):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(1000 + step)
            x = torch.randn(3, 16).requires_grad_()
            x_ref = x.detach().clone().requires_grad_()
            torch.manual_seed(2000 + step)
            expected = reference(x_ref)
            expected.backward()
            torch.manual_seed(2000 + step)
            with session:
                actual = model(x)
                actual.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert tuple(model.state_dict()) == tuple(reference.state_dict())
            del x, x_ref, actual, expected
            if not session.stable:
                session.observe(StepObservation(1 if session.executed_policy == initial else 2, 0, 0))
        assert session.trace_collections == 2  # Same cardinality, different structures.
        assert session.policy_switches == 2 and tuner.frozen_at_step == 9
        assert all(a is b for a, b in zip(parameters, model.parameters()))
        before = regions.selected
        for invalid in (('0', '0.mlp'), ('unknown',), ('0', '0'), ('0.mlp.gate', '0.mlp.pointwise')):
            with pytest.raises(ValueError):
                regions.apply(invalid)
            assert regions.selected == before


def test_pointwise_region_rejects_overlap_across_recompute_and_resident_before_switch():
    layers = torch.nn.ModuleList([FineBlock()])
    regions = qwen_recompute_regions(layers)
    regions.apply(('0.mlp',))
    wrapper = layers[0].mlp
    with pytest.raises(ValueError, match='semantically overlapping'):
        regions.apply_policy(JointPolicy(0, 64, recompute_regions=('0.mlp.pointwise',),
                                         resident_regions=('0.mlp.gate',)))
    assert layers[0].mlp is wrapper and regions.selected == {'0.mlp'}


def test_placement_switch_and_cached_return_preserve_full_model_updates():
    torch.manual_seed(37)
    reference = Model()
    reference.layers = torch.nn.ModuleList([FineBlock() for _ in range(4)])
    model = deepcopy(reference)
    regions = qwen_recompute_regions(model.layers)
    parameters = tuple(model.parameters())
    initial = JointPolicy(0, 64, recompute_regions=('0', '1'))
    tuner = JointTuningController(initial, QwenRegionNeighborhood(4, 2, probe='placement_resident'),
                                  TuningLimits(max_steps=12))
    optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.0001)
    with cpu_runtime():
        session = JointOffloadSession(tuner, apply_policy=regions.apply_policy, host_mb=64)
        for step in range(14):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(3000 + step)
            x = torch.randn(3, 16).requires_grad_()
            x_ref = x.detach().clone().requires_grad_()
            torch.manual_seed(4000 + step)
            expected = reference(x_ref)
            expected.backward()
            torch.manual_seed(4000 + step)
            with session:
                actual = model(x)
                actual.backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert tuple(model.state_dict()) == tuple(reference.state_dict())
            del x, x_ref, actual, expected
            if not session.stable:
                # Make the middle policy win: after moving fine regions to the
                # other end, restore their cached plan and original modules.
                cost = 1 if '2.self_attn' in session.executed_policy.resident_regions else 2
                session.observe(StepObservation(cost, 0, 0))
        assert session.trace_collections == 3 and session.plan_builds == 3
        assert session.policy_switches == 3 and tuner.frozen_at_step == 12
        assert '2.self_attn' in session.executed_policy.resident_regions
        assert all(a is b for a, b in zip(parameters, model.parameters()))


def test_generic_gated_adapter_preserves_original_forward_backward_and_rng():
    from copy import deepcopy
    from mindspeed.core.memory.hyper_offload.tuning.regions import decoder_recompute_regions
    torch.manual_seed(541)
    reference = torch.nn.Sequential(FineBlock(), FineBlock())
    candidate = deepcopy(reference)
    regions = decoder_recompute_regions(candidate, layout='gated')
    x = torch.randn(3, 16)
    for selected in ((), ('0.mlp.pointwise', '1.input_layernorm'), ('0',)):
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        regions.apply(selected)
        torch.manual_seed(98)
        expected = reference(x).square().sum()
        expected.backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(98)
        actual = candidate(x).square().sum()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng)
        for left, right in zip(candidate.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


class ParallelMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expand=torch.nn.Linear(16,32)
        self.act=torch.nn.GELU()
        self.project=torch.nn.Linear(32,16)
    def forward(self,x):
        return self.project(self.act(self.expand(x)))


class ParallelBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Dropout(.2))
        self.mlp=ParallelMLP()
        self.input_layernorm=torch.nn.LayerNorm(16)
        self.post_attention_layernorm=torch.nn.LayerNorm(16)
        self.output_dropout=torch.nn.Dropout(.3)
    def forward(self,x):
        attention=self.attention(self.input_layernorm(x))
        hidden=self.output_dropout(self.mlp(self.post_attention_layernorm(x)))
        return x+attention+hidden


def test_parallel_branch_adapter_preserves_updates_rng_and_cached_restore():
    # Adapter contract test with a synthetic parallel-residual network. It does
    # not consume held-out model traces, measurements or tuning decisions.
    from mindspeed.core.memory.hyper_offload.tuning.regions import decoder_recompute_regions
    torch.manual_seed(31)
    reference=torch.nn.Sequential(*(ParallelBlock() for _ in range(3)))
    model=deepcopy(reference)
    regions=decoder_recompute_regions(model,layout='parallel_mlp')
    original_parameters=tuple(model.parameters())
    initial=JointPolicy(0,64,recompute_regions=('0','1.mlp'),resident_regions=('2.attention',))
    candidate=JointPolicy(0,64,recompute_regions=('0.mlp.act','1.attention','2.post_attention_layernorm'),
                          resident_regions=('0.attention','2.mlp'))
    tuner=JointTuningController(initial,lambda trials:candidate if len(trials)==1 else None,TuningLimits(max_steps=12))
    ref_opt=torch.optim.SGD(reference.parameters(),lr=.001);opt=torch.optim.SGD(model.parameters(),lr=.001)
    inputs=torch.randn(2,4,16)
    with cpu_runtime():
        session=JointOffloadSession(tuner,apply_policy=regions.apply_policy,host_mb=64)
        for step in range(13):
            ref_opt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
            a,b=inputs.clone().requires_grad_(),inputs.clone().requires_grad_()
            torch.manual_seed(300+step)
            expected=reference(a).square().mean();expected.backward();expected_rng=torch.get_rng_state().clone()
            torch.manual_seed(300+step)
            with session:
                actual=model(b).square().mean();actual.backward()
            assert torch.equal(torch.get_rng_state(),expected_rng)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(b.grad,a.grad,rtol=0,atol=0)
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            ref_opt.step();opt.step()
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            del actual,expected,a,b
            if not session.stable:session.observe(StepObservation(1. if session.executed_policy==initial else 2.,0,0))
        assert tuner.stable and tuner.frozen_at_step<=12 and session.executed_policy==initial
        assert session.trace_collections==2 and session.plan_builds==2
        assert all(a is b for a,b in zip(original_parameters,model.parameters()))


def test_direct_fixed_control_has_no_tuning_observations_and_exact_replay():
    import importlib.util
    from pathlib import Path
    from unittest.mock import Mock
    root=next(p for p in Path(__file__).resolve().parents if (p/'tools/hyper_offload/fixed_control.py').exists())
    spec=importlib.util.spec_from_file_location('fixed_control',root/'tools/hyper_offload/fixed_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    torch.manual_seed(58)
    reference=Model();model=deepcopy(reference)
    regions=LayerRecomputeRegions(model.layers)
    proposal=Mock(side_effect=AssertionError('a direct fixed control must not search'))
    tuner=JointTuningController(JointPolicy(2,64,trim_replay_interval=2),proposal)
    control.make_direct_fixed(tuner)
    tuner.observe=Mock(side_effect=AssertionError('a direct fixed control must not observe'))
    opt=torch.optim.SGD(model.parameters(),lr=.001);ref_opt=torch.optim.SGD(reference.parameters(),lr=.001)
    inputs=torch.randn(3,16)
    with cpu_runtime():
        session=JointOffloadSession(tuner,regions.apply,host_mb=64)
        for step in range(12):
            opt.zero_grad(set_to_none=True);ref_opt.zero_grad(set_to_none=True)
            x,x_ref=inputs.clone().requires_grad_(),inputs.clone().requires_grad_()
            torch.manual_seed(700+step);expected=reference(x_ref);expected.backward();rng=torch.get_rng_state().clone()
            torch.manual_seed(700+step)
            with session:
                actual=model(x);actual.backward()
            if step==0:snapshot=control.finish_direct_warmup(session)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(x.grad,x_ref.grad,rtol=0,atol=0)
            assert torch.equal(torch.get_rng_state(),rng)
            for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            opt.step();ref_opt.step()
            for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            del actual,expected,x,x_ref
        assert session.stable and session.trace_collections==session.plan_builds==1
        assert not session._plans and not session._traces
        assert tuner.steps==tuner.frozen_at_step==0
        assert snapshot['timing_events']==session.timing_events
        assert snapshot['timing_event_count']==len(session.timing_events)
        assert snapshot['observation_and_consensus_seconds']==snapshot['search_seconds']==0
        proposal.assert_not_called();tuner.observe.assert_not_called()


def _load_legacy_qwen_candidate():
    # Megatron also exports an examples package; load the local fixture by path.
    import importlib.util
    from pathlib import Path
    relative = 'examples/qwen25_32b_fsdp2_a2/native_fsdp2/legacy_tuning.py'
    root = next(p for p in Path(__file__).resolve().parents if (p / relative).is_file())
    spec = importlib.util.spec_from_file_location('hyperoffload_legacy_qwen_fixture', root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QwenRegionNeighborhood


QwenRegionNeighborhood = _load_legacy_qwen_candidate()
