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
    from mindspeed.core.memory.hyper_offload.tuning.proposals import QwenRegionNeighborhood
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
