"""Codec policies share trace identity, never snapshot decoding identity."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload.tuning import JointPolicy, JointTuningController, StepObservation
from mindspeed.core.memory.hyper_offload.tuning.session import JointOffloadSession
from mindspeed.core.memory.hyper_offload.tuning.regions import LayerRecomputeRegions
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.api.test_joint_session import Model, cpu_runtime
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.runtime.test_codec import TestCodec


def test_transport_key_changes_policy_but_not_execution_structure():
    raw = JointPolicy(0, 64, recompute_regions=("block",))
    compressed = replace(raw, codec_key="measured-profile")
    assert raw != compressed and raw.structure_key == compressed.structure_key
    assert len({raw: 1, compressed: 2}) == 2
    for key in (None, "", " "):
        with pytest.raises(ValueError, match="codec_key"):
            replace(raw, codec_key=key)


@pytest.mark.parametrize("winner", ["raw", "lossless"])
def test_transport_switch_reuses_trace_and_freezes_without_codec_work(winner):
    regions = LayerRecomputeRegions(Model().layers)
    raw = JointPolicy(0, 64)
    coded = replace(raw, codec_key="lossless")
    tuner = JointTuningController(raw, lambda _: coded)
    codec = TestCodec()
    with cpu_runtime():
        session = JointOffloadSession(tuner, regions.apply, codecs={"lossless": codec})
        manager, pool = session.residency_manager, session.residency_manager._host_pool
        for _ in range(12):
            with session:
                pass
            if not session.stable:
                session.observe(StepObservation(1 if session.executed_policy.codec_key == winner else 2, 0, 0))
            assert manager is session.residency_manager and pool is manager._host_pool
            assert manager._codec is (codec if session.executed_policy.codec_key == "lossless" else None)
        assert session.stable and tuner.frozen_at_step <= 10
        assert session.trace_collections == 1 and session.plan_builds == 2
        assert session.executed_policy.codec_key == winner
        counters = (session.switch_seconds, session.codec_prepare_seconds, session.plan_seconds)
        with patch.object(session, "_prepare_codec", side_effect=AssertionError("stable preparation")):
            with session:
                pass
        assert counters == (session.switch_seconds, session.codec_prepare_seconds, session.plan_seconds)


def test_unknown_codec_rejected_before_policy_mutation():
    raw = JointPolicy(0, 64)
    tuner = JointTuningController(raw, lambda _: None)
    selected = []
    with cpu_runtime():
        session = JointOffloadSession(tuner, lambda count: selected.append(count))
        executor = session.executor
        with pytest.raises(ValueError, match="unregistered codec"):
            session._switch_policy(replace(raw, codec_key="missing"))
        assert selected == [0] and session.executor is executor
        assert session.executed_policy == raw and session.policy_switches == 0
        with pytest.raises(ValueError, match="reserved"):
            JointOffloadSession(tuner, lambda _: None, codecs={"raw": TestCodec()})


def test_periodic_trim_covers_full_cycles_and_restarts_on_policy_switch():
    policy = JointPolicy(0, 64, trim_replay_interval=2)
    tuner = JointTuningController(policy, lambda _: None)
    trims = []
    with cpu_runtime(), patch('mindspeed.core.memory.hyper_offload.api.session._empty_accelerator_cache') as trim:
        session = JointOffloadSession(tuner, lambda _: None)
        for step in range(12):
            before = trim.call_count
            with session:
                pass
            trims.append(trim.call_count > before)
            if not session.stable:
                session.observe(StepObservation(1., 0, 0))
        assert session.stable
        assert trims == [False,True,False,True,False,True,False,True,False,True,False,True]
        assert session.trim_before_replay
    for interval in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='interval'):
            replace(policy, trim_replay_interval=interval)
    with pytest.raises(ValueError, match='periodic'):
        replace(policy, trim_before_replay=False)


def test_periodic_trim_restarts_when_a_tuning_policy_switch_reuses_trace():
    initial = JointPolicy(0, 64, trim_replay_interval=2)
    candidate = replace(initial, transfer_window_scale=2.)
    tuner = JointTuningController(initial, lambda _: candidate)
    with cpu_runtime():
        session = JointOffloadSession(tuner, lambda _: None)
        observed = []
        for step in range(9):
            with session:
                pass
            observed.append((session.executed_policy, session.last_entry_trim_performed))
            if not session.stable:
                session.observe(StepObservation(1. if session.executed_policy == candidate else 2., 0, 0))
        assert observed[2] == (initial, False)
        assert observed[3] == (candidate, True)
        assert observed[4] == (candidate, False)
        assert session.trace_collections == 1 and session.policy_switches == 1
        assert session.stable
