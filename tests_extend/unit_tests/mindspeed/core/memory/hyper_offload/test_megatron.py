# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Verify the adapter owns an entire schedule and never an optimizer step."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindspeed.core.memory.hyper_offload.megatron import config_from_args, wrap_schedule
from mindspeed.core.memory.hyper_offload.planning import GreedyResidencyPlanner, PeakResidencyPlanner


def test_training_reuses_one_session_across_complete_schedules():
    events = []

    class Session:
        def __init__(self, config):
            events.append("create")

        def __enter__(self):
            events.append("enter")

        def __exit__(self, *exc):
            events.append("exit")

    def schedule(microbatches, *, forward_only=False):
        for index in range(microbatches):
            events.extend([("forward", index), ("backward", index)])
        return "losses"

    wrapped = wrap_schedule(schedule, object(), session_factory=Session)
    assert wrapped(2) == "losses"
    assert wrapped(2) == "losses"
    assert events.count("create") == 1
    assert events == ["create", "enter", ("forward", 0), ("backward", 0),
                      ("forward", 1), ("backward", 1), "exit", "enter",
                      ("forward", 0), ("backward", 0), ("forward", 1), ("backward", 1), "exit"]


def test_evaluation_does_not_create_or_advance_training_session():
    create_session = Mock()

    def schedule(forward_only=False):
        return forward_only

    wrapped = wrap_schedule(schedule, object(), session_factory=create_session)
    assert wrapped(True) is True
    create_session.assert_not_called()


def test_schedule_failure_exits_session_and_preserves_exception():
    session = Mock()
    session.__enter__ = Mock()
    session.__exit__ = Mock(return_value=False)

    def schedule():
        raise ValueError("schedule failed")

    wrapped = wrap_schedule(schedule, object(), session_factory=lambda config: session)
    with pytest.raises(ValueError, match="schedule failed"):
        wrapped()
    assert session.__exit__.call_args.args[0] is ValueError


@pytest.mark.parametrize("kind,expected", [("peak", PeakResidencyPlanner), ("greedy", GreedyResidencyPlanner)])
def test_cli_configuration_selects_planner_and_budgets(kind, expected):
    config = config_from_args(SimpleNamespace(
        hyper_offload_planner=kind, hyper_offload_communication_masking=False,
        hyper_offload_device_mb=24, hyper_offload_host_mb=128,
    ))
    assert isinstance(config.planner, expected)
    assert config.max_resident_activation_mb == 24
    assert config.max_offload_activation_mb == 128
