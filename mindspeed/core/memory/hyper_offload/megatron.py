# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Opt-in adapter for one complete Megatron forward/backward schedule."""

from functools import wraps
import inspect

from . import OffloadConfig, OffloadSession
from .planning import GreedyResidencyPlanner, PeakResidencyPlanner


def config_from_args(args):
    """Build an independent activation-offload configuration."""
    if args.hyper_offload_planner == "peak":
        planner = PeakResidencyPlanner(
            enable_communication_masking=args.hyper_offload_communication_masking
        )
    else:
        planner = GreedyResidencyPlanner()
    return OffloadConfig(
        max_resident_activation_mb=args.hyper_offload_device_mb,
        max_offload_activation_mb=args.hyper_offload_host_mb,
        planner=planner,
    )


def wrap_schedule(schedule, config, session_factory=OffloadSession):
    """Share one plan across training iterations, bypassing evaluation schedules.

    The context spans every microbatch and backward call in the schedule.
    Optimizer state remains owned by the existing optimizer outside this scope.
    """
    signature = inspect.signature(schedule)
    session = None

    @wraps(schedule)
    def wrapped(*args, **kwargs):
        nonlocal session
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("forward_only", False):
            return schedule(*args, **kwargs)
        if session is None:
            session = session_factory(config)
        with session:
            return schedule(*args, **kwargs)

    return wrapped


def get_forward_backward_func_wrapper(factory):
    """Wrap the training factory through MindSpeedPatchesManager."""
    @wraps(factory)
    def wrapped(*args, **kwargs):
        from mindspeed.args_utils import get_full_args

        schedule = factory(*args, **kwargs)
        graph_args = get_full_args()
        if not getattr(graph_args, "hyper_offload", False):
            return schedule
        return wrap_schedule(schedule, config_from_args(graph_args))

    return wrapped
