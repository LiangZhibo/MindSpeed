# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Test opt-in argument validation without importing the NPU feature stack."""

from argparse import ArgumentParser
from pathlib import Path
import runpy
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def feature():
    base = ModuleType("mindspeed.features_manager.feature")

    class Feature:
        def __init__(self, *args, **kwargs):
            pass

    base.MindSpeedFeature = Feature
    source = Path(__file__).resolve().parents[6] / "mindspeed/features_manager/memory/hyper_offload.py"
    with patch.dict(sys.modules, {base.__name__: base}):
        namespace = runpy.run_path(str(source))
    return namespace["HyperOffloadFeature"]()


def parse(feature, *options):
    parser = ArgumentParser()
    feature.register_args(parser)
    return parser.parse_args(options)


def test_default_does_not_register_patches(feature):
    args = parse(feature)
    manager = Mock()
    feature.register_patches(manager, args)
    manager.register_patch.assert_not_called()


def test_enabled_registers_only_the_schedule_factory(feature):
    manager = Mock()
    feature.register_patches(manager, parse(feature, "--hyper-offload"))
    assert manager.register_patch.call_count == 1
    assert manager.register_patch.call_args.args[0] == "megatron.training.training.get_forward_backward_func"


@pytest.mark.parametrize("field,value", [
    ("smart_swap", True), ("swap_layer_input", True),
    ("cuda_graph_impl", "full_iteration"), ("variable_seq_lengths", True),
    ("npu_graph_mode", "compile"), ("enable_cuda_graph", True),
    ("optimizer_cuda_graph", True),
])
def test_unsupported_execution_ownership_is_rejected(feature, field, value):
    args = parse(feature, "--hyper-offload")
    setattr(args, field, value)
    with pytest.raises(ValueError):
        feature.validate_args(args)


def test_disabled_feature_preserves_other_modes(feature):
    args = parse(feature)
    args.cuda_graph_impl = "full_iteration"
    feature.validate_args(args)


def test_invalid_budgets_are_rejected(feature):
    args = parse(feature, "--hyper-offload", "--hyper-offload-host-mb", "0")
    with pytest.raises(ValueError, match="host MiB"):
        feature.validate_args(args)
