"""Select a real accelerator for HyperOffload integration tests."""

import importlib.util
import os

import pytest
import torch


if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401

    if torch.npu.is_available():
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def pytest_collection_modifyitems(items):
    available = torch.cuda.is_available() or (
        hasattr(torch, "npu") and torch.npu.is_available()
    )
    if not available:
        marker = pytest.mark.skip(reason="HyperOffload integration requires CUDA or Ascend NPU")
        for item in items:
            item.add_marker(marker)
