# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CLI/default combinations and isolated legacy backend shims."""

from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from mindspeed.core.memory.hyper_offload import OffloadConfig
from ..test_feature import feature, parse


@pytest.mark.parametrize('field', ['max_resident_activation_mb', 'max_offload_activation_mb'])
@pytest.mark.parametrize('value', [True, 1.5, float('nan'), float('inf'), '12'])
def test_budget_requires_integer_mib(field, value):
    with pytest.raises((ValueError, TypeError), match='integer|budget|MiB'):
        OffloadConfig(**{field: value})


@pytest.mark.parametrize('field', ['cuda_graph_impl', 'npu_graph_mode'])
@pytest.mark.parametrize('value', [None, False, 'none'])
def test_disabled_graph_spellings_do_not_reject_eager(feature, field, value):
    options = parse(feature, '--hyper-offload')
    setattr(options, field, value)
    feature.validate_args(options)


@pytest.mark.parametrize('backend', ['native', 'cpu', 'cuda', 'npu', 'npu_fallback_stats'])
def test_legacy_shim_isolated_from_real_torch(backend):
    import mindspeed.core.memory.hyper_offload._patch_accelerator as module
    fake = ModuleType('torch')
    fake.device = lambda kind, index=None: (kind, index)
    native = object()
    if backend == 'native': fake.accelerator = native
    common = dict(current_device=lambda: 2, current_stream=Mock(), empty_cache=Mock(),
                  reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=17))
    fake.cuda = SimpleNamespace(is_available=lambda: backend == 'cuda', **common)
    if backend.startswith('npu'):
        fake.npu = SimpleNamespace(is_available=lambda: True, **common)
        if backend == 'npu_fallback_stats':
            del fake.npu.reset_peak_memory_stats
            del fake.npu.max_memory_allocated
    with patch.dict(sys.modules, {'torch': fake}):
        runpy.run_path(str(Path(module.__file__)))
    assert sys.modules['torch'] is torch
    if backend == 'native':
        assert fake.accelerator is native
        return
    assert fake.accelerator.is_available() == (backend != 'cpu')
    if backend == 'cpu':
        with pytest.raises(RuntimeError, match='No accelerator'):
            fake.accelerator.current_accelerator()
        with pytest.raises(RuntimeError, match='No accelerator'):
            fake.accelerator.current_stream()
        assert fake.accelerator.max_memory_allocated() == 0
    else:
        kind = 'npu' if backend.startswith('npu') else 'cuda'
        assert fake.accelerator.current_accelerator() == (kind, 2)
        assert fake.accelerator.current_stream() is common['current_stream'].return_value
        assert fake.accelerator.max_memory_allocated() == (0 if backend.endswith('stats') else 17)
    fake.accelerator.empty_cache()
    fake.accelerator.reset_peak_memory_stats()
