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
"""Reference-free signatures for validating reuse of a recorded residency plan."""

import torch
from torch.utils._pytree import tree_flatten

from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor


def op_identity(func):
    """Include the namespace and overload for native operators."""
    return str(func) if hasattr(func, "_schema") else func.__name__


def signature(value, scalar_values=True):
    """Record structure, tensor metadata and relative storage aliases, never data."""
    leaves, spec = tree_flatten(value)
    aliases = {}
    metadata = []
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            metadata.append((type(leaf), repr(leaf) if scalar_values else None))
            continue
        physical = leaf._buffer.device_buffer if isinstance(leaf, ShadowTensor) else leaf
        if physical is None:
            key = ("shadow", id(leaf._buffer))
            storage_bytes = leaf._buffer.host_buffer.numel() if leaf._buffer.host_buffer is not None else None
        else:
            try:
                storage = physical.untyped_storage()
                key = ("storage", storage._cdata)
                storage_bytes = storage.nbytes()
            except (RuntimeError, NotImplementedError):
                key = ("tensor", id(physical))
                storage_bytes = None
        alias = aliases.setdefault(key, len(aliases))
        strided = leaf.layout == torch.strided
        metadata.append((
            "tensor", tuple(leaf.shape), leaf.dtype, leaf.device, leaf.layout,
            tuple(leaf.stride()) if strided else None,
            leaf.storage_offset() if strided else None, storage_bytes, alias,
            leaf.is_conj(), leaf.is_neg(),
        ))
    return spec, tuple(metadata)


def input_bindings(args, kwargs):
    """Identify logical activation inputs, including Tensor-list and keyword leaves."""
    return {
        index: tensor.storage_id
        for index, tensor in enumerate(tree_flatten((args, kwargs))[0])
        if isinstance(tensor, ShadowTensor)
    }
