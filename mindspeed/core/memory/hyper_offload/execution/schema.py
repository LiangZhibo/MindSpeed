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
"""Bind mutable schema arguments without losing Tensor-list or keyword structure."""

from torch.utils._pytree import tree_leaves
import torch


def mutated_tensors(func, args, kwargs):
    """Return every tensor contained in an argument annotated as writable."""
    schema = getattr(func, "_schema", None)
    if schema is None or not schema.is_mutable:
        return []
    kwargs = kwargs or {}
    result = []
    for index, argument in enumerate(schema.arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        value = args[index] if index < len(args) else kwargs.get(argument.name)
        result.extend(t for t in tree_leaves(value) if isinstance(t, torch.Tensor))
    return result
