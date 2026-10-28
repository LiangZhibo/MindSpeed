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
"""Shadow tensor.

This module provides: :class:`ShadowTensor` — a :class:`torch.Tensor` subclass that
resolves a view of its :class:`PhysicalBuffer`'s storage on every dispatch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._pytree import tree_leaves, tree_map

from mindspeed.core.memory.hyper_offload.runtime.residency import PhysicalBuffer
from mindspeed.core.memory.hyper_offload.execution.schema import mutated_tensors


class ShadowTensor(torch.Tensor):
    """Tensor subclass that resolves a device view on demand.

    The shadow uses :meth:`PhysicalBuffer.device_storage` to obtain the
    raw device storage on every :meth:`resolve` call and creates a
    fresh tensor view from it.  **No cached reference is kept**, so the
    shadow never prevents the underlying storage from being freed.

    Because :class:`ShadowTensor` is created via
    :meth:`torch.Tensor._make_wrapper_subclass`, it carries all tensor
    metadata ("dtype", "size", "stride", "storage_offset",
    "device") directly.
    """

    @staticmethod
    def __new__(
        cls,
        elem: torch.Tensor,
        buffer: PhysicalBuffer,
        storage_id: int,
    ) -> ShadowTensor:
        """Create a new instance using *elem*'s metadata."""
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        torch._C._set_conj(result, elem.is_conj())
        torch._C._set_neg(result, elem.is_neg())
        return result

    def __init__(  # pylint: disable=unused-argument
        self,
        elem: torch.Tensor,
        buffer: PhysicalBuffer,
        storage_id: int,
    ) -> None:
        self._buffer = buffer
        self._storage_id = storage_id
        self._metadata_error = None
        buffer.shadows.add(self)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def storage_id(self) -> int:
        """Storage ID of the underlying physical block."""
        return self._storage_id

    # ------------------------------------------------------------------
    # Resolution (on every call)
    # ------------------------------------------------------------------

    def resolve(self) -> torch.Tensor:
        """Return a device-resident view of the physical storage.

        1. Calls :meth:`PhysicalBuffer.device_storage` to obtain the
           raw device storage (demand-paging from host if needed).
        2. Builds a fresh tensor view from the shadow's cached metadata.
        3. Returns the view — no long-lived reference is kept.
        """
        if self._metadata_error is not None:
            raise NotImplementedError(self._metadata_error)
        storage = self._buffer.device_storage()
        result = torch.empty(0, dtype=self.dtype, device=self.device)
        result.set_(storage, self.storage_offset(), self.size(), self.stride())
        torch._C._set_conj(result, self.is_conj())
        torch._C._set_neg(result, self.is_neg())
        return result

    # ------------------------------------------------------------------
    # PyTorch dispatch
    # ------------------------------------------------------------------

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        """Dispatch a torch operation."""
        kwargs = kwargs or {}
        resolved: dict[int, torch.Tensor] = {}

        def unwrap(value: Any) -> Any:
            if isinstance(value, ShadowTensor):
                if id(value) not in resolved:
                    resolved[id(value)] = value.resolve()
                return resolved[id(value)]
            return value

        with torch._C._DisableTorchDispatch():  # pylint: disable=protected-access
            writes = mutated_tensors(func, args, kwargs)
            shadow_writes = [tensor for tensor in writes if isinstance(tensor, ShadowTensor)]
            if shadow_writes and func._schema.name in ("aten::set_", "aten::resize_", "aten::resize_as_"):
                raise NotImplementedError(
                    f"{func}: storage rebinding/resizing of offloaded tensors is unsupported"
                )
            for tensor in shadow_writes:
                tensor._buffer.prepare_for_write()
            result = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            metadata_changed = any(
                tensor.size() != resolved[id(tensor)].size()
                or tensor.stride() != resolved[id(tensor)].stride()
                or tensor.storage_offset() != resolved[id(tensor)].storage_offset()
                for tensor in shadow_writes
            )
            for tensor in shadow_writes:
                storage = resolved[id(tensor)].untyped_storage()
                if storage.nbytes() != tensor._buffer.device_buffer.numel():
                    bytes_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    bytes_view.set_(storage, 0, (storage.nbytes(),), (1,))
                    tensor._buffer.device_buffer = bytes_view
            inplace_view = torch.Tag.inplace_view in getattr(func, "tags", ())
            if metadata_changed and not inplace_view:
                message = (
                    f"{func}: implicit output storage resizing is unsupported; "
                    "provide correctly sized out tensors and discard these outputs"
                )
                for tensor in shadow_writes:
                    tensor._metadata_error = message
                raise NotImplementedError(message)
            if shadow_writes and inplace_view:
                # Mirror PyTorch's wrapper-subclass metadata fixup on the meta
                # dispatch key; replaying the data kernel would mutate twice.
                previous = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(previous)
            # Inside a session the executor owns output wrapping and SID lookup.
            # Outside it, preserve aliases so a later view mutation cannot bypass
            # snapshot invalidation or escape lifecycle tracking.
            from mindspeed.core.memory.hyper_offload.api.session import OffloadSession
            if OffloadSession.get_active() is not None:
                return result
            inputs = [v for v in tree_leaves((args, kwargs)) if isinstance(v, ShadowTensor)]
            aliases = {
                v._buffer.device_buffer.untyped_storage()._cdata: v
                for v in inputs if v._buffer.device_buffer is not None
            }

            def wrap_alias(value):
                if isinstance(value, torch.Tensor) and not isinstance(value, ShadowTensor):
                    try:
                        source = aliases.get(value.untyped_storage()._cdata)
                    except (RuntimeError, NotImplementedError):
                        return value
                    if source is not None:
                        return ShadowTensor(value, source._buffer, source.storage_id)
                return value

            return tree_map(wrap_alias, result)
