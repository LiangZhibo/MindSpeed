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
"""Pinned host memory pool with bucket-based management.

Provides a thread-safe pool of host page-locked CPU memory buffers for
efficient host-device data transfers during activation offloading.
"""

import logging
import threading
from functools import wraps

import torch
from torch.utils._python_dispatch import _disable_current_modes

logger = logging.getLogger(__name__)


def _outside_dispatch(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Pool bookkeeping allocates CPU tensors while holding its lock. Those
        # allocations must not reenter the activation dispatcher and pool.
        with _disable_current_modes():
            return fn(*args, **kwargs)

    return wrapper

_BUCKET_SIZES = [2**i for i in range(10, 22)]
_LARGE_BUCKET_GRANULARITY = 2**21


def _align_to_bucket(size: int) -> int:
    """Find the smallest bucket size >= size."""
    for bucket in _BUCKET_SIZES:
        if bucket >= size:
            return bucket
    # Large activations are common in long-context models. Power-of-two
    # buckets can waste almost half of pinned Host memory for each buffer.
    return ((size + _LARGE_BUCKET_GRANULARITY - 1) //
            _LARGE_BUCKET_GRANULARITY) * _LARGE_BUCKET_GRANULARITY


def _bucket_for(size: int) -> int:
    """Return the bucket size a buffer belongs to."""
    return _align_to_bucket(size)


class PinnedMemoryPool:
    """Thread-safe Host buffer pool with deferred recycling via device events.

    If capacity is exhausted, a pending buffer may be waited for and reused.
    When no reusable buffer exists, acquire raises instead of silently
    exceeding the configured Host capacity. An optional pageable fallback
    keeps offload usable when the accelerator runtime cannot pin more pages.
    """

    def __init__(self, max_host_bytes: int, allow_pageable_fallback: bool = False) -> None:
        self._pool: dict[int, list[torch.Tensor]] = {}
        self._pending: dict[int, list[tuple[torch.Tensor, torch.Event | torch.Stream]]] = {}
        self._lock = threading.Lock()
        self._total_allocated = 0
        self._max_host_bytes = max_host_bytes
        self._owned_storages = {}
        self._leased = set()
        self._pageable_storages = set()
        self._pageable_allocated = 0
        self._pinning_exhausted = False
        self._allow_pageable_fallback = allow_pageable_fallback

    @property
    def pageable_allocated(self) -> int:
        """Bytes allocated without pinned Host pages after pinning fails."""
        return self._pageable_allocated

    def _lease(self, tensor, size):
        """Called under the lock when ownership passes to a borrower."""
        self._leased.add(tensor.untyped_storage()._cdata)
        return tensor[:size]

    @property
    def total_allocated(self) -> int:
        """Total host bytes currently held by the allocator."""
        return self._total_allocated

    @property
    def max_host_bytes(self) -> int:
        """Hard limit on host memory in bytes."""
        return self._max_host_bytes

    def _reclaim_locked(self, bucket: int) -> None:
        """Move completed tensors from pending to available pool."""
        if bucket not in self._pending:
            return

        still_pending = []
        for tensor, event in self._pending[bucket]:
            if event.query():
                self._pool.setdefault(bucket, []).append(tensor)
            else:
                still_pending.append((tensor, event))
        self._pending[bucket] = still_pending

    @_outside_dispatch
    def acquire(self, size: int) -> torch.Tensor:
        """Obtain a buffer of at least *size* bytes from the pool."""
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("host allocation size must be an integer")
        if size < 0:
            raise ValueError("host allocation size must be nonnegative")
        bucket = _bucket_for(size)
        with self._lock:
            candidates = sorted({bucket, *(b for b in self._pool if b >= bucket),
                                 *(b for b in self._pending if b >= bucket)})
            for bucket_size in candidates:
                self._reclaim_locked(bucket_size)
                entries = self._pool.get(bucket_size)
                if entries:
                    return self._lease(entries.pop(), size)

            aligned = _align_to_bucket(size)
            if self._total_allocated + aligned <= self._max_host_bytes:
                pageable = self._pinning_exhausted
                if not pageable:
                    try:
                        allocation = torch.empty(aligned, dtype=torch.uint8, pin_memory=True)
                    except torch.OutOfMemoryError:
                        if not self._allow_pageable_fallback:
                            raise
                        self._pinning_exhausted = True
                        pageable = True
                        logger.warning(
                            "Pinned Host allocation failed; using synchronous pageable "
                            "Host buffers for later HyperOffload transfers"
                        )
                if pageable:
                    allocation = torch.empty(aligned, dtype=torch.uint8)
                storage = allocation.untyped_storage()
                self._owned_storages[storage._cdata] = storage
                if pageable:
                    self._pageable_storages.add(storage._cdata)
                    self._pageable_allocated += aligned
                self._total_allocated += aligned
                logger.debug(
                    "PinnedMemoryPool: allocate %d bytes (total=%d, limit=%d)",
                    aligned,
                    self._total_allocated,
                    self._max_host_bytes,
                )
                return self._lease(allocation, size)

            for bucket_size in candidates:
                if self._pending.get(bucket_size):
                    tensor, event = self._pending[bucket_size][0]
                    # Leave the entry owned by the pool if waiting fails.
                    event.synchronize()
                    self._pending[bucket_size].pop(0)
                    return self._lease(tensor, size)

            raise RuntimeError(
                f"PinnedMemoryPool exhausted: total_allocated={self._total_allocated}, "
                f"max_host_bytes={self._max_host_bytes}, requested={size}"
            )

    @_outside_dispatch
    def release(self, tensor: torch.Tensor, event: torch.Event | torch.Stream | None = None) -> None:
        """Return an owned buffer, retaining any transfer completion fence."""

        storage = tensor.untyped_storage()
        full_tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
        full_tensor.set_(storage, 0, (storage.size(),), (1,))

        bucket = _bucket_for(full_tensor.numel())
        with self._lock:
            identity = storage._cdata
            if identity not in self._owned_storages:
                raise ValueError("release() storage is not owned by this Host pool")
            if not tensor.is_pinned() and identity not in self._pageable_storages:
                raise ValueError("release() expects a pinned buffer or owned pageable fallback")
            if identity not in self._leased:
                raise ValueError("release() storage is not leased or was already released")
            self._leased.remove(identity)
            if event is not None:
                self._pending.setdefault(bucket, []).append((full_tensor, event))
            else:
                self._pool.setdefault(bucket, []).append(full_tensor)
