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
"""Activation offload session built on TorchDispatchMode."""

from __future__ import annotations

import contextvars
import logging
import sys
import threading
import time

import torch

from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from mindspeed.core.memory.hyper_offload.api.config import OffloadConfig
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.planning.greedy import GreedyResidencyPlanner
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)

_CACHE_RECLAIM_RESERVED_NUMERATOR = 935
_CACHE_RECLAIM_RESERVED_DENOMINATOR = 1000
_CACHE_RECLAIM_MIN_INACTIVE_BYTES = 512 * 1024**2
_ALLOCATION_OPS = {
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.empty_strided.default,
    torch.ops.aten.new_empty.default,
    torch.ops.aten.new_empty_strided.default,
    torch.ops.aten._to_copy.default,
}


def _empty_accelerator_cache(*, retain_physical: bool = False) -> bool | None:
    """Release inactive cache through the initialized accelerator backend."""
    for name in ("npu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is not None and backend.is_initialized():
            virtual_trim = getattr(backend, "empty_virt_addr_cache", None)
            if retain_physical and name == "npu" and callable(virtual_trim):
                try:
                    virtual_trim()
                except RuntimeError as error:
                    if "Unsupported config for empty_virt_addr_cache" not in str(error):
                        raise
                else:
                    return True
            backend.empty_cache()
            return False
    return None


def _inside_fsdp_internal() -> bool:
    """Keep parameter gather and reshard operations out of activation tracking."""
    frame = sys._getframe(1)  # pylint: disable=protected-access
    while frame is not None:
        if "/torch/distributed/fsdp/_fully_shard/" in frame.f_code.co_filename:
            return True
        frame = frame.f_back
    return False


# Context variable tracking the active OffloadSession (set on __enter__).
_active_session: contextvars.ContextVar[OffloadSession | None] = contextvars.ContextVar("_active_session", default=None)


class ActivationDispatchMode(TorchDispatchMode):
    """Dispatch mode that delegates to a :class:`BaseExecutor`.

    The executor (warmup or replay) is selected by
    :class:`OffloadSession` and never swapped while this mode
    is active.
    """

    def __init__(self, session: OffloadSession):
        super().__init__()
        self.session = session

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        """Dispatch a torch operation."""
        self.session._reclaim_before_allocation(func)
        if _inside_fsdp_internal():
            with _disable_current_modes():
                return func(*args, **(kwargs or {}))
        return self.session.executor.dispatch(func, args, kwargs)


class OffloadSession:
    """Manage warmup and replay for activation residency control.

    Holds lifecycle state and a single active executor (:class:`WarmupExecutor`
    or :class:`ReplayExecutor`). The executor is switched during the
    warmup → replay transition in :meth:`__exit__`.
    """

    def __init__(self, config: OffloadConfig):
        self.mode: str = "warmup"

        self.planner = config.planner or GreedyResidencyPlanner()
        self.residency_manager = ResidencyManager(
            max_host_bytes=config.max_offload_activation_mb * 1024**2,
            codec=config.codec,
            allow_pageable_fallback=True,
        )
        self.executor: WarmupExecutor | ReplayExecutor = WarmupExecutor(
            residency_manager=self.residency_manager,
            memory_limit_bytes=config.max_resident_activation_mb * 1024**2,
        )

        self._dispatch_context = None
        self._session_token = None
        self._cleanup_failed = False
        self.last_trim_seconds = 0.0
        self.trim_before_replay = config.trim_before_replay
        self._virtual_cache_supported = None
        self._cache_reclaim_threshold_bytes = None
        self._cache_reclaim_lock = threading.Lock()
        self.record_cache_reclaim = config.record_cache_reclaim
        # check_seconds includes reclaim_seconds; never add the two totals.
        # These counters are independent of last_trim_seconds at step entry.
        self.cache_reclaim_stats = dict(checks=0, calls=0, check_seconds=0.0, reclaim_seconds=0.0)

    def _reclaim_before_allocation(self, func):
        """Unmap inactive NPU cache before allocator pressure forces retries."""
        if func not in _ALLOCATION_OPS and str(func) != "fsdp.all_gather_copy_in.default":
            return
        backend = getattr(torch, "npu", None)
        if backend is None or not backend.is_initialized() or self._virtual_cache_supported is False:
            return
        recording = self.record_cache_reclaim
        started = time.perf_counter() if recording else None
        try:
            with self._cache_reclaim_lock:
                if self._cache_reclaim_threshold_bytes is None:
                    total_bytes = backend.get_device_properties(backend.current_device()).total_memory
                    self._cache_reclaim_threshold_bytes = (
                        total_bytes * _CACHE_RECLAIM_RESERVED_NUMERATOR
                        // _CACHE_RECLAIM_RESERVED_DENOMINATOR
                    )
                stats = backend.memory_stats_as_nested_dict()
                reserved = stats["reserved_bytes"]["all"]["current"]
                allocated = stats["allocated_bytes"]["all"]["current"]
                if (reserved < self._cache_reclaim_threshold_bytes
                        or reserved - allocated < _CACHE_RECLAIM_MIN_INACTIVE_BYTES):
                    return
                reclaim_started = time.perf_counter() if recording else None
                try:
                    self._virtual_cache_supported = _empty_accelerator_cache(retain_physical=True)
                finally:
                    if recording:
                        self.cache_reclaim_stats['calls'] += 1
                        self.cache_reclaim_stats['reclaim_seconds'] += time.perf_counter() - reclaim_started
        finally:
            if recording:
                self.cache_reclaim_stats['checks'] += 1
                self.cache_reclaim_stats['check_seconds'] += time.perf_counter() - started

    # ------------------------------------------------------------------
    # Active session lookup (used by skip_offload decorator)
    # ------------------------------------------------------------------

    @staticmethod
    def get_active() -> OffloadSession | None:
        """Return the currently active OffloadSession, or ``None``."""
        return _active_session.get()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        """Enter the runtime context."""
        if _active_session.get() is not None or self._dispatch_context is not None:
            raise RuntimeError("nested OffloadSession contexts are unsupported")
        if self._cleanup_failed:
            raise RuntimeError("cannot reuse OffloadSession after a cleanup failure")
        self.last_trim_seconds = 0.0
        if self.mode == "replay" and self.trim_before_replay:
            trim_started = time.perf_counter()
            if self._virtual_cache_supported is not False:
                supported = _empty_accelerator_cache(retain_physical=True)
                if supported is not None:
                    self._virtual_cache_supported = supported
            else:
                _empty_accelerator_cache()
            self.last_trim_seconds = time.perf_counter() - trim_started
        self.executor.reset()

        dispatch_context = ActivationDispatchMode(self)
        self._session_token = _active_session.set(self)
        try:
            dispatch_context.__enter__()
        except Exception:
            _active_session.reset(self._session_token)
            self._session_token = None
            raise
        self._dispatch_context = dispatch_context
        self.executor._session_active = True
        return self

    def _abort(self):
        """Preserve the original failure, attempting every cleanup operation."""
        for cleanup in (self.residency_manager.sync_all_transfers, self.executor.reset):
            try:
                cleanup()
            except Exception:
                self._cleanup_failed = True
                logger.exception("HyperOffload cleanup failed; discard this session")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context."""
        if self._dispatch_context is None:
            raise RuntimeError("OffloadSession is not active")
        self.executor._session_active = False
        try:
            self._dispatch_context.__exit__(exc_type, exc_val, exc_tb)
        except Exception:
            self._abort()
            raise
        finally:
            self._dispatch_context = None
            if self._session_token is not None:
                _active_session.reset(self._session_token)
                self._session_token = None

        if exc_type is not None:
            self._abort()
            return False

        try:
            self.residency_manager.wait_for_transfers()
            if self.mode == "warmup":
                trace, guide = self.executor.finish()
                schedule = self.planner.build(trace)
                self.executor = ReplayExecutor(self.residency_manager, schedule, guide)
                self.mode = "replay"
            else:
                self.executor.finish()
        except Exception:
            self._abort()
            raise

        return False
