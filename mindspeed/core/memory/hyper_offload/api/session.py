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

from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from mindspeed.core.memory.hyper_offload.api.config import OffloadConfig
from mindspeed.core.memory.hyper_offload.execution.replay import ReplayExecutor
from mindspeed.core.memory.hyper_offload.execution.warmup import WarmupExecutor
from mindspeed.core.memory.hyper_offload.planning.greedy import GreedyResidencyPlanner
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


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
            allow_pageable_fallback=True,
        )
        self.executor: WarmupExecutor | ReplayExecutor = WarmupExecutor(
            residency_manager=self.residency_manager,
            memory_limit_bytes=config.max_resident_activation_mb * 1024**2,
        )

        self._dispatch_context = None
        self._session_token = None
        self._cleanup_failed = False

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
