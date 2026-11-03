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
"""Warmup-phase executor.

During warmup, the executor records the execution trace while applying
online greedy eviction to keep device memory within budget.  Every
input tensor that was evicted in an earlier op is faulted back
synchronously (demand-paging) before dispatch.

Physical state transitions are delegated to
:class:`~offload.runtime.residency.ResidencyManager`.
"""

from __future__ import annotations

import logging
import math
from statistics import median
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from mindspeed.core.memory.hyper_offload.execution.base import BaseExecutor
from mindspeed.core.memory.hyper_offload.execution.schema import mutated_tensors
from mindspeed.core.memory.hyper_offload.execution.validation import input_bindings, op_identity, signature
from mindspeed.core.memory.hyper_offload.runtime.timer import DeviceTimer
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from mindspeed.core.memory.hyper_offload.ir.replay import OpGuide
from mindspeed.core.memory.hyper_offload.ir.trace import AccessKind, ActivationTrace, StorageAccess, TraceOp
from mindspeed.core.memory.hyper_offload.runtime.bandwidth import profile_transfer_bandwidth
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


def iter_tensors(value: Any) -> list[torch.Tensor]:
    """Return tensor leaves from an arbitrary pytree."""
    leaves, _ = tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


class WarmupExecutor(BaseExecutor):
    """Executor for the warmup phase.

    Records the execution trace while applying online greedy eviction
    to keep device memory within the configured budget.  Evicts the
    **oldest** activations first when the memory budget is exceeded
    (oldest-first within the same op, largest-sized entries are
    preferred as tie-breaker to minimise eviction count).
    """

    def __init__(
        self,
        residency_manager: ResidencyManager,
        memory_limit_bytes: int,
        *, transfer_probe_budget=None,
    ) -> None:
        super().__init__(residency_manager)
        self._memory_limit_bytes = memory_limit_bytes
        self._transfer_probe_budget = transfer_probe_budget
        self._transfer_size_counts = {}
        self._tracker = ActivationTracker()
        self._timer = DeviceTimer()
        self._guide: list[OpGuide] = []
        self._ops: list[TraceOp] = []
        #: sid -> the op index that first produced this activation.
        self._sid_produced_at_op: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Eviction policy
    # ------------------------------------------------------------------

    def _enforce_budget(self, protected_sids: set[int]) -> None:
        """Evict warmup activations until resident bytes fit the configured budget."""
        self.residency_manager.release_unreferenced(protected_sids)
        while self.residency_manager.resident_bytes > self._memory_limit_bytes:
            # Greedy: evict oldest first; within the same op, evict largest first.
            victim_sid: int | None = None
            victim_key: tuple[int, int] | None = None

            for sid, produced_at_op in self._sid_produced_at_op.items():
                if sid in protected_sids:
                    continue
                size = self.residency_manager.device_resident_size(sid)
                if size is None:
                    continue
                key = (produced_at_op, -size)
                if victim_key is None or key < victim_key:
                    victim_key = key
                    victim_sid = sid

            if victim_sid is None:
                raise RuntimeError(
                    "Warmup memory budget exceeded but no evictable activation found. "
                    f"resident_bytes={self.residency_manager.resident_bytes}, "
                    f"limit={self._memory_limit_bytes}, "
                    f"protected_sids={protected_sids}"
                )

            if self._transfer_probe_budget is not None:
                buffer = self.residency_manager._residency[victim_sid]
                if buffer.host_buffer is None and buffer.device_buffer is not None:
                    size = buffer.device_buffer.numel()
                    self._transfer_size_counts[size] = self._transfer_size_counts.get(size,0)+1
            self.residency_manager.copy_d2h(victim_sid)
            self.residency_manager.release_device(victim_sid)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func, args, kwargs) -> None:
        """Before op: enforce memory budget and fault inputs back to device."""
        super().on_op_begin(func, args, kwargs)
        # Warmup can evict at this boundary before its plan is known.
        set_node = getattr(self.residency_manager, "set_transfer_node", None)
        if set_node is not None:
            set_node(self.op_idx, "pre", op_identity(func))

        protected_sids: set[int] = set()
        for t in iter_tensors((args, kwargs)):
            if (sid := self._tracker.get_activation_sid(t)) is not None:
                protected_sids.add(sid)
        self._enforce_budget(protected_sids)

        self._input_signature = signature((args, kwargs))
        self._input_bindings = input_bindings(args, kwargs)

        self._timer.start()

    def on_op_end(self, result) -> Any:
        """After op: record trace, residency metadata, and return shadowed result."""
        op_duration_ms = self._timer.stop()

        func, args, kwargs = self._last_func, self._last_args, self._last_kwargs

        self._tracker.register_op_activations(iter_tensors((args, kwargs)), iter_tensors(result))

        op = TraceOp(name=func.__name__, duration_ms=op_duration_ms)

        # --- Detect mutated (write) input tensors via func._schema ---
        mutated_tensor_ids = {id(t) for t in mutated_tensors(func, args, kwargs)}
        if getattr(func, '_hyper_offload_invalidates_inputs', False):
            mutated_tensor_ids.update(id(t) for t in iter_tensors((args, kwargs)))

        seen: set[tuple[int, AccessKind]] = set()

        # --- Input accesses ---
        for t in iter_tensors((args, kwargs)):
            sid = self._tracker.get_activation_sid(t)
            if sid is None:
                continue
            is_mutated = id(t) in mutated_tensor_ids
            kind = AccessKind.WRITE if is_mutated else AccessKind.READ
            if (sid, kind) not in seen:
                seen.add((sid, kind))
                op.accesses.append(StorageAccess(self.op_idx, sid, kind))

        # --- Output accesses + bindings ---
        leaves, _ = tree_flatten(result)
        output_bindings: dict[int, int] = {}

        for leaf_index, t in enumerate(leaves):
            if not isinstance(t, torch.Tensor):
                continue
            sid = self._tracker.get_activation_sid(t)
            if sid is None:
                continue

            if (sid, AccessKind.WRITE) not in seen:
                seen.add((sid, AccessKind.WRITE))
                op.accesses.append(StorageAccess(self.op_idx, sid, AccessKind.WRITE))

            # Track the op that first produced this activation (used by
            # eviction policy: oldest-first).
            if sid not in self._sid_produced_at_op:
                self._sid_produced_at_op[sid] = self.op_idx

            output_bindings[leaf_index] = sid

        guide = OpGuide(
            name=op_identity(func),
            output_leaf_count=len(leaves),
            output_bindings=output_bindings,
            input_signature=self._input_signature,
            input_bindings=self._input_bindings,
            output_signature=signature((args, kwargs, result), scalar_values=False),
        )

        self._ops.append(op)
        self._guide.append(guide)

        return self.apply_shadows(result, output_bindings)

    def finish(self) -> tuple[ActivationTrace, list[OpGuide]]:
        """Finish warmup and return recorded trace and replay guide."""
        points = []
        if self._transfer_probe_budget is None:
            d2h, h2d = profile_transfer_bandwidth()
        else:
            details = {}
            if self._transfer_probe_budget()>=16*1024**2:
                d2h, h2d = profile_transfer_bandwidth(diagnostics=details)
            else:
                d2h, h2d = 1.0, 1.0
            def retain(info):
                outbound, inbound = info.get('d2h_samples_ms',()), info.get('h2d_samples_ms',())
                if (info.get('status')=='measured' and outbound and inbound
                        and all(math.isfinite(x) and x>0 for x in (*outbound,*inbound))):
                    points.append((info['bytes'],median(outbound),median(inbound)))
            retain(details)
            # Three dominant actual D2H payload sizes, ranked by transferred
            # bytes. The legacy probe remains a fourth anchor when valid.
            ranked = sorted(self._transfer_size_counts,key=lambda n:(-n*self._transfer_size_counts[n],n))
            attempted = 0
            for size in ranked:
                if size==16*1024**2 or size<=0 or size>self._transfer_probe_budget():
                    continue
                if attempted>=3:break
                attempted += 1
                info = {}
                profile_transfer_bandwidth(size_bytes=size,diagnostics=info)
                retain(info)
            points.sort()

        trace = ActivationTrace(
            ops=list(self._ops),
            storage_sizes=self._tracker.storage_sizes,
            retained_sids=set(self.retained_sids),
            memory_limit_bytes=self._memory_limit_bytes,
            d2h_bandwidth_gbps=d2h,
            h2d_bandwidth_gbps=h2d,
            transfer_cost_samples=tuple(points),
        )
        guide = self._guide

        self.reset()
        return trace, guide

    def reset(self) -> None:
        """Reset."""
        self._sid_produced_at_op.clear()
        self._transfer_size_counts.clear()
        self._ops = []
        self._guide = []
        self._tracker.clear_activations()
        self._timer.reset()
        super().reset()
