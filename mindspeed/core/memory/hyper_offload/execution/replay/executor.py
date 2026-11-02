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
"""Replay-phase executor.

The executor executes residency schedule actions (D2H/H2D, device/host
release) and validates replayed outputs against the warmup trace.
Physical state transitions are delegated to
:class:`~offload.runtime.residency.ResidencyManager`.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_flatten

from mindspeed.core.memory.hyper_offload.execution.base import BaseExecutor
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.validation import input_bindings, op_identity, signature
from mindspeed.core.memory.hyper_offload.ir.replay import OpGuide
from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


class ReplayExecutor(BaseExecutor):
    """Executor for the replay phase.

    Executes residency schedule actions (D2H/H2D, device/host release)
    and validates replayed outputs against the warmup trace.
    """

    def __init__(
        self,
        residency_manager: ResidencyManager,
        schedule: ResidencySchedule,
        guide: list[OpGuide],
        *,
        after_offload: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(residency_manager)
        self._schedule = schedule
        self._guide = guide
        self._after_offload = after_offload
        self._after_offload_called = False
        self._last_offload_release_op = None
        if after_offload is not None:
            # Pick one fixed replay boundary after every transferred storage's
            # final prefetch and scheduled device release. If some transferred
            # storage remains live in the schedule, do not invent a boundary.
            prefetched = {}
            for index, actions in schedule.pre.items():
                for action in actions:
                    if action.kind == ResidencyActionType.COPY_H2D:
                        prefetched[action.storage_id] = max(index, prefetched.get(action.storage_id, -1))
            copied, released = {}, {}
            for index, actions in schedule.post.items():
                for action in actions:
                    sid = action.storage_id
                    if action.kind == ResidencyActionType.COPY_D2H:
                        copied[sid] = max(index, copied.get(sid, -1))
                    elif action.kind == ResidencyActionType.RELEASE_DEVICE and sid in prefetched:
                        if index >= prefetched[sid]:
                            released[sid] = max(index, released.get(sid, -1))
            if (copied and set(copied) == set(prefetched) == set(released)
                    and all(prefetched[sid] > index for sid, index in copied.items())):
                self._last_offload_release_op = max(released.values())
        # A complete guide permits native ownership for storages which neither
        # move nor require subclass identity at an autograd boundary. In Torch
        # 2.7, AccumulateGrad detaches a wrapper gradient twice but a native one
        # once. Preserve detach inputs as shadows so replay keeps the exact
        # operator stream and schedule indices recorded during warmup.
        # Keep legacy hand-built guides on their original contract.
        self._managed_sids = None if any(
            op.input_signature is None or op.output_signature is None for op in guide
        ) else {
            action.storage_id
            for phase in (schedule.pre, schedule.post)
            for actions in phase.values()
            for action in actions
            if action.kind in (ResidencyActionType.COPY_D2H, ResidencyActionType.COPY_H2D)
        }
        if self._managed_sids is not None:
            self._managed_sids.update(
                sid
                for op in guide if op.name == "aten.detach.default"
                for sid in op.input_bindings.values()
            )
        def managed(bindings):
            if self._managed_sids is None:
                return bindings
            return {index: sid for index, sid in bindings.items() if sid in self._managed_sids}
        self._replay_output_bindings = [managed(op.output_bindings) for op in guide]
        # Mirror the identity carried by each shadow, not every raw alias of
        # its storage. FSDP can deliberately create an untracked native view
        # while dispatch modes are disabled; warmup does not bind that view.
        # Weak tensor references neither retain activations nor run Tensor.eq.
        # Dead entries are harmless (identity is checked) and cleared per step.
        self._native_sids = {}

    def reset(self) -> None:
        super().reset()
        self._native_sids.clear()
        self._after_offload_called = False

    def _input_bindings(self, args, kwargs):
        if self._managed_sids is None:
            return input_bindings(args, kwargs)
        bindings = {}
        for index, value in enumerate(tree_flatten((args, kwargs))[0]):
            if isinstance(value, ShadowTensor):
                bindings[index] = value.storage_id
            elif isinstance(value, torch.Tensor):
                entry = self._native_sids.get(id(value))
                if entry is not None and entry[0]() is value:
                    bindings[index] = entry[1]
        return bindings

    def _register_native_bindings(self, leaves, bindings):
        if self._managed_sids is not None:
            for index, sid in bindings.items():
                if sid not in self._managed_sids:
                    value = leaves[index]
                    self._native_sids[id(value)] = (weakref.ref(value), sid)

    def execute_opaque_op(self, func_name, fn, args, kwargs, *, track_backward_inputs=False):
        nested = self.in_opaque_region
        result = super().execute_opaque_op(func_name, fn, args, kwargs,
                                         track_backward_inputs=track_backward_inputs)
        if not nested:
            # Autograd's custom-Function boundary may return a new Tensor
            # object. Register its final identity, as shadow fixup does.
            self._register_native_bindings(
                tree_flatten(result)[0], self._guide[self.op_idx].output_bindings
            )
        return result

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func, args, kwargs) -> None:
        """Before op: execute pre-actions."""
        super().on_op_begin(func, args, kwargs)
        if self.op_idx >= len(self._guide):
            raise RuntimeError(
                "replay op count exceeds warmup trace: "
                f"op_idx={self.op_idx}, "
                f"trace_ops={len(self._guide)}, "
                f"op={func.__name__}"
            )

        guide = self._guide[self.op_idx]
        if op_identity(func) != guide.name:
            raise RuntimeError(
                f"replay operator differs at op={self.op_idx}: "
                f"expected={guide.name}, actual={op_identity(func)}"
            )
        if guide.input_signature is not None and (
            signature((args, kwargs)) != guide.input_signature
            or self._input_bindings(args, kwargs) != guide.input_bindings
        ):
            raise RuntimeError(
                f"replay input metadata or bindings differ at op={self.op_idx}: "
                f"name={guide.name}, "
                f"metadata_equal={signature((args, kwargs)) == guide.input_signature}, "
                f"expected_bindings={guide.input_bindings}, "
                f"actual_bindings={self._input_bindings(args, kwargs)}"
            )

        for action in self._schedule.pre_actions(self.op_idx):
            if action.kind == ResidencyActionType.COPY_H2D:
                self.residency_manager.copy_h2d(action.storage_id)
            elif action.kind == ResidencyActionType.RELEASE_HOST:
                self.residency_manager.release_host(action.storage_id)
            else:
                raise RuntimeError(f"unsupported pre action {action} at op={self.op_idx}")

    def on_op_end(self, result) -> Any:
        """After op: validate output structure, shadow outputs, then execute post-actions.

        The order is critical: :meth:`apply_shadows` must run **before**
        post-actions so that output tensors are registered in the residency
        table (via :meth:`bind`) before the scheduler tries to copy or
        release them.  The previous order (post-actions → apply_shadows)
        caused ``COPY_D2H`` / ``RELEASE_DEVICE`` to silently skip output
        sids that had not yet been bound, leaving stale device buffers
        and missing host copies.
        """
        func = self._last_func
        op_guide = self._guide[self.op_idx]
        leaves, _ = tree_flatten(result)
        if len(leaves) != op_guide.output_leaf_count:
            raise RuntimeError(
                "replay output structure differs from warmup trace: "
                f"op_idx={self.op_idx}, "
                f"name={op_guide.name}, "
                f"expected_leaves={op_guide.output_leaf_count}, "
                f"actual_leaves={len(leaves)}, "
                f"op={func.__name__}"
            )

        # 1. Shadow outputs first — binds PhysicalBuffers so that
        #    post-actions can find and operate on them.
        if op_guide.output_signature is not None and signature(
            (self._last_args, self._last_kwargs, result), scalar_values=False
        ) != op_guide.output_signature:
            raise RuntimeError(f"replay output metadata or aliases differ at op={self.op_idx}")
        self._register_native_bindings(leaves, op_guide.output_bindings)
        result = self.apply_shadows(result, self._replay_output_bindings[self.op_idx])

        # 2. Execute post-actions (COPY_D2H, RELEASE_DEVICE, RELEASE_HOST).
        for action in self._schedule.post_actions(self.op_idx):
            sid = action.storage_id
            if action.kind == ResidencyActionType.COPY_D2H:
                self.residency_manager.copy_d2h(sid)
            elif action.kind == ResidencyActionType.RELEASE_DEVICE:
                if self._managed_sids is None or sid in self._managed_sids:
                    self.residency_manager.release_device(sid)
            elif action.kind == ResidencyActionType.RELEASE_HOST:
                self.residency_manager.release_host(sid)
            else:
                raise RuntimeError(f"unsupported post action {action} at op={self.op_idx}")

        if self.op_idx == self._last_offload_release_op and not self._after_offload_called:
            # Runtime housekeeping must never become a recorded model op.
            # Live tensor storage and allocator stream fences remain intact.
            self._after_offload_called = True
            with _disable_current_modes():
                self._after_offload()

        return result

    def finish(self) -> None:
        """Reject truncated steps as well as steps with additional operations."""
        if self.op_idx + 1 != len(self._guide):
            raise RuntimeError(
                f"replay op count differs from warmup trace: "
                f"expected={len(self._guide)}, actual={self.op_idx + 1}"
            )
