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
"""Scheduler input model for activation residency planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
import math


class AccessKind(Enum):
    """How an op accesses a storage."""

    READ = auto()
    WRITE = auto()


@dataclass
class StorageAccess:
    """A storage access made by an op."""

    op_id: int
    storage_id: int
    kind: AccessKind


@dataclass
class TraceOp:
    """A scheduler-visible execution op."""

    name: str
    duration_ms: float = 0.0
    accesses: list[StorageAccess] = field(default_factory=list)


@dataclass
class ActivationTrace:
    """Scheduler input: ordered ops plus storage metadata."""

    ops: list[TraceOp] = field(default_factory=list)
    storage_sizes: dict[int, int] = field(default_factory=dict)
    retained_sids: set[int] = field(default_factory=set)
    memory_limit_bytes: int | None = None
    d2h_bandwidth_gbps: float = 16.0
    h2d_bandwidth_gbps: float = 16.0

    def validate(self) -> ActivationTrace:
        """Validate metadata and return a dense copy of legacy sparse traces."""
        def nonnegative_integer(value, name):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"trace {name} must be a nonnegative integer")

        if self.memory_limit_bytes is not None:
            nonnegative_integer(self.memory_limit_bytes, "budget")
        for sid, size in self.storage_sizes.items():
            nonnegative_integer(sid, "storage ID")
            nonnegative_integer(size, "storage size")
        positions = []
        previous = -1
        for index, op in enumerate(self.ops):
            if not math.isfinite(float(op.duration_ms)) or float(op.duration_ms) < 0:
                raise ValueError("trace duration must be finite and nonnegative")
            for access in op.accesses:
                nonnegative_integer(access.op_id, "access op_id")
                nonnegative_integer(access.storage_id, "access storage ID")
                if access.storage_id not in self.storage_sizes:
                    raise ValueError(f"trace storage size missing for {access.storage_id}")
                if not isinstance(access.kind, AccessKind):
                    raise ValueError("trace access kind must be READ or WRITE")
            identifiers = {access.op_id for access in op.accesses}
            if len(identifiers) > 1:
                raise ValueError("trace accesses in one op must share an op_id")
            position = next(iter(identifiers)) if identifiers else previous + 1
            if position < previous:
                raise ValueError("trace access op_ids must be ordered across ops")
            positions.append(position)
            previous = position
        if not self.retained_sids <= self.storage_sizes.keys():
            raise ValueError("trace retained storage IDs must have sizes")
        if positions == list(range(len(self.ops))):
            return self
        dense_ops = []
        for position, op in zip(positions, self.ops):
            dense_ops.extend(TraceOp("unrecorded") for _ in range(position - len(dense_ops)))
            if position < len(dense_ops):
                previous_op = dense_ops[position]
                dense_ops[position] = replace(
                    previous_op, accesses=previous_op.accesses + op.accesses,
                    duration_ms=float(previous_op.duration_ms) + float(op.duration_ms),
                )
            else:
                dense_ops.append(op)
        return replace(self, ops=dense_ops)
