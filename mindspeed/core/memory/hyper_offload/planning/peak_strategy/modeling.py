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
"""Trace modeling helpers for the peak residency strategy."""

from __future__ import annotations

from mindspeed.core.memory.hyper_offload.ir.trace import (
    AccessKind,
    ActivationTrace,
    StorageAccess,
)
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    CandidateBuildContext,
    PeakCandidate,
    ProfileAndCandidates,
)


def group_accesses_by_storage(trace: ActivationTrace) -> dict[int, list[StorageAccess]]:
    """Return ordered accesses grouped by storage id."""
    accesses_by_storage: dict[int, list[StorageAccess]] = {}
    for op in trace.ops:
        for access in op.accesses:
            accesses_by_storage.setdefault(access.storage_id, []).append(access)
    return accesses_by_storage


def build_profile_and_candidates(
    trace: ActivationTrace,
    accesses_by_storage: dict[int, list[StorageAccess]],
) -> ProfileAndCandidates:
    """Build the resident-memory profile and all evictable candidates."""
    resident_bytes = [0] * len(trace.ops)
    edge_bytes = [0] * max(0, len(resident_bytes) - 1)
    candidates: list[PeakCandidate] = []
    boundary_candidates: list[PeakCandidate] = []
    releasable: dict[int, StorageAccess] = {}

    for sid, accesses in accesses_by_storage.items():
        size_bytes = trace.storage_sizes.get(sid)
        if size_bytes is None or size_bytes <= 0 or not accesses:
            continue

        ordered = sorted(accesses, key=lambda access: access.op_id)
        first_op_id = ordered[0].op_id
        last_op_id = extend_last_use_for_alias_ops(trace, sid, ordered[-1].op_id)
        if sid in trace.retained_sids:
            last_op_id = len(trace.ops) - 1
        if first_op_id > last_op_id:
            continue

        _add_lifetime_bytes(resident_bytes, edge_bytes, first_op_id, last_op_id, size_bytes)
        if sid in trace.retained_sids:
            continue

        releasable[sid] = ordered[-1]
        context = CandidateBuildContext(
            storage_id=sid,
            size_bytes=size_bytes,
            ordered_accesses=ordered,
            safe_copy_starts=compute_safe_copy_starts(ordered),
            first_op_id=first_op_id,
            last_op_id=last_op_id,
        )
        storage_candidates, storage_boundary_candidates = build_candidates_for_storage(context)
        candidates.extend(storage_candidates)
        boundary_candidates.extend(storage_boundary_candidates)

    candidates.sort(key=lambda candidate: candidate.peak_reduction_score, reverse=True)
    return ProfileAndCandidates(
        resident_bytes=resident_bytes,
        edge_bytes=edge_bytes,
        candidates=candidates,
        boundary_candidates=boundary_candidates,
        releasable=releasable,
    )


def _add_lifetime_bytes(
    resident_bytes: list[int],
    edge_bytes: list[int],
    first_op_id: int,
    last_op_id: int,
    size_bytes: int,
) -> None:
    """Add one storage lifetime to op and edge resident-memory models."""
    for op_id in range(first_op_id, last_op_id + 1):
        if 0 <= op_id < len(resident_bytes):
            resident_bytes[op_id] += size_bytes
    for edge_id in range(first_op_id, last_op_id):
        if 0 <= edge_id < len(edge_bytes):
            edge_bytes[edge_id] += size_bytes


def extend_last_use_for_alias_ops(trace: ActivationTrace, sid: int, last_op_id: int) -> int:
    """Extend the lifetime through alias ops that forward the same storage."""
    extended = last_op_id
    for idx in range(last_op_id + 1, len(trace.ops)):
        op = trace.ops[idx]
        if is_alias_forward_op(op, sid):
            extended = idx
    return extended


def is_alias_forward_op(op, sid: int) -> bool:
    """Infer view-like storage forwarding from the trace-local access pattern."""
    accesses = [access for access in op.accesses if access.storage_id == sid]
    if not accesses:
        return False
    op_storage_ids = {access.storage_id for access in op.accesses}
    access_kinds = {access.kind for access in accesses}
    return (
        len(op_storage_ids) == 1
        and AccessKind.READ in access_kinds
        and AccessKind.WRITE in access_kinds
    )


def build_candidates_for_storage(
    context: CandidateBuildContext,
) -> tuple[list[PeakCandidate], list[PeakCandidate]]:
    """Enumerate evictable gaps for each write-produced tensor version."""
    result: list[PeakCandidate] = []
    boundary_result: list[PeakCandidate] = []
    version_starts = [
        index
        for index, access in enumerate(context.ordered_accesses)
        if is_version_start_access(context.ordered_accesses, index, access)
    ]
    if not version_starts:
        version_starts = [0]

    for position, version_start in enumerate(version_starts):
        version_stop = version_stop_at(version_starts, position, len(context.ordered_accesses))
        version_candidates = build_interior_candidates(context, version_start, version_stop)

        result.extend(version_candidates)
        if version_candidates or has_same_op_read_write(context.ordered_accesses[version_start:version_stop]):
            continue

        boundary_result.extend(candidate_for_boundary(context, version_start, version_stop))
    return result, boundary_result


def version_stop_at(version_starts: list[int], position: int, access_count: int) -> int:
    """Return the exclusive access index for one tensor version."""
    if position + 1 < len(version_starts):
        return version_starts[position + 1]
    return access_count


def build_interior_candidates(
    context: CandidateBuildContext,
    version_start: int,
    version_stop: int,
) -> list[PeakCandidate]:
    """Build candidates that free at least one operator slot."""
    return [
        candidate
        for candidate in candidate_for_boundary(context, version_start, version_stop)
        if candidate.start_interior <= candidate.end_interior
    ]


def candidate_for_boundary(
    context: CandidateBuildContext,
    version_start: int,
    version_stop: int,
) -> list[PeakCandidate]:
    """Build candidate objects for every adjacent access boundary."""
    candidates: list[PeakCandidate] = []
    for release_index in range(version_start, version_stop - 1):
        candidates.append(
            PeakCandidate(
                storage_id=context.storage_id,
                size_bytes=context.size_bytes,
                copy_start=context.safe_copy_starts[release_index],
                release_start=context.ordered_accesses[release_index],
                end=context.ordered_accesses[release_index + 1],
                first_op_id=context.first_op_id,
                last_op_id=context.last_op_id,
            ),
        )
    return candidates


def is_version_start_access(
    accesses: list[StorageAccess],
    index: int,
    access: StorageAccess,
) -> bool:
    """Return whether a write starts a new logical tensor version."""
    if access.kind != AccessKind.WRITE:
        return False
    return not any(other.kind == AccessKind.READ and other.op_id == access.op_id for other in accesses[: index + 1])


def has_same_op_read_write(accesses: list[StorageAccess]) -> bool:
    """Return whether accesses contain a same-op read/write alias pattern."""
    kinds_by_op: dict[int, set[AccessKind]] = {}
    for access in accesses:
        kinds_by_op.setdefault(access.op_id, set()).add(access.kind)
    return any(AccessKind.READ in kinds and AccessKind.WRITE in kinds for kinds in kinds_by_op.values())


def compute_safe_copy_starts(accesses: list[StorageAccess]) -> list[StorageAccess]:
    """Precompute the tensor production point for every release boundary."""
    safe_copy_starts: list[StorageAccess] = []
    latest_write = accesses[0]
    for access in accesses:
        if access.kind == AccessKind.WRITE:
            latest_write = access
        safe_copy_starts.append(latest_write)
    return safe_copy_starts
