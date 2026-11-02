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
"""Communication masking helpers for selected peak candidates."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.schedule_builder import (
    build_masked_schedule,
)
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    MaskingContext,
    PeakCandidate,
    PositionSearchContext,
    ScheduleBuildContext,
    SelectionKey,
)


def apply_communication_masking(context: MaskingContext) -> tuple[ResidencySchedule, list[int]]:
    """Rebuild the schedule with optional release/prefetch overlap."""
    release_bounds, prefetch_bounds = transfer_window_bounds(context)
    release_positions, resident_bytes = maximize_release_positions(
        context.selected_candidates,
        context.resident_bytes,
        context.limit,
        release_bounds,
    )
    prefetch_positions = maximize_prefetch_positions(
        context.selected_candidates,
        resident_bytes,
        context.limit,
        release_positions,
        prefetch_bounds,
    )
    schedule = build_masked_schedule(
        ScheduleBuildContext(
            selected_candidates=context.selected_candidates,
            release_positions=release_positions,
            prefetch_positions=prefetch_positions,
            releasable=context.releasable,
            retained_sids=context.retained_sids,
        ),
    )
    return schedule, resident_bytes


def transfer_window_bounds(context: MaskingContext) -> tuple[dict, dict]:
    """Bound overlap windows using the measured compute timeline.

    These are scheduling hints, not completion guarantees: runtime events still
    protect every release and consumer. Memory feasibility is checked by the
    same resident-profile searches as the original planner. Preparing all
    bounds costs O(candidates * log(ops)), without another operator traversal.
    """
    prefix = context.time_prefix_ms
    scale = context.transfer_window_scale
    if scale is None:
        return {}, {}
    if prefix is None or len(prefix) != len(context.resident_bytes) + 1:
        raise ValueError("transfer windows require a complete compute timeline")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("transfer window scale must be finite and nonnegative")
    if any(not math.isfinite(bw) or bw <= 0 for bw in (context.d2h_bandwidth_gbps, context.h2d_bandwidth_gbps)):
        raise ValueError("transfer windows require positive finite bandwidths")
    release, prefetch = {}, {}
    for candidate in context.selected_candidates:
        # Trace bandwidths use GiB/s; the timer reports milliseconds.
        d2h_ms = candidate.size_bytes / (context.d2h_bandwidth_gbps * 2**30) * 1000 * scale
        h2d_ms = candidate.size_bytes / (context.h2d_bandwidth_gbps * 2**30) * 1000 * scale
        copy_after = candidate.copy_start.op_id + 1
        release_after = bisect_left(prefix, prefix[copy_after] + d2h_ms)
        release[candidate.selection_key] = max(candidate.release_start.op_id, release_after - 1)
        needed_before = candidate.end.op_id
        prefetch_before = bisect_right(prefix, prefix[needed_before] - h2d_ms) - 1
        if context.limit_prefetch_window:
            prefetch[candidate.selection_key] = max(0, min(needed_before, prefetch_before))
    return release, prefetch


def maximize_release_positions(
    selected_candidates: list[PeakCandidate],
    resident_bytes: list[int],
    limit: float,
    release_bounds: dict[SelectionKey, int] | None = None,
) -> tuple[dict[SelectionKey, int], list[int]]:
    """Push RELEASE_DEVICE actions rightward while keeping the budget safe."""
    release_positions = {candidate.selection_key: candidate.release_start.op_id for candidate in selected_candidates}
    prefetch_positions = {candidate.selection_key: candidate.end.op_id for candidate in selected_candidates}
    ordered_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            release_positions[candidate.selection_key],
            candidate.end.op_id,
            prefetch_positions[candidate.selection_key],
            candidate.storage_id,
        ),
    )

    next_release_position = len(resident_bytes)
    for candidate in reversed(ordered_candidates):
        key = candidate.selection_key
        current_release = release_positions[key]
        latest_release = find_latest_release_position(
            PositionSearchContext(
                resident_bytes=resident_bytes,
                size_bytes=candidate.size_bytes,
                origin_position=current_release,
                bound_position=min(
                    prefetch_positions[key] - 1,
                    next_release_position,
                    (release_bounds or {}).get(key, len(resident_bytes)),
                ),
                limit=limit,
            ),
        )
        apply_residency_delta(
            resident_bytes,
            candidate.size_bytes,
            current_release + 1,
            latest_release + 1,
        )
        release_positions[key] = latest_release
        next_release_position = latest_release

    return release_positions, resident_bytes


def find_latest_release_position(context: PositionSearchContext) -> int:
    """Binary-search the farthest safe release position."""
    if context.bound_position <= context.origin_position:
        return context.origin_position
    lower = context.origin_position
    upper = min(context.bound_position, len(context.resident_bytes) - 1)
    if upper <= lower:
        return context.origin_position

    left = lower
    right = upper
    best = context.origin_position
    while left <= right:
        mid = (left + right) // 2
        if can_add_bytes(context, context.origin_position + 1, mid + 1):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def find_earliest_prefetch_position(context: PositionSearchContext) -> int:
    """Binary-search the earliest safe prefetch position."""
    if context.bound_position >= context.origin_position:
        return context.origin_position
    lower = max(context.bound_position, 0)
    upper = context.origin_position
    if lower >= upper:
        return context.origin_position

    left = lower
    right = upper
    best = context.origin_position
    while left <= right:
        mid = (left + right) // 2
        if can_add_bytes(context, mid, context.origin_position):
            best = mid
            right = mid - 1
        else:
            left = mid + 1
    return best


def can_add_bytes(context: PositionSearchContext, start_op_id: int, end_op_id: int) -> bool:
    """Return whether adding bytes keeps a half-open interval within budget."""
    return all(
        context.resident_bytes[op_id] + context.size_bytes <= context.limit for op_id in range(start_op_id, end_op_id)
    )


def apply_residency_delta(
    resident_bytes: list[int],
    size_bytes: int,
    start_op_id: int,
    end_op_id: int,
) -> None:
    """Apply a size delta on a half-open interval."""
    for op_id in range(start_op_id, end_op_id):
        resident_bytes[op_id] += size_bytes


def maximize_prefetch_positions(
    selected_candidates: list[PeakCandidate],
    resident_bytes: list[int],
    limit: float,
    release_positions: dict[SelectionKey, int],
    prefetch_bounds: dict[SelectionKey, int] | None = None,
) -> dict[SelectionKey, int]:
    """Pull COPY_H2D actions leftward while preserving order and budget."""
    prefetch_positions = {candidate.selection_key: candidate.end.op_id for candidate in selected_candidates}
    ordered_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            candidate.end.op_id,
            candidate.release_start.op_id,
            candidate.storage_id,
        ),
    )

    previous_prefetch_position = -1
    for candidate in ordered_candidates:
        key = candidate.selection_key
        origin = prefetch_positions[key]
        lower_bound = max(release_positions[key] + 1, previous_prefetch_position, (prefetch_bounds or {}).get(key, 0))
        earliest_safe = find_earliest_prefetch_position(
            PositionSearchContext(
                resident_bytes=resident_bytes,
                size_bytes=candidate.size_bytes,
                origin_position=origin,
                bound_position=lower_bound,
                limit=limit,
            ),
        )
        apply_residency_delta(
            resident_bytes,
            candidate.size_bytes,
            earliest_safe,
            origin,
        )
        prefetch_positions[key] = earliest_safe
        previous_prefetch_position = earliest_safe

    return prefetch_positions
