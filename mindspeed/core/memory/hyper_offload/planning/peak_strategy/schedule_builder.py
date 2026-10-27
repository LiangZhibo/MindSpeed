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
"""Schedule materialization helpers for the peak residency strategy."""

from __future__ import annotations

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import StorageAccess
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    PeakCandidate,
    PostAction,
    ScheduleBuildContext,
    SelectionKey,
)


def build_unmasked_schedule(
    selected_candidates: list[PeakCandidate],
    releasable: dict[int, StorageAccess],
    retained_sids: set[int],
) -> ResidencySchedule:
    """Materialize the default schedule without communication masking."""
    release_positions = {
        candidate.selection_key: candidate.release_start.op_id for candidate in selected_candidates
    }
    prefetch_positions = {
        candidate.selection_key: candidate.end.op_id for candidate in selected_candidates
    }
    return build_schedule(
        ScheduleBuildContext(
            selected_candidates=selected_candidates,
            release_positions=release_positions,
            prefetch_positions=prefetch_positions,
            releasable=releasable,
            retained_sids=retained_sids,
        ),
    )


def build_masked_schedule(context: ScheduleBuildContext) -> ResidencySchedule:
    """Materialize a schedule from optimized action positions."""
    return build_schedule(context)


def build_schedule(context: ScheduleBuildContext) -> ResidencySchedule:
    """Materialize a schedule from selected gaps and action positions."""
    schedule = ResidencySchedule()
    ordered_candidates = order_selected_versions(context.selected_candidates)
    add_prefetch_actions(schedule, ordered_candidates, context.prefetch_positions)
    post_actions = collect_post_actions(ordered_candidates, context)
    add_deduplicated_post_actions(schedule, post_actions)
    return schedule


def order_selected_versions(selected_candidates: list[PeakCandidate]) -> list[PeakCandidate]:
    """Return the first selected candidate for each tensor version in action order."""
    selected_by_version: dict[SelectionKey, PeakCandidate] = {}
    for candidate in selected_candidates:
        selected_by_version.setdefault(candidate.selection_key, candidate)
    return sorted(
        selected_by_version.values(),
        key=lambda item: (
            item.copy_start.op_id,
            item.release_start.op_id,
            item.end.op_id,
            item.storage_id,
        ),
    )


def add_prefetch_actions(
    schedule: ResidencySchedule,
    ordered_candidates: list[PeakCandidate],
    prefetch_positions: dict[SelectionKey, int],
) -> None:
    """Add COPY_H2D and RELEASE_HOST pre-actions."""
    for candidate in sorted(
        ordered_candidates,
        key=lambda item: (
            prefetch_positions[item.selection_key],
            item.end.op_id,
            item.storage_id,
        ),
    ):
        schedule.add_pre(
            prefetch_positions[candidate.selection_key],
            ResidencyActionType.COPY_H2D,
            candidate.storage_id,
        )
        schedule.add_pre(
            prefetch_positions[candidate.selection_key],
            ResidencyActionType.RELEASE_HOST,
            candidate.storage_id,
        )


def collect_post_actions(
    ordered_candidates: list[PeakCandidate],
    context: ScheduleBuildContext,
) -> list[PostAction]:
    """Collect ordered COPY_D2H and RELEASE_DEVICE post-actions."""
    post_actions: list[PostAction] = []
    order = 0
    for candidate in ordered_candidates:
        sid = candidate.storage_id
        post_actions.append((candidate.copy_start.op_id, 10, order, ResidencyActionType.COPY_D2H, sid))
        order += 1

        post_actions.append(
            (
                context.release_positions[candidate.selection_key],
                20,
                order,
                ResidencyActionType.RELEASE_DEVICE,
                sid,
            )
        )
        order += 1

    for sid, access in context.releasable.items():
        if sid in context.retained_sids:
            continue
        post_actions.append((access.op_id, 20, order, ResidencyActionType.RELEASE_DEVICE, sid))
        order += 1
    return post_actions


def add_deduplicated_post_actions(schedule: ResidencySchedule, post_actions: list[PostAction]) -> None:
    """Add post-actions while preserving first occurrence after sorting."""
    emitted: set[tuple[int, ResidencyActionType, int]] = set()
    for op_id, _, _, kind, sid in sorted(post_actions):
        action_key = (op_id, kind, sid)
        if action_key in emitted:
            continue
        schedule.add_post(op_id, kind, sid)
        emitted.add(action_key)
