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
"""Peak-driven planner for activation residency actions.

This planner builds a simulated resident-memory profile from a captured
:class:`ActivationTrace`, then repeatedly selects storage gaps that cover the
current global peak until the configured memory budget is met. Selected gaps
are materialized as copy, release, and prefetch actions for replay.
"""

from __future__ import annotations

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import logging_utils
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import masking
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import modeling
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import schedule_builder
from mindspeed.core.memory.hyper_offload.planning.peak_strategy import selection
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    MaskingContext,
    ScoreWeights,
    SelectionState,
)


class PeakResidencyPlanner:
    """Plan residency by repeatedly reducing the current global peak."""

    def __init__(
        self,
        *,
        weights: ScoreWeights | None = None,
        enable_communication_masking: bool = False,
        **score_weights: float,
    ) -> None:
        self._weights = weights or self._parse_score_weights(score_weights)
        self._enable_communication_masking = enable_communication_masking

    @staticmethod
    def _parse_score_weights(score_weights: dict[str, float]) -> ScoreWeights:
        """Parse legacy score weight keyword arguments."""
        allowed_keys = {
            "score_weight_size": "size",
            "score_weight_cover_start": "cover_start",
            "score_weight_cover_end": "cover_end",
            "score_weight_neighbor_impact": "neighbor_impact",
        }
        unknown_keys = set(score_weights) - set(allowed_keys)
        if unknown_keys:
            raise TypeError(f"Unexpected score weight keyword(s): {sorted(unknown_keys)}")

        values = {
            field_name: score_weights.get(option_name, 1.0)
            for option_name, field_name in allowed_keys.items()
        }
        return ScoreWeights(**values)

    def build(self, trace: ActivationTrace) -> ResidencySchedule:
        """Build a residency schedule from a captured activation trace."""
        limit = trace.memory_limit_bytes if trace.memory_limit_bytes is not None else float("inf")
        if not trace.ops or limit == float("inf"):
            return ResidencySchedule()

        accesses_by_storage = self._group_accesses_by_storage(trace)
        profile = self._build_profile_and_candidates(trace, accesses_by_storage)
        unavoidable_peak = self._compute_unavoidable_peak(profile.resident_bytes, profile.candidates)
        if unavoidable_peak is not None and unavoidable_peak[1] > limit:
            self._log_unavoidable_peak_warning(unavoidable_peak, limit)

        selection_state = SelectionState(selected_candidate_keys=set(), selected_lifecycles=[])
        selected_candidates = self._select_peak_candidates(profile, selection_state, limit)

        selected_candidates.extend(
            self._select_boundary_candidates(
                profile.boundary_candidates,
                selection_state.selected_candidate_keys,
                profile.edge_bytes,
                limit,
            ),
        )

        if self._enable_communication_masking and selected_candidates:
            masking_context = MaskingContext(
                selected_candidates=selected_candidates,
                resident_bytes=profile.resident_bytes,
                limit=limit,
                releasable=profile.releasable,
                retained_sids=trace.retained_sids,
            )
            schedule, profile.resident_bytes = self._apply_communication_masking(masking_context)
        else:
            schedule = self._build_unmasked_schedule(selected_candidates, profile.releasable, trace.retained_sids)
        self._log_planning_summary(trace, schedule, profile.resident_bytes, limit, unavoidable_peak)
        return schedule

    _group_accesses_by_storage = staticmethod(modeling.group_accesses_by_storage)
    _build_profile_and_candidates = staticmethod(modeling.build_profile_and_candidates)
    _extend_last_use_for_alias_ops = staticmethod(modeling.extend_last_use_for_alias_ops)
    _is_alias_forward_op = staticmethod(modeling.is_alias_forward_op)
    _build_candidates_for_storage = staticmethod(modeling.build_candidates_for_storage)
    _version_stop = staticmethod(modeling.version_stop_at)
    _build_interior_candidates = staticmethod(modeling.build_interior_candidates)
    _candidate_for_boundary = staticmethod(modeling.candidate_for_boundary)
    _is_version_start_access = staticmethod(modeling.is_version_start_access)
    _has_same_op_read_write = staticmethod(modeling.has_same_op_read_write)
    _compute_safe_copy_starts = staticmethod(modeling.compute_safe_copy_starts)

    _find_global_peak = staticmethod(selection.find_global_peak)
    _compute_unavoidable_peak = staticmethod(selection.compute_unavoidable_peak)
    _mark_candidate_selected = staticmethod(selection.mark_candidate_selected)
    _effective_peak_index = staticmethod(selection.effective_peak_index)
    _has_covering_candidate = staticmethod(selection.has_covering_candidate)
    _select_best_candidate = staticmethod(selection.select_best_candidate)
    _is_better_candidate = staticmethod(selection.is_better_candidate)
    _is_candidate_eligible = staticmethod(selection.is_candidate_eligible)
    _compute_score_normalizers = staticmethod(selection.compute_score_normalizers)
    _select_boundary_candidates = staticmethod(selection.select_boundary_candidates)
    _score_boundary_candidate = staticmethod(selection.score_boundary_candidate)
    _compute_candidate_score = staticmethod(selection.compute_candidate_score)
    _compute_neighbor_impact = staticmethod(selection.compute_neighbor_impact)
    _apply_selection = staticmethod(selection.apply_selection)
    _apply_edge_selection = staticmethod(selection.apply_edge_selection)

    _apply_communication_masking = staticmethod(masking.apply_communication_masking)
    _maximize_release_positions = staticmethod(masking.maximize_release_positions)
    _find_latest_release_position = staticmethod(masking.find_latest_release_position)
    _find_earliest_prefetch_position = staticmethod(masking.find_earliest_prefetch_position)
    _can_add_bytes = staticmethod(masking.can_add_bytes)
    _apply_residency_delta = staticmethod(masking.apply_residency_delta)
    _maximize_prefetch_positions = staticmethod(masking.maximize_prefetch_positions)

    _build_unmasked_schedule = staticmethod(schedule_builder.build_unmasked_schedule)
    _build_masked_schedule = staticmethod(schedule_builder.build_masked_schedule)
    _build_schedule = staticmethod(schedule_builder.build_schedule)
    _order_selected_versions = staticmethod(schedule_builder.order_selected_versions)
    _add_prefetch_actions = staticmethod(schedule_builder.add_prefetch_actions)
    _collect_post_actions = staticmethod(schedule_builder.collect_post_actions)
    _add_deduplicated_post_actions = staticmethod(schedule_builder.add_deduplicated_post_actions)

    _log_unavoidable_peak_warning = staticmethod(logging_utils.log_unavoidable_peak_warning)
    _log_selection_exhaustion = staticmethod(logging_utils.log_selection_exhaustion)
    _log_planning_summary = staticmethod(logging_utils.log_planning_summary)

    def _select_peak_candidates(self, profile, state, limit: float):
        """Select gap candidates until the simulated peak fits the budget."""
        return selection.select_peak_candidates(
            profile,
            state,
            limit,
            self._weights,
            self._log_selection_exhaustion,
        )
