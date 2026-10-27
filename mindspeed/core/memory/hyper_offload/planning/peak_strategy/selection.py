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
"""Peak candidate selection helpers."""

from __future__ import annotations

from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    CandidateScoreContext,
    PeakCandidate,
    PeakPlateauContext,
    PeakSelectionContext,
    ProfileAndCandidates,
    ScoreNormalizers,
    SelectionKey,
    SelectionState,
)


def find_global_peak(resident_bytes: list[int]) -> tuple[int, int] | None:
    """Return ``(op_id, resident_bytes)`` for the current global peak."""
    if not resident_bytes:
        return None

    peak_idx = 0
    peak_val = resident_bytes[0]
    for idx in range(1, len(resident_bytes)):
        if resident_bytes[idx] >= peak_val:
            peak_idx = idx
            peak_val = resident_bytes[idx]
    return peak_idx, peak_val


def compute_unavoidable_peak(
    resident_bytes: list[int],
    candidates: list[PeakCandidate],
) -> tuple[int, int, int] | None:
    """Return the highest resident floor that candidate selection cannot remove."""
    if not resident_bytes:
        return None

    reducible_bytes = [0] * len(resident_bytes)
    covered_storage_ids = [set() for _ in resident_bytes]
    for candidate in candidates:
        start = max(0, candidate.start_interior)
        end = min(len(resident_bytes) - 1, candidate.end_interior)
        for op_id in range(start, end + 1):
            if candidate.storage_id in covered_storage_ids[op_id]:
                continue
            covered_storage_ids[op_id].add(candidate.storage_id)
            reducible_bytes[op_id] += candidate.size_bytes

    peak_op = 0
    peak_unavoidable = 0
    peak_reducible = 0
    for op_id, resident in enumerate(resident_bytes):
        reducible = min(resident, reducible_bytes[op_id])
        unavoidable = resident - reducible
        if unavoidable >= peak_unavoidable:
            peak_op = op_id
            peak_unavoidable = unavoidable
            peak_reducible = reducible
    return peak_op, peak_unavoidable, peak_reducible


def select_peak_candidates(
    profile: ProfileAndCandidates,
    state: SelectionState,
    limit: float,
    weights,
    log_selection_exhaustion,
) -> list[PeakCandidate]:
    """Select gap candidates until the simulated peak fits the budget."""
    selected_candidates: list[PeakCandidate] = []
    while True:
        peak = find_global_peak(profile.resident_bytes)
        if peak is None or peak[1] <= limit:
            return selected_candidates

        peak_idx = effective_peak_index(
            PeakPlateauContext(
                candidates=profile.candidates,
                selected_candidate_keys=state.selected_candidate_keys,
                resident_bytes=profile.resident_bytes,
                right_peak_idx=peak[0],
                peak_value=peak[1],
            ),
        )
        best = select_best_candidate(
            PeakSelectionContext(profile.candidates, profile.resident_bytes, peak_idx, weights),
            state,
        )
        if best is None:
            log_selection_exhaustion(peak, limit, len(selected_candidates))
            return selected_candidates

        mark_candidate_selected(state, selected_candidates, best)
        apply_selection(profile.resident_bytes, best)
        apply_edge_selection(profile.edge_bytes, best)


def mark_candidate_selected(
    state: SelectionState,
    selected_candidates: list[PeakCandidate],
    candidate: PeakCandidate,
) -> None:
    """Record a selected candidate in all selection indexes."""
    state.selected_candidate_keys.add(candidate.selection_key)
    state.selected_lifecycles.append((candidate.first_op_id, candidate.last_op_id))
    selected_candidates.append(candidate)


def effective_peak_index(context: PeakPlateauContext) -> int:
    """Prefer a coverable slot inside a rightmost peak plateau."""
    left = context.right_peak_idx
    while left > 0 and context.resident_bytes[left - 1] == context.peak_value:
        left -= 1

    effective = context.right_peak_idx
    while effective > left and not has_covering_candidate(
        context.candidates,
        context.selected_candidate_keys,
        effective,
    ):
        effective -= 1
    return effective


def has_covering_candidate(
    candidates: list[PeakCandidate],
    selected_candidate_keys: set[SelectionKey],
    peak_idx: int,
) -> bool:
    """Return whether any unselected candidate can reduce ``peak_idx``."""
    return any(
        candidate.selection_key not in selected_candidate_keys
        and candidate.first_op_id < peak_idx
        and candidate.start_interior <= peak_idx <= candidate.end_interior
        for candidate in candidates
    )


def select_best_candidate(
    context: PeakSelectionContext,
    state: SelectionState,
) -> PeakCandidate | None:
    """Pick the highest-scoring candidate that covers the current peak."""
    eligible = [
        candidate
        for candidate in context.candidates
        if is_candidate_eligible(
            candidate,
            state.selected_candidate_keys,
            context.resident_bytes,
            context.peak_idx,
        )
    ]
    if not eligible:
        return None

    score_context = CandidateScoreContext(
        peak_idx=context.peak_idx,
        normalizers=compute_score_normalizers(eligible, context.peak_idx),
        selected_lifecycles=state.selected_lifecycles,
        weights=context.weights,
    )
    best: PeakCandidate | None = None
    best_score = float("-inf")
    for candidate in eligible:
        score = compute_candidate_score(candidate, score_context)
        if is_better_candidate(candidate, score, best, best_score):
            best = candidate
            best_score = score
    return best


def is_better_candidate(
    candidate: PeakCandidate,
    score: float,
    best: PeakCandidate | None,
    best_score: float,
) -> bool:
    """Return whether ``candidate`` should replace the current best."""
    if best is None:
        return True
    if score != best_score:
        return score > best_score
    return candidate.peak_reduction_score > best.peak_reduction_score


def is_candidate_eligible(
    candidate: PeakCandidate,
    selected_candidate_keys: set[SelectionKey],
    resident_bytes: list[int],
    peak_idx: int,
) -> bool:
    """Return whether a candidate can reduce the current peak slot."""
    if candidate.selection_key in selected_candidate_keys:
        return False
    if candidate.first_op_id >= peak_idx:
        return False
    if not candidate.start_interior <= peak_idx <= candidate.end_interior:
        return False
    return peak_idx < len(resident_bytes) and resident_bytes[peak_idx] > 0


def compute_score_normalizers(candidates: list[PeakCandidate], peak_idx: int) -> ScoreNormalizers:
    """Compute max denominators for normalized candidate scoring."""
    max_size = max((candidate.size_bytes for candidate in candidates), default=1)
    max_cover_start = max((peak_idx - candidate.first_op_id for candidate in candidates), default=1)
    max_cover_end = max((candidate.end.op_id - peak_idx for candidate in candidates), default=1)
    return ScoreNormalizers(
        size=float(max(1, max_size)),
        cover_start=float(max(1, max_cover_start)),
        cover_end=float(max(1, max_cover_end)),
    )


def select_boundary_candidates(
    boundary_candidates: list[PeakCandidate],
    selected_candidate_keys: set[SelectionKey],
    edge_bytes: list[int],
    limit: float,
) -> list[PeakCandidate]:
    """Select adjacent-op gaps that only free memory between operator slots."""
    if limit == float("inf") or not edge_bytes:
        return []

    selected: list[PeakCandidate] = []
    unavailable_keys = set(selected_candidate_keys)
    while True:
        best = None
        best_score = (0, 0, 0, 0)
        for candidate in boundary_candidates:
            if candidate.selection_key in unavailable_keys:
                continue
            score = score_boundary_candidate(candidate, edge_bytes, limit)
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break

        selected.append(best)
        unavailable_keys.add(best.selection_key)
        apply_edge_selection(edge_bytes, best)
    return selected


def score_boundary_candidate(
    candidate: PeakCandidate,
    edge_bytes: list[int],
    limit: float,
) -> tuple[int, int, int, int]:
    """Score a boundary candidate by its simulated edge-memory reduction."""
    start = candidate.release_start.op_id
    end = candidate.end.op_id
    if end - start != 1 or start < 0 or start >= len(edge_bytes):
        return (0, 0, 0, 0)

    before = edge_bytes[start]
    after = max(0, before - candidate.size_bytes)
    peak_reduction = max(0, before - max(after, limit))
    over_budget_reduction = max(0, before - limit) - max(0, after - limit)
    if peak_reduction <= 0 and over_budget_reduction <= 0:
        return (0, 0, 0, 0)
    return (
        peak_reduction,
        over_budget_reduction,
        candidate.size_bytes,
        -candidate.release_start.op_id,
    )


def compute_candidate_score(
    candidate: PeakCandidate,
    context: CandidateScoreContext,
) -> float:
    """Compute a normalized peak selection score."""
    norm_size = candidate.size_bytes / context.normalizers.size
    norm_cover_start = (context.peak_idx - candidate.first_op_id) / context.normalizers.cover_start
    norm_cover_end = (candidate.end.op_id - context.peak_idx) / context.normalizers.cover_end
    norm_neighbor = compute_neighbor_impact(candidate, context.selected_lifecycles)
    return (
        context.weights.size * norm_size
        + context.weights.cover_start * norm_cover_start
        + context.weights.cover_end * norm_cover_end
        - context.weights.neighbor_impact * norm_neighbor
    )


def compute_neighbor_impact(
    candidate: PeakCandidate,
    selected_lifecycles: list[tuple[int, int]],
) -> float:
    """Return normalized overlap with already selected offload lifecycles."""
    if candidate.last_op_id <= candidate.first_op_id or not selected_lifecycles:
        return 0.0

    overlaps: list[tuple[int, int]] = []
    for start, end in selected_lifecycles:
        overlap_start = max(candidate.first_op_id, start)
        overlap_end = min(candidate.last_op_id, end)
        if overlap_start < overlap_end:
            overlaps.append((overlap_start, overlap_end))
    if not overlaps:
        return 0.0

    overlaps.sort()
    merged = 0
    current_start, current_end = overlaps[0]
    for start, end in overlaps[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged += current_end - current_start
        current_start, current_end = start, end
    merged += current_end - current_start

    return merged / max(1, candidate.last_op_id - candidate.first_op_id)


def apply_selection(resident_bytes: list[int], candidate: PeakCandidate) -> None:
    """Update the simulated profile for one selected candidate."""
    for op_id in range(candidate.start_interior, candidate.end_interior + 1):
        resident_bytes[op_id] = max(0, resident_bytes[op_id] - candidate.size_bytes)


def apply_edge_selection(edge_bytes: list[int], candidate: PeakCandidate) -> None:
    """Update adjacent-op edge memory for one selected candidate."""
    for edge_id in range(candidate.release_start.op_id, candidate.end.op_id):
        if 0 <= edge_id < len(edge_bytes):
            edge_bytes[edge_id] = max(0, edge_bytes[edge_id] - candidate.size_bytes)
