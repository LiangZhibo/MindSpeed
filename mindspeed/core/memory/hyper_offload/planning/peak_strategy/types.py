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
"""Shared data structures for the peak residency strategy."""

from __future__ import annotations

from dataclasses import dataclass

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencyActionType
from mindspeed.core.memory.hyper_offload.ir.trace import StorageAccess

SelectionKey = tuple[int, int, int, int]
PostAction = tuple[int, int, int, ResidencyActionType, int]


@dataclass(frozen=True)
class PeakCandidate:
    """A candidate offload gap between two adjacent accesses."""

    storage_id: int
    size_bytes: int
    copy_start: StorageAccess
    release_start: StorageAccess
    end: StorageAccess
    first_op_id: int
    last_op_id: int

    @property
    def start_interior(self) -> int:
        """First op that can benefit from the eviction."""
        return self.release_start.op_id + 1

    @property
    def end_interior(self) -> int:
        """Last op that can benefit from the eviction."""
        return self.end.op_id - 1

    @property
    def gap_width(self) -> int:
        """Width of the evictable gap in operator slots."""
        return max(0, self.end_interior - self.start_interior + 1)

    @property
    def peak_reduction_score(self) -> tuple[int, int, int]:
        """Score used to prefer candidates that reduce the peak more aggressively."""
        return (self.size_bytes, self.gap_width, self.release_start.op_id)

    @property
    def selection_key(self) -> SelectionKey:
        """Stable key for the tensor version produced by the latest write."""
        return (self.storage_id, self.copy_start.op_id, self.copy_start.kind.value, self.size_bytes)


@dataclass(frozen=True)
class CandidateBuildContext:
    """Inputs shared while building candidates for one storage."""

    storage_id: int
    size_bytes: int
    ordered_accesses: list[StorageAccess]
    safe_copy_starts: list[StorageAccess]
    first_op_id: int
    last_op_id: int


@dataclass
class ProfileAndCandidates:
    """Resident-memory model and eviction choices derived from a trace."""

    resident_bytes: list[int]
    edge_bytes: list[int]
    candidates: list[PeakCandidate]
    boundary_candidates: list[PeakCandidate]
    releasable: dict[int, StorageAccess]

    def __iter__(self):
        """Keep compatibility with the historical tuple return shape."""
        yield self.resident_bytes
        yield self.edge_bytes
        yield self.candidates
        yield self.boundary_candidates
        yield self.releasable


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for normalized peak candidate scoring."""

    size: float = 1.0
    cover_start: float = 1.0
    cover_end: float = 1.0
    neighbor_impact: float = 1.0


@dataclass(frozen=True)
class ScoreNormalizers:
    """Denominators for per-peak normalized scoring."""

    size: float
    cover_start: float
    cover_end: float


@dataclass
class SelectionState:
    """Mutable selection state for one planning run."""

    selected_candidate_keys: set[SelectionKey]
    selected_lifecycles: list[tuple[int, int]]


@dataclass(frozen=True)
class PeakPlateauContext:
    """Inputs used to choose a reducible slot on a peak plateau."""

    candidates: list[PeakCandidate]
    selected_candidate_keys: set[SelectionKey]
    resident_bytes: list[int]
    right_peak_idx: int
    peak_value: int


@dataclass(frozen=True)
class PeakSelectionContext:
    """Inputs needed to choose one candidate for a peak slot."""

    candidates: list[PeakCandidate]
    resident_bytes: list[int]
    peak_idx: int
    weights: ScoreWeights


@dataclass(frozen=True)
class CandidateScoreContext:
    """Inputs needed to score one candidate."""

    peak_idx: int
    normalizers: ScoreNormalizers
    selected_lifecycles: list[tuple[int, int]]
    weights: ScoreWeights


@dataclass(frozen=True)
class PositionSearchContext:
    """Inputs for binary-searching a safe action position."""

    resident_bytes: list[int]
    size_bytes: int
    origin_position: int
    bound_position: int
    limit: float


@dataclass(frozen=True)
class MaskingContext:
    """Inputs for optional communication masking."""

    selected_candidates: list[PeakCandidate]
    resident_bytes: list[int]
    limit: float
    releasable: dict[int, StorageAccess]
    retained_sids: set[int]


@dataclass(frozen=True)
class ScheduleBuildContext:
    """Inputs needed to materialize a residency schedule."""

    selected_candidates: list[PeakCandidate]
    release_positions: dict[SelectionKey, int]
    prefetch_positions: dict[SelectionKey, int]
    releasable: dict[int, StorageAccess]
    retained_sids: set[int]


# Backward-compatible private names used by historical callers/tests.
_PeakCandidate = PeakCandidate
_CandidateBuildContext = CandidateBuildContext
_ProfileAndCandidates = ProfileAndCandidates
_ScoreWeights = ScoreWeights
_ScoreNormalizers = ScoreNormalizers
_SelectionState = SelectionState
_PeakPlateauContext = PeakPlateauContext
_PeakSelectionContext = PeakSelectionContext
_CandidateScoreContext = CandidateScoreContext
_PositionSearchContext = PositionSearchContext
_MaskingContext = MaskingContext
_ScheduleBuildContext = ScheduleBuildContext
_SelectionKey = SelectionKey
_PostAction = PostAction
