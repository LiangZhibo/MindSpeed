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
"""Peak-driven activation residency strategy."""

from mindspeed.core.memory.hyper_offload.planning.peak_strategy.planner import PeakResidencyPlanner
from mindspeed.core.memory.hyper_offload.planning.peak_strategy.types import (
    _CandidateBuildContext,
    _CandidateScoreContext,
    _MaskingContext,
    _PeakCandidate,
    _PeakPlateauContext,
    _PeakSelectionContext,
    _PositionSearchContext,
    _PostAction,
    _ProfileAndCandidates,
    _ScheduleBuildContext,
    _ScoreNormalizers,
    _ScoreWeights,
    _SelectionKey,
    _SelectionState,
)

__all__ = [
    "PeakResidencyPlanner",
    "_CandidateBuildContext",
    "_CandidateScoreContext",
    "_MaskingContext",
    "_PeakCandidate",
    "_PeakPlateauContext",
    "_PeakSelectionContext",
    "_PositionSearchContext",
    "_PostAction",
    "_ProfileAndCandidates",
    "_ScheduleBuildContext",
    "_ScoreNormalizers",
    "_ScoreWeights",
    "_SelectionKey",
    "_SelectionState",
]
