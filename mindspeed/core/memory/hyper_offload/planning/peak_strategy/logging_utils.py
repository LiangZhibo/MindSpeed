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
"""Logging helpers for peak residency planning."""

from __future__ import annotations

import logging

from mindspeed.core.memory.hyper_offload.ir.schedule import ResidencySchedule
from mindspeed.core.memory.hyper_offload.ir.trace import ActivationTrace

logger = logging.getLogger("mindspeed.core.memory.hyper_offload.planning.peak")


def log_unavoidable_peak_warning(unavoidable_peak: tuple[int, int, int], limit: float) -> None:
    """Warn when the configured budget is below the offloadable floor."""
    logger.warning(
        "PeakResidencyPlanner replay budget is below the offloadable floor: "
        "unavoidable_peak=%d bytes at op=%d, budget=%d bytes, reducible_bytes_at_op=%d. "
        "The peak strategy only emits replay actions for trace gaps between two accesses; storage that has "
        "no candidate covering an op slot, such as retained storage, single-access storage, or storage used "
        "by that slot, remains resident in the replay memory model.",
        unavoidable_peak[1],
        unavoidable_peak[0],
        limit,
        unavoidable_peak[2],
    )


def log_selection_exhaustion(peak: tuple[int, int], limit: float, selected_count: int) -> None:
    """Warn when no remaining candidate can reduce an over-budget peak."""
    logger.warning(
        "PeakResidencyPlanner could not reduce peak further: peak=%d bytes at op=%d "
        "still exceeds budget=%d bytes; selected_candidates=%d; "
        "no remaining candidate covers the peak plateau",
        peak[1],
        peak[0],
        limit,
        selected_count,
    )


def log_planning_summary(
    trace: ActivationTrace,
    schedule: ResidencySchedule,
    resident_bytes: list[int],
    limit: float,
    unavoidable_peak: tuple[int, int, int] | None = None,
) -> None:
    """Log a short planning summary and the final peak location."""
    peak = max(resident_bytes, default=0)
    peak_op = 0
    for idx, value in enumerate(resident_bytes):
        if value >= peak:
            peak_op = idx
    logger.info(
        "PeakResidencyPlanner peak after selection: %.2f MiB / budget %.2f MiB",
        peak / 1024**2,
        limit / 1024**2,
    )
    logger.info(
        "PeakResidencyPlanner final peak position: op=%d, resident=%.2f MiB, selected_actions=%d pre / %d post",
        peak_op,
        peak / 1024**2,
        sum(len(v) for v in schedule.pre.values()),
        sum(len(v) for v in schedule.post.values()),
    )
    if unavoidable_peak is not None:
        logger.info(
            "PeakResidencyPlanner unavoidable floor: op=%d, resident=%.2f MiB, reducible=%.2f MiB",
            unavoidable_peak[0],
            unavoidable_peak[1] / 1024**2,
            unavoidable_peak[2] / 1024**2,
        )
    logger.debug(
        "PeakResidencyPlanner schedule: %d pre-actions, %d post-actions across %d ops",
        sum(len(v) for v in schedule.pre.values()),
        sum(len(v) for v in schedule.post.values()),
        len(trace.ops),
    )
