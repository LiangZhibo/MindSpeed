"""Small feedback-directed neighborhood for a sequential transformer stack."""

from __future__ import annotations

from dataclasses import replace
import math

from .controller import JointPolicy, Trial


class JointNeighborhood:
    """Propose at most one joint move from completed measurements.

    The first move trades a bounded group of checkpointed layers for automatic
    storage offload and calibrated transfer windows. If it helps, continue in
    that direction; otherwise probe more recomputation and less transfer.
    No Cartesian product of layer choices and schedules is enumerated.
    The caller's initial policy must be a known feasible conservative setup.
    """

    def __init__(
        self,
        layer_count: int,
        *,
        layer_stride: int | None = None,
        window_scale: float = 1.0,
        minimum_gain: float = 0.005,
    ):
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 1:
            raise ValueError('layer_count must be positive')
        if layer_stride is not None and (
            isinstance(layer_stride, bool) or not isinstance(layer_stride, int) or layer_stride < 1
        ):
            raise ValueError('layer_stride must be positive')
        if not math.isfinite(window_scale) or window_scale < 0:
            raise ValueError('window_scale must be finite and nonnegative')
        if not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1:
            raise ValueError('minimum_gain must be in [0,1)')
        self.layer_count = layer_count
        self.stride = layer_stride or max(1, math.ceil(layer_count / 4))
        self.window_scale = window_scale
        self.minimum_gain = minimum_gain

    def __call__(self, trials: tuple[Trial, ...]) -> JointPolicy | None:
        initial = trials[0]
        if initial.policy.recompute_count > self.layer_count:
            raise ValueError('initial recomputation count exceeds registered layers')
        best = min(trials, key=lambda trial: trial.score)
        tried = {trial.policy for trial in trials}
        advanced = len(trials) == 1 or best.score < initial.score * (1 - self.minimum_gain)
        count = (
            max(0, best.policy.recompute_count - self.stride)
            if advanced
            else min(self.layer_count, initial.policy.recompute_count + self.stride)
        )
        candidate = replace(best.policy, recompute_count=count, masking=True, transfer_window_scale=self.window_scale)
        return candidate if candidate not in tried else None
