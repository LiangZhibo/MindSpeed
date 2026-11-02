"""Small feedback-directed neighborhood for a sequential transformer stack."""

from __future__ import annotations

from dataclasses import replace
import math

from .controller import JointPolicy, Trial


class QwenRegionNeighborhood:
    """A bounded group search over named Qwen checkpoint regions.

    By default probe cheap normalization regions in the offloaded suffix with
    legacy masking, then calibrated release with memory-bounded prefetch.
    This separates their compute/transfer tradeoff
    from re-executing expensive projections. The optional ``mlp`` neighborhood
    first removes attention recomputation from the initial checkpoint prefix,
    preserving MLP recomputation and adding cheap normalization checkpoints.
    If measured performance improves, test the finer gate-only MLP group. If
    it regresses, retain the known layer policy and add gate checkpoints to
    its offloaded suffix. Each move jointly tests calibrated transfer masking.
    The groups keep search cost linear in layer count, not exponential in the
    number of registered operator regions; measured complete-step time chooses
    the fixed final policy.
    """

    def __init__(
        self, layer_count: int, initial_count: int, *, window_scale=1.0, probe='norm', resident_mlp_fraction=0.5,
        window_probe_multiplier=8.0, window_probe_limit_prefetch=True,
    ):
        if not 0 <= initial_count <= layer_count or layer_count < 1:
            raise ValueError('invalid layer count')
        self.layer_count = layer_count
        self.initial_count = initial_count
        self.window_scale = window_scale
        if probe not in (
            'norm',
            'mlp',
            'attention_resident',
            'suffix_resident',
            'mlp_resident',
            'compute_resident',
            'budget_resident',
            'stream_resident',
            'cache_resident',
            'pointwise_resident',
            'pointwise_window_resident',
            'gate_recompute_resident',
            'placement_resident',
            'window_resident',
            'release_resident',
            'partial_compute_resident',
            'gate_compute_resident',
            'phase_trim_resident',
            'phase_cache_resident',
        ):
            raise ValueError('invalid region probe')
        if not math.isfinite(resident_mlp_fraction) or not 0 < resident_mlp_fraction <= 1:
            raise ValueError('resident_mlp_fraction must be in (0,1]')
        self.resident_mlp_fraction = resident_mlp_fraction
        if not math.isfinite(window_probe_multiplier) or window_probe_multiplier < 0:
            raise ValueError('window_probe_multiplier must be finite and nonnegative')
        if not isinstance(window_probe_limit_prefetch, bool):
            raise ValueError('window_probe_limit_prefetch must be a boolean')
        self.window_probe_multiplier = window_probe_multiplier
        self.window_probe_limit_prefetch = window_probe_limit_prefetch
        self.probe = probe

    def __call__(self, trials: tuple[Trial, ...]) -> JointPolicy | None:
        initial = trials[0]
        if self.probe == 'compute_resident':
            improved = len(trials) > 1 and trials[1].score < initial.score * 0.995
            if len(trials) == 1 or improved:
                component = 'mlp' if len(trials) == 1 else 'mlp.gate'
                names = [f'{i}.{component}' for i in range(self.initial_count)]
                suffix = 0
            else:
                names = [str(i) for i in range(self.initial_count)]
                suffix = self.initial_count
            names += [
                f'{i}.{norm}'
                for i in range(suffix, self.layer_count)
                for norm in ('input_layernorm', 'post_attention_layernorm')
            ]
            candidate = replace(
                initial.policy,
                recompute_count=0,
                recompute_regions=tuple(names),
                resident_regions=tuple(f'{i}.self_attn' for i in range(suffix, self.layer_count)),
                masking=True,
                transfer_window_scale=self.window_scale,
                limit_prefetch_window=False,
            )
            return candidate if candidate not in {t.policy for t in trials} else None
        if self.probe in (
            'norm',
            'attention_resident',
            'suffix_resident',
            'mlp_resident',
            'budget_resident',
            'stream_resident',
            'cache_resident',
            'pointwise_resident',
            'pointwise_window_resident',
            'gate_recompute_resident',
            'placement_resident',
            'window_resident',
            'release_resident',
            'partial_compute_resident',
            'gate_compute_resident',
            'phase_trim_resident',
            'phase_cache_resident',
        ):
            names = [str(i) for i in range(self.initial_count)]
            names += [
                f'{i}.{norm}'
                for i in range(self.initial_count, self.layer_count)
                for norm in ('input_layernorm', 'post_attention_layernorm')
            ]
            candidate = replace(
                initial.policy,
                recompute_count=0,
                recompute_regions=tuple(names),
                masking=True,
                transfer_window_scale=None if len(trials) == 1 else self.window_scale,
                limit_prefetch_window=len(trials) == 1,
            )
            if self.probe in (
                'attention_resident',
                'suffix_resident',
                'mlp_resident',
                'budget_resident',
                'stream_resident',
                'cache_resident',
                'pointwise_resident',
                'pointwise_window_resident',
                'gate_recompute_resident',
                'placement_resident',
                'window_resident',
                'release_resident',
                'partial_compute_resident',
                'gate_compute_resident',
                'phase_trim_resident',
                'phase_cache_resident',
            ):
                candidate = replace(
                    candidate,
                    transfer_window_scale=self.window_scale,
                    limit_prefetch_window=False,
                    resident_regions=(
                        ()
                        if len(trials) == 1 and self.probe == 'attention_resident'
                        else tuple(f'{i}.self_attn' for i in range(self.initial_count, self.layer_count))
                    ),
                )
                if self.probe == 'suffix_resident' and len(trials) > 1 or self.probe == 'mlp_resident':
                    count = (
                        math.ceil((self.layer_count - self.initial_count) * self.resident_mlp_fraction)
                        if self.probe == 'mlp_resident' and len(trials) > 1
                        else 0
                    )
                    candidate = replace(
                        candidate,
                        resident_regions=candidate.resident_regions
                        + tuple(
                            f'{i}.mlp' if i >= self.layer_count - count else f'{i}.mlp.gate'
                            for i in range(self.initial_count, self.layer_count)
                        ),
                    )
                if self.probe in ('pointwise_resident', 'pointwise_window_resident', 'gate_recompute_resident') and len(trials) > 1:
                    component = 'mlp.gate' if self.probe == 'gate_recompute_resident' else 'mlp.pointwise'
                    candidate = replace(
                        candidate,
                        recompute_regions=candidate.recompute_regions
                        + tuple(f'{i}.{component}' for i in range(self.initial_count, self.layer_count)),
                    )
                if self.probe == 'pointwise_window_resident':
                    candidate = replace(
                        candidate, transfer_window_scale=self.window_probe_multiplier * self.window_scale,
                        limit_prefetch_window=self.window_probe_limit_prefetch,
                    )
                if self.probe == 'placement_resident' and len(trials) > 1:
                    boundary = self.layer_count - self.initial_count
                    candidate = replace(
                        candidate,
                        recompute_regions=tuple(str(i) for i in range(boundary, self.layer_count))
                        + tuple(f'{i}.{norm}' for i in range(boundary)
                                for norm in ('input_layernorm', 'post_attention_layernorm')),
                        resident_regions=tuple(f'{i}.self_attn' for i in range(boundary)),
                    )
                if self.probe == 'partial_compute_resident' and len(trials) > 1:
                    # Keep MLP recomputation while replacing only a bounded
                    # prefix-adjacent group of attention checkpoints with
                    # resident saves. The offloaded suffix MLPs stay unchanged.
                    boundary = max(0, self.initial_count - max(1, math.ceil(self.layer_count / 6)))
                    candidate = replace(
                        candidate,
                        recompute_regions=tuple(str(i) for i in range(boundary))
                        + tuple(f'{i}.mlp' for i in range(boundary, self.initial_count))
                        + tuple(f'{i}.{norm}' for i in range(boundary, self.layer_count)
                                for norm in ('input_layernorm', 'post_attention_layernorm')),
                        resident_regions=tuple(f'{i}.self_attn' for i in range(boundary, self.layer_count)),
                    )
                if self.probe == 'gate_compute_resident' and len(trials) > 1:
                    candidate = replace(
                        candidate,
                        recompute_regions=tuple(f'{i}.mlp.gate' for i in range(self.initial_count))
                        + tuple(f'{i}.{norm}' for i in range(self.layer_count)
                                for norm in ('input_layernorm', 'post_attention_layernorm')),
                        resident_regions=tuple(f'{i}.self_attn' for i in range(self.layer_count)),
                    )
                if self.probe == 'cache_resident' and len(trials) > 1:
                    candidate = replace(candidate, trim_before_replay=False)
                if self.probe in ('phase_trim_resident', 'phase_cache_resident'):
                    candidate = replace(
                        candidate, transfer_window_scale=self.window_probe_multiplier * self.window_scale,
                        limit_prefetch_window=self.window_probe_limit_prefetch,
                        trim_after_offload=self.probe == 'phase_cache_resident' or len(trials) > 1,
                    )
                    if self.probe == 'phase_cache_resident' and len(trials) > 1:
                        candidate = replace(candidate, trim_before_replay=False)
                if self.probe == 'release_resident' and len(trials) > 1:
                    candidate = replace(candidate, defer_h2d_release=True)
                if self.probe == 'window_resident' and len(trials) > 1:
                    # An intermediate lookahead can reduce transient residency
                    # while allowing more queueing slack than one copy time.
                    # It reuses the same trace and ordinary memory checks.
                    candidate = replace(
                        candidate, transfer_window_scale=self.window_probe_multiplier * self.window_scale,
                        limit_prefetch_window=self.window_probe_limit_prefetch,
                    )
                if self.probe == 'stream_resident' and len(trials) > 1:
                    candidate = replace(candidate, prefetch_on_copy_stream=True)
                if self.probe == 'budget_resident' and len(trials) > 1:
                    candidate = replace(candidate, resident_mb=initial.policy.resident_mb // 2)
            return candidate if candidate not in {t.policy for t in trials} else None
        improved = len(trials) > 1 and min(t.score for t in trials[1:]) < initial.score * 0.995
        if len(trials) == 1 or improved:
            component = 'mlp' if len(trials) == 1 else 'mlp.gate'
            names = [f'{i}.{component}' for i in range(self.initial_count)]
            norm_start = 0
        else:
            names = [str(i) for i in range(self.initial_count)]
            names += [f'{i}.mlp.gate' for i in range(self.initial_count, self.layer_count)]
            norm_start = self.initial_count
        names += [
            f'{i}.{norm}'
            for i in range(norm_start, self.layer_count)
            for norm in ('input_layernorm', 'post_attention_layernorm')
        ]
        candidate = replace(
            initial.policy,
            recompute_count=0,
            recompute_regions=tuple(names),
            masking=True,
            transfer_window_scale=self.window_scale,
        )
        return candidate if candidate not in {t.policy for t in trials} else None


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
