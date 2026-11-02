"""Checkpoint region switching at complete training-step boundaries."""

from __future__ import annotations

from collections.abc import MutableSequence
from functools import wraps

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint

from mindspeed.core.memory.hyper_offload.api.opaque import skip_offload
from mindspeed.core.memory.hyper_offload.api.session import OffloadSession


class _ResidentRegion(ActivationWrapper):
    """Keep a region's internal autograd saves native, without recomputation.

    The overall measured device peak must qualify this extra residency. The
    ordinary storage planner still handles the region's visible boundaries.
    """

    def __init__(self, module):
        super().__init__(module)
        self._call = skip_offload(module)

    def forward(self, *args, **kwargs):
        return self._call(*args, **kwargs)


class NamedRecomputeRegions:
    """Explicit, non-overlapping module and operator-group checkpoint regions.

    Register after distributed wrapping and before the first training step.
    Module paths and callable names share one dotted namespace. Callables must
    describe the actual scope of their operations: a ``mlp.gate`` group cannot
    also be registered independently through an overlapping alias. Selection
    changes only between completed steps; no Parameter is replaced or copied.
    """

    def __init__(self, root):
        self.root = root
        self._modules = {}
        self._callables = {}
        self._wrappers = {}
        self._conflicts = set()
        self.selected = frozenset()
        self.resident = frozenset()

    @property
    def names(self):
        return tuple(sorted(self._modules.keys() | self._callables.keys()))

    def _check_registration(self, name):
        if self.selected or self.resident or OffloadSession.get_active() is not None:
            raise RuntimeError('register regions before selecting a policy or entering an offload step')
        if not isinstance(name, str) or not name or any(not p for p in name.split('.')):
            raise ValueError('region name must be a nonempty dotted path')
        if name in self._modules or name in self._callables:
            raise ValueError(f'duplicate region: {name}')

    def register_module(self, name):
        self._check_registration(name)
        parts = name.split('.')
        parent = self.root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])
        self._modules[name] = (parent, parts[-1], original)

    def register_callable(self, name, function, *, offload_checkpoint_inputs=False):
        """Return a call site with an optional non-reentrant checkpoint."""
        self._check_registration(name)
        if not isinstance(offload_checkpoint_inputs, bool):
            raise TypeError('offload_checkpoint_inputs must be a boolean')
        body = skip_offload(function, track_backward_inputs=True) if offload_checkpoint_inputs else function

        @wraps(function)
        def recompute(*args, **kwargs):
            return checkpoint(body, *args, use_reentrant=False, **kwargs)

        checkpointed = recompute if offload_checkpoint_inputs else skip_offload(recompute)
        resident = skip_offload(function)
        self._callables[name] = function

        @wraps(function)
        def call(*args, **kwargs):
            target = checkpointed if name in self.selected else resident if name in self.resident else function
            return target(*args, **kwargs)

        return call

    def register_conflict(self, left, right):
        """Declare semantic overlap that is not represented by dotted parents."""
        if self.selected or self.resident or OffloadSession.get_active() is not None:
            raise RuntimeError('register conflicts before selecting a policy or entering an offload step')
        if left == right or left not in self.names or right not in self.names:
            raise ValueError('conflicts require two distinct registered regions')
        self._conflicts.add(frozenset((left, right)))

    def apply(self, names: tuple[str, ...]):
        self._select(names, ())

    def apply_policy(self, policy):
        if policy.recompute_count:
            raise ValueError('named adapter requires recompute_regions, not a layer count')
        self._select(policy.recompute_regions or (), policy.resident_regions)

    def _select(self, names, resident_names):
        if OffloadSession.get_active() is not None:
            raise RuntimeError('cannot switch checkpoint regions inside an active offload step')
        if (
            not isinstance(names, tuple)
            or not isinstance(resident_names, tuple)
            or len(set(names + resident_names)) != len(names) + len(resident_names)
        ):
            raise ValueError('select unique named regions as a tuple')
        selected = frozenset(names)
        resident = frozenset(resident_names)
        chosen = selected | resident
        unknown = chosen.difference(self.names)
        if unknown:
            raise ValueError(f'unregistered recompute regions: {sorted(unknown)}')
        for conflict in self._conflicts:
            if conflict <= chosen:
                raise ValueError(f'semantically overlapping regions: {sorted(conflict)}')
        for name in chosen:
            parts = name.split('.')
            if any('.'.join(parts[:i]) in chosen for i in range(1, len(parts))):
                raise ValueError('parent and child recompute regions overlap')
        before = {name: 'recompute' for name in self.selected} | {name: 'resident' for name in self.resident}
        after = {name: 'recompute' for name in selected} | {name: 'resident' for name in resident}
        # Restore old entries before installing any child/parent replacements.
        for name, mode in before.items():
            if after.get(name) != mode and name in self._modules:
                parent, attribute, original = self._modules[name]
                setattr(parent, attribute, original)
        for name, mode in after.items():
            if before.get(name) != mode and name in self._modules:
                parent, attribute, original = self._modules[name]
                key = (name, mode)
                if key not in self._wrappers:
                    if mode == 'recompute':
                        wrapper = checkpoint_wrapper(original, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                        wrapper.forward = skip_offload(wrapper.forward)
                    else:
                        wrapper = _ResidentRegion(original)
                    self._wrappers[key] = wrapper
                setattr(parent, attribute, self._wrappers[key])
        self.selected = selected
        self.resident = resident


def qwen_recompute_regions(layers):
    """Expose Qwen modules, projection-plus-SiLU and SiLU-plus-product groups.

    With pointwise selected, both independent projections precede the SiLU and
    product checkpoint. Otherwise the original evaluation order is retained.
    No kernels are fused or replaced, and no parameters or state keys are added.
    """
    regions = NamedRecomputeRegions(layers)
    for index, layer in enumerate(layers):
        prefix = str(index)
        for suffix in ('', '.self_attn', '.mlp', '.input_layernorm', '.post_attention_layernorm'):
            regions.register_module(prefix + suffix)
        mlp = layer.mlp

        def gate(x, module=mlp):
            return module.act_fn(module.gate_proj(x))

        gate_call = regions.register_callable(prefix + '.mlp.gate', gate)

        def pointwise(gate_value, up_value, module=mlp):
            return module.act_fn(gate_value) * up_value

        pointwise_name = prefix + '.mlp.pointwise'
        pointwise_call = regions.register_callable(pointwise_name, pointwise, offload_checkpoint_inputs=True)
        regions.register_conflict(prefix + '.mlp.gate', pointwise_name)

        def mlp_forward(x, module=mlp, gate_region=gate_call, pointwise_region=pointwise_call, name=pointwise_name):
            if name in regions.selected or name in regions.resident:
                return module.down_proj(pointwise_region(module.gate_proj(x), module.up_proj(x)))
            return module.down_proj(gate_region(x) * module.up_proj(x))

        mlp.forward = mlp_forward
    return regions


class LayerRecomputeRegions:
    """Select a prefix of a sequential layer collection for recomputation.

    Construct after the ordinary distributed wrapper, as in the existing Qwen
    integration. Original modules and Parameter objects retain their identity;
    only ModuleList entries change. This adapter owns the checkpoint wrappers.
    Call only between complete forward/backward steps with gradients cleared.
    General nested or overlapping recomputation regions are intentionally not
    inferred from an operator trace.
    """

    def __init__(self, layers: MutableSequence):
        self.layers = layers
        self.originals = tuple(layers)
        self.wrappers = {}
        self.count = 0

    def apply(self, count: int) -> None:
        if OffloadSession.get_active() is not None:
            raise RuntimeError('cannot switch checkpoint regions inside an active offload step')
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(self.originals):
            raise ValueError('recompute count is outside the registered layer range')
        for index in range(min(count, self.count), max(count, self.count)):
            if index < count:
                if index not in self.wrappers:
                    wrapper = checkpoint_wrapper(self.originals[index], checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                    wrapper.forward = skip_offload(wrapper.forward)
                    self.wrappers[index] = wrapper
                self.layers[index] = self.wrappers[index]
            else:
                self.layers[index] = self.originals[index]
        self.count = count
