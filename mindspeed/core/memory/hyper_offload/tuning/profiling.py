"""Opt-in, first-forward region observations for model-independent proposals.

Boundary storage bytes are NOT saved-activation estimates. The trace interval
and opaque flag let a provider distinguish visible evidence from a checkpoint
whose internal storage lifetimes were not observed. All hooks are removed by
finish_forward(), before backward and before any stable replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_flatten


def _active_session():
    from mindspeed.core.memory.hyper_offload.api.session import OffloadSession
    return OffloadSession.get_active()


def _boundary(tensors):
    shapes, storages = [], {}
    for tensor in tree_flatten(tensors)[0]:
        if not isinstance(tensor, torch.Tensor):
            continue
        shapes.append((tuple(tensor.shape), str(tensor.dtype)))
        physical = getattr(tensor, '_buffer', None)
        if physical is not None and hasattr(tensor, 'storage_id'):
            # Never demand-page a ShadowTensor merely to collect metadata.
            key = ('shadow', id(physical), tensor.storage_id)
            storages[key] = physical.logical_nbytes
        else:
            storage = tensor.untyped_storage()
            storages[('storage', storage._cdata)] = storage.nbytes()
    return tuple(shapes), sum(storages.values())


def _boundary_ids(tensors, session):
    """Read existing trace identities without registering or materializing data."""
    tracker = getattr(session.executor, '_tracker', None)
    values = set()
    for tensor in tree_flatten(tensors)[0]:
        if not isinstance(tensor, torch.Tensor):
            continue
        sid = getattr(tensor, 'storage_id', None)
        if sid is None and tracker is not None:
            sid = tracker.get_activation_sid(tensor)
        if sid is not None:
            values.add(sid)
    return tuple(sorted(values))


@dataclass
class RegionObservation:
    name: str
    structure: tuple
    input_shapes: tuple
    input_boundary_bytes: int
    first_op: int
    opaque: bool
    input_storage_ids: tuple = ()
    output_storage_ids: tuple = ()
    output_shapes: tuple = ()
    output_boundary_bytes: int = 0
    last_op: int = -1
    inclusive_interval_ms: float = 0.0
    complete: bool = False


class RegionProfiler:
    """A finite observer over adapter-certified regions, independent of names.

    Module identity is represented by its type and parameter shapes. Callable
    groups use their code identity and observed input/output shapes. These are
    candidate equivalence features, not proof of interchangeable semantics.
    Real trace versions and full-step admission still determine validity.
    """
    def __init__(self, regions, *, max_records=2048, event_factory=None,
                 stream_provider=None, session_provider=None):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError('max_records must be a positive integer')
        self.regions, self.max_records = regions, max_records
        self._event = event_factory or (lambda: torch.Event(enable_timing=True))
        self._stream = stream_provider or (lambda: torch.accelerator.current_stream())
        self._session = session_provider or _active_session
        self._handles, self._pending, self._events = [], {}, []
        self._structures = {}
        self.rows = []
        self.started = self.closed = self.truncated = False
        self.forward_end_op = None
        self.finalize_seconds = 0.0

    def start(self):
        if self.started or getattr(self.regions, 'observer', None) is not None:
            raise RuntimeError('region profiling is one-shot and requires an unused observer')
        self.started = True
        self.regions.observer = self
        for name, (_, _, module) in self.regions._modules.items():
            self._structures[name] = (type(module).__module__, type(module).__qualname__,
                                      tuple(tuple(p.shape) for p in module.parameters()))
            def before(module, args, kwargs, name=name):
                self._pending.setdefault(name, []).append(self.begin(name, (args, kwargs)))
            def after(module, args, kwargs, result, name=name):
                stack = self._pending.get(name)
                if stack:
                    self.end(stack.pop(), result)
            self._handles.extend((module.register_forward_pre_hook(before, with_kwargs=True),
                                  module.register_forward_hook(after, with_kwargs=True)))
        for name, function in self.regions._callables.items():
            self._structures[name] = (function.__module__, function.__qualname__)
        return self

    def begin(self, name, args):
        session = self._session()
        if self.closed or session is None or session.mode != 'warmup':
            return None
        if len(self.rows) >= self.max_records:
            self.truncated = True
            return None
        with _disable_current_modes():
            shapes, size = _boundary(args)
            event = self._event()
            event.record(self._stream())
        row = RegionObservation(name, self._structures.get(name, ()), shapes, size,
                                session.executor.op_idx + 1, session.executor.in_opaque_region)
        with _disable_current_modes():
            row.input_storage_ids = _boundary_ids(args, session)
        self.rows.append(row)
        token = (row, event)
        return token

    def end(self, token, result):
        if token is None:
            return
        row, started = token
        session = self._session()
        with _disable_current_modes():
            ended = self._event()
            ended.record(self._stream())
            row.output_shapes, row.output_boundary_bytes = _boundary(result)
            row.output_storage_ids = _boundary_ids(result, session)
        row.last_op = session.executor.op_idx
        row.complete = True
        self._events.append((row, started, ended))

    def abort(self, token):
        # An incomplete call is not a zero-cost region.
        if token is not None:
            token[0].complete = False

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()
        if getattr(self.regions, 'observer', None) is self:
            self.regions.observer = None
        self.closed = True

    def finish_forward(self):
        started = time.perf_counter()
        session = self._session()
        self.forward_end_op = None if session is None else session.executor.op_idx
        self.close()
        try:
            with _disable_current_modes():
                for row, first, last in self._events:
                    last.synchronize()
                    row.inclusive_interval_ms = first.elapsed_time(last)
        finally:
            self._events.clear()
            self.finalize_seconds += time.perf_counter() - started
        return dict(regions=[asdict(row) for row in self.rows], truncated=self.truncated,
                    forward_end_op=self.forward_end_op, finalize_seconds=self.finalize_seconds,
                    timing_scope='inclusive device interval, not pure compute or recomputation cost',
                    memory_scope='unique boundary storage bytes, not saved activations')
