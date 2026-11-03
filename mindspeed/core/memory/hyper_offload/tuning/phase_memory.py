"""Opt-in first-warmup allocator and opaque checkpoint lifetime observations.

Reads allocator counters after ordinary timed warmup ops. It never resets peak
statistics, reads activation values, or inserts tensor operations. The original
executor methods and all hooks are restored before warmup finish returns.
"""
from __future__ import annotations

import time
import hashlib
import json
import torch
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_flatten


def _memory(device):
    # The four public counters each rebuild the same statistics dictionary on
    # NPU/CUDA. Read one coherent snapshot without resetting any peak.
    stats = device.memory_stats()
    return tuple(stats[key] for key in ('allocated_bytes.all.current','allocated_bytes.all.peak',
                                       'reserved_bytes.all.current','reserved_bytes.all.peak'))


class PhaseMemoryObserver:
    def __init__(self, session, regions, *, reader=None, on_finish=None, max_ops=100000):
        if session.mode != 'warmup':
            raise ValueError('phase memory observations require an unexecuted warmup')
        self.session, self.regions = session, regions
        self.executor = session.executor
        if reader is None:
            device = torch.get_device_module(torch.accelerator.current_accelerator())
            self.reader = lambda: _memory(device)
        else:
            self.reader = reader
        self.on_finish = on_finish
        self.max_ops = max_ops
        self.samples, self.regions_seen = [], {}
        self._handles = []
        self._gradient_handles = []
        self.closed = False
        self.truncated = False
        self.seconds = 0.0
        self.report = None
        self.forward_end_op = None
        self._original_end = self.executor.on_op_end
        self._original_finish = self.executor.finish
        self.executor.on_op_end = self._after_op
        self.executor.finish = self._finish
        for name, (_, _, module) in regions._modules.items():
            def after(module, args, result, name=name):
                if name not in self.regions.selected or not self.executor.in_opaque_region:
                    return
                row = dict(name=name, forward_op=self.executor.op_idx, backward_ops=[])
                self.regions_seen.setdefault(name, []).append(row)
                for value in tree_flatten(result)[0]:
                    if isinstance(value, torch.Tensor) and value.requires_grad:
                        def backward(gradient, row=row):
                            row['backward_ops'].append(self.executor.op_idx)
                            return gradient
                        self._gradient_handles.append(value.register_hook(backward))
            self._handles.append(module.register_forward_hook(after))

    def finish_forward(self):
        """Stop module observation before backward can recompute these modules."""
        self.forward_end_op = self.executor.op_idx
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _after_op(self, result):
        value = self._original_end(result)
        started = time.perf_counter()
        try:
            if len(self.samples) >= self.max_ops:
                self.truncated = True
            else:
                with _disable_current_modes():
                    self.samples.append((self.executor.op_idx, *self.reader()))
        finally:
            self.seconds += time.perf_counter() - started
        return value

    def _finish(self):
        try:
            result = self._original_finish()
            finalize_started = time.perf_counter()
            trace = result[0] if isinstance(result, tuple) and result else None
            signature = None
            if hasattr(trace, 'ops'):
                shape = [(op.name, [(a.storage_id, a.kind.name) for a in op.accesses]) for op in trace.ops]
                signature = hashlib.sha256(json.dumps((shape,trace.storage_sizes),sort_keys=True).encode()).hexdigest()
            self.report = dict(trace_signature=signature, columns=('op','allocated_bytes','cumulative_peak_bytes','reserved_bytes','cumulative_reserved_peak_bytes'),
                               samples=self.samples, regions=self.regions_seen, truncated=self.truncated, forward_end_op=self.forward_end_op,
                               observation_seconds=self.seconds,
                               scope='allocator counters after warmup operators; cumulative peaks are conservative, not per-op peaks')
            from .phase_capacity import summarize_phase_capacity
            self.report['capacity_summary'] = None
            if self.samples and not self.truncated and self.forward_end_op is not None:
                self.report['capacity_summary'] = summarize_phase_capacity(self.report,op_count=len(self.samples))
            self.seconds += time.perf_counter()-finalize_started
            self.report['observation_seconds'] = self.seconds
            if self.on_finish is not None:
                self.on_finish(self.report)
            return result
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.executor.on_op_end = self._original_end
        self.executor.finish = self._original_finish
        for handle in self._handles + self._gradient_handles:
            handle.remove()
        self._handles.clear()
        self._gradient_handles.clear()
        self.closed = True
