"""Bounded actual-activation calibration, entirely outside scored replay.

Owned CPU snapshots are copied only during the initial raw warmup. Sampling
never changes a training tensor, retains a recycled Host-pool alias, patches a
global method, or resets the caller's allocator high-water counters. Native
codec computation remains on NPU; CPU data only builds its fixed codebook.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import _disable_current_modes

from .calibrated_transport import profiles_from_calibration


@dataclass(frozen=True)
class CalibrationLimits:
    sample_bytes: int = 512 * 1024**2
    size_groups: int = 3
    samples_per_group: int = 2
    repetitions: int = 2

    def __post_init__(self):
        for field in ('sample_bytes', 'size_groups', 'samples_per_group', 'repetitions'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('positive integer compression calibration limits required')
        if self.size_groups > 3 or self.samples_per_group > 2 or not 2 <= self.repetitions <= 5:
            raise ValueError('calibration is bounded to three sizes, two snapshots and two to five repetitions')


class ActivationSamples:
    """Instance-local first-warmup snapshots, with byte and stratum limits."""
    def __init__(self, limits=None):
        self.limits = limits or CalibrationLimits()
        self.items = []
        self.bytes = self.peak_bytes = self.combined_host_peak_bytes = 0
        self.capture_seconds = 0.0
        self.observed = 0
        self.closed = False
        self._manager = self._original = self._wrapper = None

    def _reserve(self, size, pool_bytes, host_limit):
        if size < 2 or size % 2 or size > self.limits.sample_bytes:
            return None
        group = (size - 1).bit_length()
        if sum(item['group'] == group for item in self.items) >= self.limits.samples_per_group:
            return None
        keep = list(self.items)
        while (sum(item['bytes'] for item in keep) + size > self.limits.sample_bytes
               or len({item['group'] for item in keep} | {group}) > self.limits.size_groups):
            smaller = [item for item in keep if item['group'] < group]
            if not smaller:
                return None
            victim = min(smaller, key=lambda item: (item['group'], item['bytes']))
            keep = [item for item in keep if item is not victim]
        if pool_bytes + sum(item['bytes'] for item in keep) + size > host_limit:
            return None
        self.items = keep
        self.bytes = sum(item['bytes'] for item in keep)
        return group

    def install(self, manager, host_limit):
        if self.closed or self._manager is not None:
            raise RuntimeError('activation sampling is one-shot')
        self._manager = manager
        original = self._original = manager.copy_d2h
        def copy(sid):
            buffer = manager._residency.get(sid)
            fresh = buffer is not None and buffer.host_buffer is None
            if fresh:
                from ..runtime.pinned_memory import _align_to_bucket
                # Training's Host cache can grow after sampling. Evict owned
                # snapshots BEFORE a worst-case new raw bucket is allocated,
                # including unsupported dtypes and already-full size strata.
                bound = manager._host_pool.total_allocated + _align_to_bucket(buffer.logical_nbytes)
                while self.items and bound + self.bytes > host_limit:
                    victim = min(self.items, key=lambda row:(row['group'], row['bytes']))
                    self.items = [item for item in self.items if item is not victim]
                    self.bytes -= victim['bytes']
                    del victim
            result = original(sid)
            self.combined_host_peak_bytes = max(self.combined_host_peak_bytes,
                                                manager._host_pool.total_allocated + self.bytes)
            if not fresh or self.closed:
                return result
            self.observed += 1
            buffer = manager._residency[sid]
            if buffer.storage_dtype != torch.bfloat16 or buffer.host_encoding is not None:
                return result
            group = self._reserve(buffer.logical_nbytes, manager._host_pool.total_allocated, host_limit)
            if group is None:
                return result
            started = time.perf_counter()
            try:
                with _disable_current_modes():
                    if buffer.host_event is not None:
                        buffer.host_event.synchronize()
                    owned = buffer.host_buffer.clone()
                size = owned.numel() * owned.element_size()
                if size != buffer.logical_nbytes:
                    raise RuntimeError('raw snapshot byte count differs from logical storage')
                self.items.append(dict(group=group, bytes=size, data=owned, storage_id=sid))
                self.bytes += size
                self.peak_bytes = max(self.peak_bytes, self.bytes)
                self.combined_host_peak_bytes = max(self.combined_host_peak_bytes,
                                                    manager._host_pool.total_allocated + self.bytes)
            finally:
                self.capture_seconds += time.perf_counter() - started
            return result
        self._wrapper = copy
        manager.copy_d2h = copy

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._manager is not None:
            if self._manager.copy_d2h is not self._wrapper:
                raise RuntimeError('activation capture method changed before retirement')
            self._manager.copy_d2h = self._original
            self._manager = self._original = self._wrapper = None

    def release(self):
        if not self.closed:
            raise RuntimeError('close sampling before releasing snapshots')
        self.items.clear()
        self.bytes = 0


@dataclass
class CalibrationResult:
    report: dict
    profiles: tuple = ()
    codebook_sample: object = None


def calibration_capability(library=None):
    path = library or os.environ.get('MINDSPEED_SPLITZIP_LIBRARY')
    if (not torch.accelerator.is_available() or not hasattr(torch, 'npu')
            or not torch.npu.is_available() or torch.accelerator.current_accelerator().type != 'npu'):
        return None, 'unavailable_npu'
    if not path or not Path(path).is_file():
        return None, 'unavailable_native_library'
    return str(path), 'available'


def calibrate_activations(samples, *, library, device, device_limit, host_limit, training_host_bytes):
    """Losslessly verify serialized raw/native costs on owned real data.

    Isolated inclusive times only rank full-step experiments. Existing device
    peaks are preserved, including failed or skipped calibration attempts.
    Capacity guards include source, restored bytes, worst-case packet and
    scratch; a favorable ratio never discounts memory reservations.
    """
    from mindspeed.core.memory.compression import SplitZipCodec
    from mindspeed.core.memory.compression.transport import TransferContext
    from ..runtime.residency import ResidencyManager
    from ..runtime.pinned_memory import _align_to_bucket
    if not samples.closed:
        raise ValueError('close activation capture before native calibration')
    started = time.perf_counter()
    report = dict(status='no_supported_samples', rows=[], capture_seconds=samples.capture_seconds,
        captured_snapshots=len(samples.items), sample_peak_bytes=samples.peak_bytes,
        combined_host_peak_bytes=max(training_host_bytes, samples.combined_host_peak_bytes),
        scope='isolated native calibration in excluded warmup; full-step trials decide performance',
        training_values_modified=False, resets_peak_counters=False)
    if not samples.items:
        return CalibrationResult(report)
    with _disable_current_modes():
        portions = []
        for item in samples.items:
            data = item['data']
            count = min(data.numel(), 64*1024) // 2 * 2
            for offset in (0, ((data.numel()-count)//4)*2, data.numel()-count):
                portions.append(data[offset:offset+count].view(torch.bfloat16))
        book_bytes = sum(part.numel()*part.element_size() for part in portions)
        host_base = training_host_bytes + samples.bytes + 3*book_bytes + 4*1024**2
        if host_base > host_limit or torch.npu.memory_allocated(device) + 4*1024**2 > device_limit:
            report['status'] = 'skipped_capacity'
            return CalibrationResult(report)
        codebook_sample = torch.cat(portions)
        codec = SplitZipCodec(library, sample=codebook_sample, max_inflight=1)
        managers = {}
        try:
            before = time.perf_counter()
            codec.prepare(device)
            torch.npu.synchronize(device)
            report['prepare_seconds'] = time.perf_counter() - before
            report['codebook_sha256'] = hashlib.sha256(memoryview(codebook_sample.view(torch.uint8).numpy()).cast('B')).hexdigest()
            for item in sorted(samples.items, key=lambda row: row['bytes']):
                size = item['bytes']; padded = ((size+16383)//16384)*16384
                device_bound = 6*padded + 4*1024**2
                pool_limit = 2*_align_to_bucket(padded+2048)
                host_bound = samples.bytes + 3*book_bytes + 2*pool_limit + size + 4*1024**2
                row = dict(bytes=size, source_sha256=hashlib.sha256(memoryview(item['data'].numpy()).cast('B')).hexdigest(),
                    required_extra_device_bytes=device_bound, required_extra_host_bytes=host_bound,
                    status='skipped_capacity', measurements=[])
                report['rows'].append(row)
                if not codec.accepts(TransferContext(None, item['storage_id'], size, torch.bfloat16, device)):
                    row['status'] = 'unsupported_codec_storage'
                    continue
                if (torch.npu.memory_allocated(device)+device_bound > device_limit
                        or training_host_bytes+host_bound > host_limit):
                    continue
                source = item['data'].view(torch.bfloat16).to(device)
                torch.npu.synchronize(device)
                managers = {name:ResidencyManager(pool_limit, codec=value)
                            for name,value in (('raw',None),('codec',codec))}
                def roundtrip(name):
                    manager = managers[name]
                    buffer = manager.bind(1, source)
                    torch.npu.synchronize(device)
                    before = time.perf_counter()
                    manager.copy_d2h(1)
                    if buffer.host_event is not None:
                        buffer.host_event.synchronize()
                    d2h = time.perf_counter()-before
                    encoding = buffer.host_encoding
                    packet = (codec.host_view(buffer.host_buffer, encoding.metadata).numel()
                              if encoding is not None else buffer.host_buffer.numel())
                    if name == 'codec' and encoding is None:
                        raise RuntimeError('native calibration unexpectedly fell back to raw transport')
                    manager.release_device(1)
                    before = time.perf_counter()
                    manager.copy_h2d(1)
                    if buffer.device_event is not None:
                        buffer.device_event.synchronize()
                    h2d = time.perf_counter()-before
                    restored = buffer.device_buffer.cpu()
                    if not torch.equal(restored, item['data']):
                        raise RuntimeError('native activation compression failed bit-exact roundtrip')
                    peak = torch.npu.max_memory_allocated(device)
                    if peak > device_limit:
                        raise RuntimeError('native calibration exceeded the device capacity')
                    manager.clear_runtime(set())
                    return dict(d2h_inclusive_seconds=d2h, h2d_inclusive_seconds=h2d,
                                packet_bytes=packet, lossless_exact=True, peak_allocated_bytes=peak)
                row['warmup'] = {name:roundtrip(name) for name in ('raw','codec')}
                for index in range(samples.limits.repetitions):
                    order = ('raw','codec') if index % 2 == 0 else ('codec','raw')
                    row['measurements'].append(dict(index=index, order=order,
                                                   results={name:roundtrip(name) for name in order}))
                row['status'] = 'complete'
                report['combined_host_peak_bytes'] = max(report['combined_host_peak_bytes'],
                                                         training_host_bytes+host_bound)
                for manager in managers.values():
                    manager.sync_all_transfers(); manager.clear_runtime(set())
                managers.clear()
                del source, manager
            report.update(status='complete', codec_base=codec._base, codec_stats=dict(codec.stats),
                          peak_allocated_bytes=torch.npu.max_memory_allocated(device))
            profiles = profiles_from_calibration(report)
            return CalibrationResult(report, profiles, codebook_sample if profiles else None)
        finally:
            # Cleanup errors are not swallowed: a failed drain cannot certify
            # that training may safely resume after a native codec failure.
            for manager in managers.values():
                manager.sync_all_transfers(); manager.clear_runtime(set())
            codec.close()
            report['calibration_seconds'] = time.perf_counter()-started
