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
"""Pure physical control plane for byte-level residency.

This module provides the data-plane (:class:`PhysicalBuffer`) and
the control-plane (:class:`ResidencyManager`) for raw byte buffers.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field
from functools import wraps

import torch
from torch.utils._python_dispatch import _disable_current_modes

from mindspeed.core.memory.hyper_offload.runtime.pinned_memory import PinnedMemoryPool
from mindspeed.core.memory.compression.transport import HostEncoding, TransferContext, TransferNode

logger = logging.getLogger(__name__)


def _without_dispatch_modes(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # Runtime copies and allocations are not model activations.
        with _disable_current_modes():
            return fn(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class _HostReuseFence:
    """Preserve both the writer and reader when restoration fails early.

    A deferred writer can fail before its wait has reached the reader stream.
    Synchronizing only that reader then says nothing about the Host writer.
    """

    writer: object
    reader: object

    def query(self):
        return self.writer.query() and self.reader.query()

    def synchronize(self):
        try:
            self.writer.synchronize()
        finally:
            self.reader.synchronize()


@dataclass
class PhysicalBuffer:
    """Pure physical memory block tracking host/device byte buffers.

    This dataclass is intentionally minimal — it carries **no** knowledge
    of logical tensors or ShadowTensor objects.  Every field is
    ``None`` when the corresponding resource is not held.

    The **control plane** (:class:`ResidencyManager`) orchestrates
    allocations and copies.
    """

    device: torch.device | None = None
    """Target accelerator device (set on first registration)."""

    host_buffer: torch.Tensor | None = None
    """1-D uint8 pinned CPU buffer, or ``None`` when not resident on host."""

    host_event: torch.Event | None = None
    """:class:`torch.Event` recorded after the latest D2H copy completes."""

    device_buffer: torch.Tensor | None = None
    """1-D uint8 device buffer, or ``None`` when not resident on device."""

    device_event: torch.Event | None = None
    """:class:`torch.Event` recorded after the latest H2D copy completes."""

    host_access_event: torch.Event | torch.Stream | None = None
    """Latest access to the current Host allocation on the copy stream."""

    host_pool_ref: weakref.ReferenceType | None = None
    """Do not retain a session's entire cache through one surviving output."""

    host_encoding: HostEncoding | None = None
    logical_nbytes: int = 0
    storage_dtype: torch.dtype | None = None

    @property
    def host_pool(self):
        return self.host_pool_ref() if self.host_pool_ref is not None else None

    shadows: weakref.WeakSet = field(default_factory=weakref.WeakSet, repr=False, compare=False)
    """Live aliases, including views created outside the dispatch session."""

    device_copy_stream_recorded: bool = False
    """The current destination's allocator lifetime includes its H2D stream."""

    def prepare_for_write(self) -> None:
        """Restore device data and retire its host snapshot before a mutation."""
        # D2H may still be reading the device storage being modified.
        if self.host_event is not None:
            self.host_event.synchronize()
            self.host_event = None
        self.device_storage()
        if self.host_buffer is not None:
            pool = self.host_pool
            if pool is not None:
                pool.release(self.host_buffer, event=self.host_reuse_event)
            self.host_buffer = None
            self.host_access_event = None
            self.host_encoding = None

    @property
    def host_reuse_event(self):
        """Preserve access order, including a D2H following an older H2D."""
        if self.host_access_event is not None:
            return self.host_access_event
        return self.device_event if self.device_event is not None else self.host_event

    # ------------------------------------------------------------------
    # Device-storage access (called by ShadowTensor)
    # ------------------------------------------------------------------

    def device_storage(self) -> torch.UntypedStorage:
        """Return the device storage, waiting for any in-flight H2D event.

        If the device buffer is not resident but host data is available,
        this will synchronously demand-page the data back to the device.

        Returns:
            The underlying :class:`torch.UntypedStorage` of ``device_buffer``.

        """
        if self.device_buffer is None:
            if self.host_buffer is not None and self.device is not None:
                if self.host_encoding is not None:
                    # Surviving shadows must remain usable after their session
                    # and residency manager are gone. The snapshot owns its codec.
                    with _disable_current_modes():
                        _restore_encoded(self, torch.accelerator.current_stream())
                    return self.device_storage()
                # Synchronous demand-paging fallback.
                # Wait for any in-flight D2H copy to complete before reading
                # from host_buffer (race-condition avoidance).
                if self.host_event is not None:
                    self.host_event.synchronize()
                    self.host_event = None

                size_bytes = self.host_buffer.numel()
                dev_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
                dev_bytes.copy_(self.host_buffer, non_blocking=False)
                self.device_buffer = dev_bytes
                self.device_copy_stream_recorded = False
            else:
                raise RuntimeError(
                    "No device buffer available and cannot demand-page. "
                    "Ensure host data is available if device data is evicted."
                )
        if self.device_event is not None:
            current_stream = torch.accelerator.current_stream()
            self.device_event.wait(current_stream)
            self.device_buffer.record_stream(current_stream)
            # A stream wait orders consumers; it does not finish the transfer.
            # Keep the event until Host recycling can wait for its H2D reader.
        return self.device_buffer.untyped_storage()


def _restore_encoded(buffer, stream):
    """Restore a snapshot on a known stream; record completion after decoding."""
    encoding = buffer.host_encoding
    packet = None
    producer = torch.accelerator.current_stream()
    try:
        with stream:
            stream.wait_stream(producer)
            if buffer.host_event is not None:
                buffer.host_event.wait(stream)
            host = buffer.host_buffer
            host_view = getattr(encoding.codec, "host_view", None)
            if host_view is not None:
                host = host_view(host, encoding.metadata)
            packet = torch.empty(host.numel(), dtype=torch.uint8, device=buffer.device)
            packet.record_stream(stream)
            packet.copy_(host, non_blocking=host.is_pinned())
            output = encoding.codec.decode(packet, encoding.metadata, stream)
            if (output.dtype != torch.uint8 or output.device != buffer.device
                    or output.numel() != encoding.original_bytes or not output.is_contiguous()
                    or output.storage_offset() != 0):
                raise RuntimeError("codec returned an invalid restored storage")
            output.record_stream(stream)
            event = torch.Event()
            event.record(stream)
    except Exception as error:
        buffer.host_access_event = (
            _HostReuseFence(buffer.host_event, stream) if buffer.host_event is not None else stream
        )
        try:
            stream.synchronize()
        except Exception as sync_error:
            raise error from sync_error
        raise
    buffer.device_buffer = output
    buffer.device_event = event
    buffer.host_access_event = event


class ResidencyManager:
    """Pure physical controller for byte-level tensor residency.

    Owns
    ----
    * Physical residency table (``storage_id → PhysicalBuffer``).
    * High-level state transitions (``copy_d2h``, ``copy_h2d``).

    All public methods accept ``storage_id: int``.
    """

    def __init__(
        self,
        max_host_bytes: int,
        allow_pageable_fallback: bool = False,
        codec=None,
        prefetch_on_copy_stream: bool = False,
        defer_h2d_release: bool = False,
    ) -> None:
        self._copy_stream = None
        self.prefetch_on_copy_stream = prefetch_on_copy_stream
        self.defer_h2d_release = defer_h2d_release
        self._host_pool = PinnedMemoryPool(max_host_bytes, allow_pageable_fallback)
        self._residency: dict[int, PhysicalBuffer] = {}
        self._codec = codec
        self._transfer_node = None
        self._quarantined = []

    def set_transfer_node(self, op_id, phase, name=""):
        """Adapter boundary: another execution system may supply its own node."""
        if self._codec is not None:
            self._transfer_node = TransferNode(op_id, phase, name)

    def _get_copy_stream(self) -> torch.Stream:
        """Return the internal copy stream, creating it lazily on first access.

        Delaying stream creation avoids requiring an accelerator device context
        at construction time (e.g. when running on CPU-only hosts or before
        NPU/CUDA is initialised).
        """
        if self._copy_stream is None:
            self._copy_stream = torch.Stream()
        return self._copy_stream

    # ------------------------------------------------------------------
    # Device-side memory query
    # ------------------------------------------------------------------

    @property
    def resident_bytes(self) -> int:
        """Total bytes currently resident on the device side across all storage IDs."""
        total = 0
        for buf in self._residency.values():
            if buf.device_buffer is not None:
                total += buf.device_buffer.numel()
        return total

    def device_resident_size(self, sid: int) -> int | None:
        """Return the device buffer size in bytes, or ``None`` if not device-resident."""
        buffer = self._residency.get(sid)
        if buffer is None or buffer.device_buffer is None:
            return None
        return buffer.device_buffer.numel()

    @property
    def retained_sids(self) -> set[int]:
        """Include aliases whose wrapper was created outside executor.make_shadow."""
        return {sid for sid, buffer in self._residency.items() if buffer.shadows}

    # ------------------------------------------------------------------
    # Stream synchronisation
    # ------------------------------------------------------------------

    def wait_for_transfers(self) -> None:
        """Make the current accelerator stream wait for pending async transfers on the copy stream."""
        consumer = torch.accelerator.current_stream()
        consumer.wait_stream(self._get_copy_stream())
        wait = getattr(self._codec, "wait_for_transfers", None)
        if wait is not None:
            wait(consumer)

    def sync_all_transfers(self) -> None:
        """Synchronise streams."""
        self._get_copy_stream().synchronize()
        synchronize = getattr(self._codec, "synchronize", None)
        if synchronize is not None:
            synchronize()

    # ------------------------------------------------------------------
    # Registration: bind a storage ID to a tensor's device storage
    # ------------------------------------------------------------------

    def bind(self, sid: int, tensor: torch.Tensor) -> PhysicalBuffer:
        """Point the physical buffer for *sid* at *tensor*'s device storage.

        Returns the :class:`PhysicalBuffer` so that the caller can pass
        it to a new :class:`~offload.execution.tensor.ShadowTensor`.
        """
        storage = tensor.untyped_storage()
        buffer = self._residency.get(sid)
        if buffer is not None and buffer.device_buffer is not None:
            current = buffer.device_buffer
            # Aliased outputs already share the full-storage byte view. Reuse
            # it without retaining another tensor or changing transfer fences.
            # A storage can grow without changing identity, so check its size.
            if (current.untyped_storage()._cdata == storage._cdata
                    and current.numel() == storage.nbytes()):
                # View reuse must still invalidate codec eligibility when a
                # storage is reinterpreted through a different scalar type.
                if buffer.storage_dtype != tensor.dtype:
                    buffer.storage_dtype = None
                return buffer
        dev_view = torch.empty(0, dtype=torch.uint8, device=tensor.device)
        dev_view.set_(storage, 0, (storage.size(),), (1,))
        if sid not in self._residency:
            self._residency[sid] = PhysicalBuffer(host_pool_ref=weakref.ref(self._host_pool))
        buffer = self._residency[sid]
        old_storage = buffer.device_buffer.untyped_storage() if buffer.device_buffer is not None else None
        if old_storage is None or old_storage._cdata != storage._cdata:
            if buffer.host_buffer is not None:
                self._host_pool.release(buffer.host_buffer, event=buffer.host_reuse_event)
            buffer.host_buffer = None
            buffer.host_event = None
            buffer.host_access_event = None
            buffer.device_event = None
            buffer.device_copy_stream_recorded = False
            buffer.host_encoding = None
            buffer.storage_dtype = tensor.dtype
        elif buffer.storage_dtype != tensor.dtype:
            # A reinterpreted storage is not known to contain BF16 activations.
            buffer.storage_dtype = None
        buffer.device = tensor.device
        buffer.device_buffer = dev_view
        buffer.logical_nbytes = storage.nbytes()
        return buffer

    # ------------------------------------------------------------------
    # State transition: copy D2H
    # ------------------------------------------------------------------

    @_without_dispatch_modes
    def copy_d2h(self, sid: int) -> None:
        """Copy the physical storage for *sid* from device to host.

        1. Look up the physical buffer for ``sid``.
        2. If ``host_buffer`` is already present → no-op.
        3. Launch an async D2H copy.
        4. Keep the device buffer resident until ``release_device``.
        """
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(f"copy_d2h sid={sid}: no physical buffer registered")
        if buffer.host_buffer is not None:
            logger.debug("copy_d2h sid=%d: already on host, skip", sid)
            return
        if buffer.device_buffer is None:
            raise RuntimeError(f"copy_d2h sid={sid}: no device data to copy")

        dev_src = buffer.device_buffer
        size_bytes = dev_src.numel()
        logger.debug(
            "copy_d2h sid=%d: copying %d bytes (%.2f MiB) D2H",
            sid,
            size_bytes,
            size_bytes / 1024**2,
        )

        copy_stream = self._get_copy_stream()
        host_buf = None

        # 2. Prevent the caching allocator from recycling the source
        #    memory while the copy stream reads it.
        event = None
        encoding = None
        try:
            if dev_src.device == copy_stream.device:
                dev_src.record_stream(copy_stream)

            # 3. Launch asynchronous D2H copy.
            if dev_src.device != copy_stream.device:
                host_buf = self._host_pool.acquire(size_bytes)
                host_buf.copy_(dev_src, non_blocking=False)
            else:
                producer_stream = torch.accelerator.current_stream()
                with copy_stream:
                    copy_stream.wait_stream(producer_stream)
                    transfer_src = dev_src
                    selected = self._codec is not None and self._codec.accepts(TransferContext(
                        self._transfer_node, sid, size_bytes, buffer.storage_dtype, buffer.device,
                    ))
                    capacity = getattr(self._codec, "host_capacity", None)
                    host_bytes = capacity(size_bytes) if selected and capacity is not None else None
                    if host_bytes is not None:
                        host_buf = self._host_pool.acquire(host_bytes)
                        if host_buf.is_pinned():
                            submission = self._codec.encode_to_host(dev_src, host_buf, copy_stream)
                            encoding = HostEncoding(self._codec, submission.metadata, size_bytes)
                            event = submission.completion
                        else:
                            # Native async DMA requires pinned pages. Preserve
                            # the upstream synchronous pageable fallback.
                            self._host_pool.release(host_buf)
                            host_buf = None
                    if selected and encoding is None:
                        packed = self._codec.encode(dev_src, copy_stream)
                        if (packed.data.dtype != torch.uint8 or packed.data.device != dev_src.device
                                or not packed.data.is_contiguous()):
                            raise RuntimeError("codec returned an invalid encoded buffer")
                        packed.data.record_stream(copy_stream)
                        transfer_src = packed.data
                        encoding = HostEncoding(self._codec, packed.metadata, size_bytes)
                    if host_buf is None:
                        host_buf = self._host_pool.acquire(transfer_src.numel())
                        pinned = host_buf.is_pinned()
                        event = torch.Event() if pinned else None
                        host_buf.copy_(transfer_src, non_blocking=pinned)
                        if pinned:
                            event.record(copy_stream)
                        else:
                            copy_stream.synchronize()
        except Exception as transfer_error:
            # A failed event record may follow an already enqueued copy. Only
            # recycle after that stream is safe; failed synchronization leaves
            # the allocation quarantined rather than reusing in-flight bytes.
            if dev_src.device == copy_stream.device:
                try:
                    copy_stream.synchronize()
                    synchronize = getattr(self._codec, "synchronize", None)
                    if synchronize is not None:
                        synchronize()
                except Exception as sync_error:
                    # Streams expose query()/synchronize() like events. A
                    # failed drain still needs an ownership fence in the pool.
                    if host_buf is not None:
                        if getattr(self._codec, "synchronize", None) is not None:
                            self._quarantined.append(host_buf)
                        else:
                            self._host_pool.release(host_buf, event=copy_stream)
                    raise transfer_error from sync_error
            if host_buf is not None:
                self._host_pool.release(host_buf)
            raise

        # 4. Update physical buffer.
        buffer.host_buffer = host_buf
        buffer.host_event = event
        buffer.host_access_event = event
        buffer.host_encoding = encoding

        logger.debug("copy_d2h sid=%d: done", sid)

    # ------------------------------------------------------------------
    # State transition: copy H2D
    # ------------------------------------------------------------------

    @_without_dispatch_modes
    def copy_h2d(self, sid: int) -> None:
        """Asynchronously copy (H2D) the physical storage for *sid* to device.

        Allocates fresh device memory, launches an async H2D copy on the
        copy stream, and updates the physical buffer with the
        new ``device_buffer`` and ``device_event``.  Returns immediately
        without waiting for the copy to complete.
        """
        buffer = self._residency.get(sid)
        if buffer is None:
            raise RuntimeError(f"copy_h2d sid={sid}: no physical buffer registered")

        if buffer.device_buffer is not None:
            logger.debug("copy_h2d sid=%d: already on device, skip", sid)
            return

        if buffer.host_buffer is None:
            raise RuntimeError(f"copy_h2d sid={sid}: no host data to copy")

        if buffer.device is None:
            raise RuntimeError(f"copy_h2d sid={sid}: target device unknown")

        if buffer.host_encoding is not None:
            stream = self._get_copy_stream()
            _restore_encoded(buffer, stream)
            return

        size_bytes = buffer.host_buffer.numel()
        logger.debug(
            "copy_h2d sid=%d: copying %d bytes (%.2f MiB) H2D to %s",
            sid,
            size_bytes,
            size_bytes / 1024**2,
            buffer.device,
        )

        copy_stream = self._get_copy_stream()
        # Allocate on the copy stream only when it will own the async copy.
        # Its allocator reuse is then ordered by that same stream; no dependency
        # on unrelated work already queued on the compute stream is necessary.
        # Consumers still wait on device_event and record their stream below
        # in PhysicalBuffer.device_storage().
        copy_owned = (
            self.prefetch_on_copy_stream and buffer.host_buffer.is_pinned() and buffer.device == copy_stream.device
        )
        if copy_owned:
            with copy_stream:
                dev_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=buffer.device)
        else:
            dev_bytes = torch.empty(size_bytes, dtype=torch.uint8, device=buffer.device)

        # Launch H2D, preserving both Host reuse and destination reuse fences.
        event = None
        try:
            if not buffer.host_buffer.is_pinned():
                if dev_bytes.device == copy_stream.device:
                    producer_stream = torch.accelerator.current_stream()
                    with copy_stream:
                        copy_stream.wait_stream(producer_stream)
                        if buffer.host_event is not None:
                            buffer.host_event.wait(copy_stream)
                        dev_bytes.copy_(buffer.host_buffer, non_blocking=False)
                    copy_stream.synchronize()
                else:
                    dev_bytes.copy_(buffer.host_buffer, non_blocking=False)
            elif dev_bytes.device != copy_stream.device:
                dev_bytes.copy_(buffer.host_buffer, non_blocking=False)
            else:
                dev_bytes.record_stream(copy_stream)
                producer_stream = torch.accelerator.current_stream()
                event = torch.Event()
                with copy_stream:
                    if not copy_owned:
                        copy_stream.wait_stream(producer_stream)
                    if buffer.host_event is not None:
                        buffer.host_event.wait(copy_stream)
                    dev_bytes.copy_(buffer.host_buffer, non_blocking=True)
                    event.record(copy_stream)
        except Exception as transfer_error:
            if dev_bytes.device == copy_stream.device:
                buffer.host_access_event = copy_stream
                try:
                    copy_stream.synchronize()
                except Exception as sync_error:
                    raise transfer_error from sync_error
            raise

        # 3. Update physical buffer.
        buffer.device_buffer = dev_bytes
        buffer.device_event = event
        buffer.device_copy_stream_recorded = event is not None
        buffer.host_access_event = event

    # ------------------------------------------------------------------
    # Release helpers
    # ------------------------------------------------------------------

    def release_device(self, sid: int) -> None:
        """Release device-resident bytes for a storage ID.

        Normally waits for an in-flight H2D before dropping the destination.
        With deferred release enabled, destinations whose copy stream was
        recorded can instead rely on allocator stream fences. Consumers also
        record their stream in ``device_storage``; Host recycling retains the
        H2D reader event independently. Untracked destinations still wait.
        Pending D2H copies are protected by ``record_stream`` during
        ``copy_d2h``.
        """
        buffer = self._residency.get(sid)
        if buffer is None or buffer.device_buffer is None:
            return

        if buffer.device_event is not None and not (
            self.defer_h2d_release and buffer.device_copy_stream_recorded
        ):
            buffer.device_event.synchronize()
            buffer.device_event = None

        if buffer.host_buffer is None and buffer.shadows:
            # Final-use release retires scheduler ownership. Python may keep a
            # different set of outputs alive than it did during warmup; these
            # shadows must continue owning their device storage until they die.
            del self._residency[sid]
            return
        buffer.device_buffer = None
        buffer.device_copy_stream_recorded = False
        if buffer.host_buffer is None:
            del self._residency[sid]

    @_without_dispatch_modes
    def release_host(self, sid: int) -> None:
        """Release host-resident bytes for a storage ID.

        Waits for any in-flight H2D copy that may be reading from the
        host buffer before returning it to the pool.
        """
        buffer = self._residency.get(sid)
        if buffer is None or buffer.host_buffer is None:
            return

        # Ensure any in-flight H2D copy that reads this host buffer
        # has completed before we return it to the pool.
        if buffer.device_buffer is None and buffer.shadows:
            buffer.device_storage()
        self._host_pool.release(buffer.host_buffer, event=buffer.host_reuse_event)
        buffer.host_buffer = None
        buffer.host_event = None
        buffer.host_access_event = None
        buffer.host_encoding = None
        if buffer.device_buffer is None:
            del self._residency[sid]

    @_without_dispatch_modes
    def release_unreferenced(self, protected_sids: set[int]) -> None:
        """Recycle buffers whose activation wrappers have all been freed."""
        for sid, buffer in list(self._residency.items()):
            if sid in protected_sids or buffer.shadows:
                continue
            if buffer.device_event is not None:
                buffer.device_event.synchronize()
            if buffer.host_buffer is not None:
                self._host_pool.release(buffer.host_buffer, event=buffer.host_reuse_event)
            del self._residency[sid]

    # ------------------------------------------------------------------
    # Runtime clear
    # ------------------------------------------------------------------

    def clear_runtime(self, retained_sids: set[int] | None = None) -> None:
        """Detach live shadows onto device storage before recycling host buffers.

        Executors supply their live shadow IDs; standalone callers conservatively
        preserve all buffers. Only retained buffers are restored, so dead
        activations do not cause a device-memory spike during cleanup.
        """
        for sid, buffer in self._residency.items():
            if retained_sids is None or sid in retained_sids or buffer.shadows:
                buffer.device_storage()
            if buffer.host_buffer is not None:
                self._host_pool.release(buffer.host_buffer, event=buffer.host_reuse_event)
                buffer.host_buffer = None
                buffer.host_event = None
                buffer.host_access_event = None
                buffer.host_encoding = None
        self._residency.clear()
