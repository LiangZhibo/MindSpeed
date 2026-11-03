"""Whole-storage SplitZip v2 using the experiment's native AscendC kernels.

Phase one transfers one whole storage at a time. A native progress worker may
read the 2 KiB header and submit its payload DMA without blocking model dispatch.
There is no chunk pipeline or automatic performance policy. Allocation/copy
operations below move bytes only; all compression computation is native.
"""
from __future__ import annotations

from collections import Counter, deque
from contextlib import nullcontext
import ctypes as ct
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import weakref

import torch

from .transport import EncodedBuffer, HostSubmission, TransferContext


_QUARANTINED = []


def _destroy_async(native, pointer, resources, counter, observed):
    # Keep every DMA resource alive while the native request finishes. A failed
    # device drain must not release buffers back to either caching allocator.
    if pointer.value:
        if not observed[0]:
            stored, raw = ct.c_uint32(), ct.c_uint32()
            status = native.lib.sz_async_stats(pointer, ct.byref(stored), ct.byref(raw))
            if not status:
                counter.update(packet_bytes=stored.value, raw_packets=raw.value)
                observed[0] = True
        if native.lib.sz_async_destroy(pointer):
            _QUARANTINED.append((native, pointer, resources))


class _PacketPool:
    """A byte-bounded cache of completed device packets and pinned headers.

    Only an explicit successful DMA completion returns a pair. Native Host
    snapshots retain parsed CPU metadata, not the original encode packet.
    Active requests and failed/quarantined requests are never pool entries.
    """
    def __init__(self, capacity, stats):
        self.capacity, self.stats = capacity, stats
        self.cached_bytes = 0
        self._free = {}

    def take(self, device, size):
        key = (str(device), size)
        entries = self._free.get(key)
        if not entries:
            self.stats['packet_pool_misses'] += 1
            return None
        packet, header = entries.pop()
        if not entries:
            del self._free[key]
        self.cached_bytes -= packet.numel() + header.numel()
        self.stats['packet_pool_hits'] += 1
        self.stats['packet_pool_cached_bytes'] = self.cached_bytes
        return packet, header

    def release(self, resources):
        packet, header = resources[:2]
        size = packet.numel() + header.numel()
        if self.cached_bytes + size > self.capacity:
            return
        self._free.setdefault((str(packet.device), packet.numel()), []).append((packet, header))
        self.cached_bytes += size
        self.stats['packet_pool_cached_bytes'] = self.cached_bytes
        self.stats['packet_pool_high_water_bytes'] = max(self.stats['packet_pool_high_water_bytes'], self.cached_bytes)

    def clear(self):
        self._free.clear()
        self.cached_bytes = 0
        self.stats['packet_pool_cached_bytes'] = 0


class _AsyncOwner:
    """CPU metadata plus a completion fence, retaining all private DMA buffers."""

    def __init__(self, native, resources, counter, on_retire=None):
        self.native, self.pointer = native, ct.c_void_p()
        self._counter, self._observed = counter, [False]
        self._lock = threading.Lock()
        self._stored = None
        self._retired = False
        self._on_retire = on_retire
        self._finalizer = weakref.finalize(self, _destroy_async, native, self.pointer, resources, counter, self._observed)

    def stored_bytes(self):
        with self._lock:
            if self._stored is None:
                stored, raw = ct.c_uint32(), ct.c_uint32()
                self.native.check(self.native.lib.sz_async_stats(self.pointer, ct.byref(stored), ct.byref(raw)))
                self._stored = stored.value
                if not self._observed[0]:
                    self._counter.update(packet_bytes=stored.value, raw_packets=raw.value)
                    self._observed[0] = True
            return self._stored

    def query(self):
        ready = ct.c_uint32()
        self.native.check(self.native.lib.sz_async_query(self.pointer, ct.byref(ready)))
        if ready.value:
            self.stored_bytes()
            self.retire_buffers()
        return bool(ready.value)

    def synchronize(self):
        self.native.check(self.native.lib.sz_async_sync(self.pointer))
        self.stored_bytes()
        self.retire_buffers()

    def retire_buffers(self):
        """Call only after DMA completion; preserve CPU metadata for later H2D."""
        if self._retired:
            return
        old = self._finalizer.detach()
        self._finalizer = weakref.finalize(
            self, _destroy_async, self.native, self.pointer, (), self._counter, self._observed,
        )
        self._retired = True
        if old is not None and self._on_retire is not None:
            # detach() returns (owner, callback, callback_args, callback_kwargs).
            # callback_args[2] is the original, still-live DMA resource tuple.
            self._on_retire(old[2][2])
            self._on_retire = None
        del old

    def wait(self, stream):
        with SplitZipCodec._device_context(stream.device), SplitZipCodec._stream_context(stream, stream.device):
            raw_stream = torch.npu.current_stream(stream.device).npu_stream
            self.native.check(self.native.lib.sz_async_wait_stream(self.pointer, raw_stream))
        self.stored_bytes()
        # Enqueuing a consumer wait does not itself prove DMA completion.
        # Retire only after a successful completion query; the last encodes
        # otherwise retain their packets throughout backward until step exit.
        if not self._retired:
            self.query()


@dataclass(frozen=True)
class SplitZipChunk:
    offset: int
    original_bytes: int
    padded_bytes: int
    header: bytes


@dataclass(frozen=True)
class SplitZipPacket:
    # Whole-storage execution is the one-chunk case. Future pipelining can
    # extend execution without changing the framework's Host snapshot contract.
    chunks: tuple[SplitZipChunk, ...]
    native_packet: object = None


class _PackedResult(ct.Structure):
    _fields_ = [("packet_pointer", ct.c_void_p), ("header_pointer", ct.c_void_p),
                ("stored", ct.c_uint32), ("raw", ct.c_uint32)]


class _NativeOwner:
    """Own CPU metadata only. No device access occurs when this object dies."""

    def __init__(self, native, pointer, destroy):
        self.native, self.pointer = native, pointer
        self._finalizer = weakref.finalize(self, getattr(native.lib, destroy), pointer)


class _Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        p, n, u = ct.c_void_p, ct.c_size_t, ct.c_uint32
        signatures = {
            "sz_book": [p, n, p, ct.POINTER(u)],
            "sz_encode": [ct.c_uint64, p, u, p, n, p, n, p, p, p, p],
            "sz_info": [ct.c_uint64, p, n, p, p, ct.POINTER(u), ct.POINTER(u)],
            "sz_decode": [ct.c_uint64, p, p, n, p, n, p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.lib.sz_status.argtypes = [ct.c_int]
        self.lib.sz_status.restype = ct.c_char_p
        prepared = {
            "sz_context_create": [ct.c_uint64, p, u, p, p, p, p, ct.POINTER(p)],
            "sz_encode_prepared": [p, p, n, p, n, p, ct.POINTER(_PackedResult)],
            "sz_decode_prepared": [p, p, n, p, n, p],
            "sz_context_destroy": [p], "sz_packet_destroy": [p],
        }
        self.prepared = all(hasattr(self.lib, name) for name in prepared)
        if self.prepared:
            for name, args in prepared.items():
                fn = getattr(self.lib, name)
                fn.argtypes = args
                fn.restype = None if name.endswith("destroy") else ct.c_int
        asynchronous = {
            "sz_async_engine_create": [p, ct.POINTER(p)],
            "sz_async_engine_destroy": [p], "sz_async_engine_flush": [p], "sz_async_engine_sync": [p],
            "sz_async_submit": [p, p, p, n, p, n, p, p, n, p, ct.POINTER(p)],
            "sz_async_wait_stream": [p, p], "sz_async_query": [p, ct.POINTER(u)],
            "sz_async_stats": [p, ct.POINTER(u), ct.POINTER(u)], "sz_async_sync": [p],
            "sz_async_decode": [p, p, n, p, n, p], "sz_async_destroy": [p],
        }
        self.asynchronous = self.prepared and all(hasattr(self.lib, name) for name in asynchronous)
        if self.asynchronous:
            for name, args in asynchronous.items():
                fn = getattr(self.lib, name)
                fn.argtypes = args
                fn.restype = None if name == "sz_async_engine_destroy" else ct.c_int

    def check(self, status):
        if status:
            raise RuntimeError("SplitZip: " + self.lib.sz_status(status).decode())


class SplitZipCodec:
    """Opt-in native codec. By default every supported request is selected.

    ``nodes`` selects exact ``(op_id, phase)`` pairs; ``selector`` instead
    accepts a TransferContext, allowing later algorithmic decisions. Neither
    selection nor initialization imports HyperOffload or installs tensor hooks.
    Codebooks are fixed for the instance; a CPU BF16 calibration sample may be
    supplied. The deterministic default is synthetic N(0,1), not model tuning.
    Instances are owned by one session; writable scratch is private per stream.
    """

    def __init__(self, library_path=None, *, nodes=None, selector=None, sample=None,
                 use_native_context=True, async_transport=True, max_inflight=8, packet_pool_bytes=0):
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        if isinstance(packet_pool_bytes, bool) or not isinstance(packet_pool_bytes, int) or packet_pool_bytes < 0:
            raise ValueError("packet_pool_bytes must be nonnegative integer bytes")
        if nodes is not None and selector is not None:
            raise ValueError("choose either nodes or selector")
        self.nodes = None if nodes is None else frozenset(nodes)
        if self.nodes is not None and any(
            not isinstance(op, int) or isinstance(op, bool) or op < 0 or not isinstance(phase, str) or not phase
            for op, phase in self.nodes
        ):
            raise ValueError("nodes must contain nonnegative op IDs and nonempty phase names")
        if selector is not None and not callable(selector):
            raise TypeError("selector must be callable")
        if sample is not None and (
            sample.device.type != "cpu" or sample.dtype != torch.bfloat16 or not sample.numel()
        ):
            raise ValueError("sample must be a nonempty CPU BF16 tensor")
        self.library_path = library_path or os.environ.get("MINDSPEED_SPLITZIP_LIBRARY")
        self.selector = selector
        self._sample = sample.detach().contiguous().clone() if sample is not None else None
        self._native = None
        self._book = None
        self._states = {}
        self.stats = Counter()
        self.packet_pool_bytes = packet_pool_bytes
        self._packet_pool = _PacketPool(packet_pool_bytes, self.stats)
        self.node_stats = {}
        self.use_native_context = use_native_context
        self.async_transport = async_transport
        self.max_inflight = max_inflight

    def accepts(self, context: TransferContext) -> bool:
        reason = None
        if context.device.type != "npu":
            reason = "device"
        elif context.dtype != torch.bfloat16:
            reason = "dtype"
        elif not 0 < context.original_bytes <= 512 * 1024**2 or context.original_bytes % 2:
            reason = "size"
        elif self.nodes is not None and (
            context.node is None or (context.node.op_id, context.node.phase) not in self.nodes
        ):
            reason = "node"
        elif self.selector is not None and not self.selector(context):
            reason = "selector"
        if reason:
            self.stats["skip_" + reason] += 1
            return False
        if context.node is not None:
            key = f"{context.node.op_id}:{context.node.phase}:{context.node.name}"
            row = self.node_stats.setdefault(key, Counter())
            row["selected_requests"] += 1
            row["original_bytes"] += context.original_bytes
        return True

    def _initialize(self):
        if self._native is not None:
            return
        if not self.library_path or not Path(self.library_path).is_file():
            raise RuntimeError("Build compression/native/build.sh and pass library_path or MINDSPEED_SPLITZIP_LIBRARY")
        native = _Native(self.library_path)
        sample = self._sample
        if sample is None:
            sample = torch.randn(524288, generator=torch.Generator().manual_seed(1729)).to(torch.bfloat16)
        book = torch.empty(544, dtype=torch.uint8)
        base = ct.c_uint32()
        native.check(native.lib.sz_book(sample.data_ptr(), sample.numel(), book.data_ptr(), ct.byref(base)))
        self._native, self._book, self._base = native, book, base.value
        self._sample = None

    def prepare(self, device, stream=None):
        """Load the library and upload the fixed codebook outside model capture."""
        self._initialize()
        stream = stream or torch.npu.current_stream(device)
        with torch.npu.device(device), self._stream_context(stream, device):
            key = (stream.device.index, stream.stream_id)
            if key not in self._states:
                table = self._book.to(device=device)
                self._states[key] = {
                    "table": table,
                    "sparse": torch.empty(655392, dtype=torch.uint8, device=device),
                    "counts": torch.empty(1536, dtype=torch.uint8, device=device),
                    "header": torch.empty(2048, dtype=torch.uint8, pin_memory=True),
                    "raw_stream": torch.npu.current_stream(device).npu_stream,
                    "contexts": {},
                }
        return self._states[key]

    def host_capacity(self, size):
        """Optional transport extension. None keeps the regular encode/copy API."""
        self._initialize()
        if not self.async_transport or not self.use_native_context or not self._native.asynchronous:
            return None
        return ((size+16383)//16384)*16384+2048

    def _prepared_context(self, state, size):
        context = state["contexts"].get(size)
        if context is None:
            pointer = ct.c_void_p()
            self._native.check(self._native.lib.sz_context_create(
                size//2, self._book.data_ptr(), self._base,
                state["sparse"].data_ptr(), state["counts"].data_ptr(), state["table"].data_ptr(),
                state["header"].data_ptr(), ct.byref(pointer),
            ))
            context = _NativeOwner(self._native, pointer.value, "sz_context_destroy")
            if len(state["contexts"]) >= 128:
                del state["contexts"][next(iter(state["contexts"]))]
            state["contexts"][size] = context
        return context

    def encode_to_host(self, source, host, stream):
        """Queue a whole tensor; a native progress worker submits actual-size D2H.

        The worker has its own DMA stream and never executes Python or model
        operators. Each request owns a private packet and pinned header.
        """
        size = source.numel()
        padded = ((size+16383)//16384)*16384
        if (source.dtype != torch.uint8 or source.device.type != "npu" or source.device != stream.device
                or not source.is_contiguous() or not 0 < size <= 512*1024**2 or size % 2):
            raise ValueError("encode_to_host requires supported contiguous NPU bytes")
        if (host.device.type != "cpu" or host.dtype != torch.uint8 or not host.is_pinned()
                or not host.is_contiguous() or host.numel() < padded+2048):
            raise ValueError("encode_to_host requires sufficient pinned CPU bytes")
        with self._device_context(source.device), self._stream_context(stream, source.device):
            state = self._states.get((stream.device.index, stream.stream_id))
            if state is None:
                state = self.prepare(source.device, stream)
            context = self._prepared_context(state, size)
            if "engine" not in state:
                dma = torch.Stream(device=source.device)
                with self._stream_context(dma, source.device):
                    dma_pointer = torch.npu.current_stream(source.device).npu_stream
                pointer = ct.c_void_p()
                self._native.check(self._native.lib.sz_async_engine_create(dma_pointer, ct.byref(pointer)))
                state["dma"] = dma
                state["engine"] = _NativeOwner(self._native, pointer.value, "sz_async_engine_destroy")
                state["inflight"] = deque()
            pending = state["inflight"]
            while pending and pending[0].query():
                pending.popleft().retire_buffers()
            if len(pending) >= self.max_inflight:
                pending[0].synchronize()
                pending.popleft().retire_buffers()
            cached = self._packet_pool.take(source.device, padded+2048) if self.packet_pool_bytes else None
            if cached is None:
                packet = torch.empty(padded+2048, dtype=torch.uint8, device=source.device)
                header = torch.empty(2048, dtype=torch.uint8, pin_memory=True)
            else:
                packet, header = cached
            packet.record_stream(stream)
            packet.record_stream(state["dma"])
            input_data = source
            if padded != size:
                input_data = self._scratch(state, "input", padded, source.device)
                input_data.record_stream(stream)
                input_data[:size].copy_(source, non_blocking=True)
            resources = (packet, header, host, state["sparse"], state["counts"], state["table"],
                         state["engine"], state["dma"], stream, input_data if input_data is not source else None)
            owner = _AsyncOwner(self._native, resources, self.stats,
                                self._packet_pool.release if self.packet_pool_bytes else None)
            raw_stream = torch.npu.current_stream(source.device).npu_stream
            status = self._native.lib.sz_async_submit(
                state["engine"].pointer, context.pointer, input_data.data_ptr(), input_data.numel(),
                packet.data_ptr(), packet.numel(), header.data_ptr(), host.data_ptr(), host.numel(),
                raw_stream, ct.byref(owner.pointer),
            )
            self._native.check(status)
            pending.append(owner)
        metadata = SplitZipPacket((SplitZipChunk(0,size,padded,b""),), owner)
        self.stats.update(encode_calls=1, input_bytes=size, async_encodes=1)
        return HostSubmission(metadata, owner)

    def host_view(self, host, metadata):
        if isinstance(metadata.native_packet, _AsyncOwner):
            return host[:metadata.native_packet.stored_bytes()]
        return host

    def flush_submissions(self):
        if self._native is not None and self._native.asynchronous:
            for state in self._states.values():
                if "engine" in state:
                    self._native.check(self._native.lib.sz_async_engine_flush(state["engine"].pointer))
                    while state["inflight"] and state["inflight"][0].query():
                        state["inflight"].popleft().retire_buffers()

    def wait_for_transfers(self, consumer):
        self.flush_submissions()
        for state in self._states.values():
            if "dma" in state:
                consumer.wait_stream(state["dma"])

    def synchronize(self):
        if self._native is not None and self._native.asynchronous:
            for state in self._states.values():
                if "engine" in state:
                    self._native.check(self._native.lib.sz_async_engine_sync(state["engine"].pointer))
                    while state["inflight"]:
                        owner = state["inflight"].popleft()
                        owner.synchronize()
                        owner.retire_buffers()

    def close(self):
        """Finish background transfers and retire idle per-stream workspaces."""
        self.synchronize()
        self._states.clear()
        self._packet_pool.clear()

    @staticmethod
    def _device_context(device):
        return nullcontext() if torch.npu.current_device() == device.index else torch.npu.device(device)

    def _encode_prepared(self, source, stream, size, padded):
        with self._device_context(source.device), self._stream_context(stream, source.device):
            state = self._states.get((stream.device.index, stream.stream_id))
            if state is None:
                state = self.prepare(source.device, stream)
            context = self._prepared_context(state, size)
            input_data = source
            if padded != size:
                with self._stream_context(stream, source.device):
                    input_data = self._scratch(state, "input", padded, source.device)
                    input_data.record_stream(stream)
                    input_data[:size].copy_(source, non_blocking=True)
            packet = self._scratch(state, "packet", padded+2048, source.device)
            if state.get("recorded_packet") is not packet:
                packet.record_stream(stream)
                state["recorded_packet"] = packet
            result = _PackedResult()
            # npu_stream is more than a pointer lookup: torch_npu drains its
            # asynchronous Host task queue before exposing the native stream.
            # A cached pointer can let direct ACL kernels overtake queued
            # producer waits, staging copies, or H2D transfers.
            raw_stream = torch.npu.current_stream(source.device).npu_stream
            self._native.check(self._native.lib.sz_encode_prepared(
                context.pointer, input_data.data_ptr(), input_data.numel(),
                packet.data_ptr(), packet.numel(), raw_stream, ct.byref(result),
            ))
            owner = _NativeOwner(self._native, result.packet_pointer, "sz_packet_destroy")
            header = ct.string_at(result.header_pointer, 2048)
            metadata = SplitZipPacket((SplitZipChunk(0,size,padded,header),), owner)
            self.stats.update(encode_calls=1, input_bytes=size, packet_bytes=result.stored,
                              raw_packets=result.raw, prepared_encodes=1)
            return EncodedBuffer(packet[:result.stored], metadata)

    @staticmethod
    def _stream_context(stream, device):
        # torch.Stream forbids entering the same context object twice. A codec
        # is also called from inside its owner's transfer-stream context.
        current = torch.npu.current_stream(device)
        return nullcontext() if current.stream_id == stream.stream_id else stream

    @staticmethod
    def _scratch(state, name, size, device):
        data = state.get(name)
        if data is None or data.numel() < size:
            data = torch.empty(size, dtype=torch.uint8, device=device)
            state[name] = data
        return data

    def encode(self, source, stream):
        size = source.numel()
        if source.dtype != torch.uint8 or source.device.type != "npu" or not source.is_contiguous():
            raise ValueError("encode expects contiguous NPU bytes")
        if not 0 < size <= 512 * 1024**2 or size % 2 or source.data_ptr() % 32:
            raise ValueError("unsupported SplitZip input size/alignment")
        padded = ((size + 16383) // 16384) * 16384
        if stream.device != source.device:
            raise ValueError("SplitZip source and stream devices differ")
        self._initialize()
        if self.use_native_context and self._native.prepared:
            return self._encode_prepared(source, stream, size, padded)
        with torch.npu.device(source.device), self._stream_context(stream, source.device):
            state = self.prepare(source.device, stream)
            # The release zeros the padded tail. Never write past live storage.
            input_data = source
            if padded != size:
                input_data = self._scratch(state, "input", padded, source.device)
                input_data[:size].copy_(source, non_blocking=True)
            packet = self._scratch(state, "packet", padded + 2048, source.device)
            native_stream = torch.npu.current_stream(source.device).npu_stream
            native = self._native
            native.check(native.lib.sz_encode(
                size // 2, self._book.data_ptr(), self._base,
                input_data.data_ptr(), input_data.numel(), packet.data_ptr(), packet.numel(),
                state["sparse"].data_ptr(), state["counts"].data_ptr(), state["table"].data_ptr(), native_stream,
            ))
            stored, raw = ct.c_uint32(), ct.c_uint32()
            native.check(native.lib.sz_info(
                size // 2, packet.data_ptr(), packet.numel(), state["header"].data_ptr(), native_stream,
                ct.byref(stored), ct.byref(raw),
            ))
            header = bytes(state["header"].numpy())
        self.stats.update(encode_calls=1, input_bytes=size, packet_bytes=stored.value, raw_packets=raw.value)
        metadata = SplitZipPacket((SplitZipChunk(0, size, padded, header),))
        return EncodedBuffer(packet[:stored.value], metadata)

    def _decode_prepared(self, packet, metadata, stream):
        chunk = metadata.chunks[0]
        native_packet = metadata.native_packet
        with self._device_context(packet.device):
            output = torch.empty(chunk.padded_bytes, dtype=torch.uint8, device=packet.device)
            output.record_stream(stream)
            # The caller may restore on a stream different from the encoder.
            with self._stream_context(stream, packet.device):
                raw_stream = torch.npu.current_stream(packet.device).npu_stream
            decode = (native_packet.native.lib.sz_async_decode if isinstance(native_packet, _AsyncOwner)
                      else native_packet.native.lib.sz_decode_prepared)
            native_packet.native.check(decode(
                native_packet.pointer, packet.data_ptr(), packet.numel(),
                output.data_ptr(), output.numel(), raw_stream,
            ))
            if chunk.padded_bytes != chunk.original_bytes:
                with self._stream_context(stream, packet.device):
                    exact = torch.empty(chunk.original_bytes, dtype=torch.uint8, device=packet.device)
                    exact.copy_(output[:chunk.original_bytes], non_blocking=True)
                    output = exact
        self.stats.update(decode_calls=1, prepared_decodes=1)
        return output

    def decode(self, packet, metadata, stream):
        if not isinstance(metadata, SplitZipPacket) or len(metadata.chunks) != 1:
            raise ValueError("phase one requires one SplitZip chunk")
        if stream.device != packet.device:
            raise ValueError("SplitZip packet and stream devices differ")
        if metadata.native_packet is not None:
            return self._decode_prepared(packet, metadata, stream)
        chunk = metadata.chunks[0]
        self._initialize()
        header = ct.create_string_buffer(chunk.header, len(chunk.header))
        if stream.device != packet.device:
            raise ValueError("SplitZip packet and stream devices differ")
        with torch.npu.device(packet.device), self._stream_context(stream, packet.device):
            output = torch.empty(chunk.padded_bytes, dtype=torch.uint8, device=packet.device)
            native_stream = torch.npu.current_stream(packet.device).npu_stream
            self._native.check(self._native.lib.sz_decode(
                chunk.original_bytes // 2, ct.addressof(header), packet.data_ptr(), packet.numel(),
                output.data_ptr(), output.numel(), native_stream,
            ))
            if chunk.padded_bytes != chunk.original_bytes:
                # Do not expose padded backing storage through ordinary views:
                # the model/replay contract includes the original storage size.
                exact = torch.empty(chunk.original_bytes, dtype=torch.uint8, device=packet.device)
                exact.copy_(output[:chunk.original_bytes], non_blocking=True)
                output = exact
        self.stats["decode_calls"] += 1
        return output
