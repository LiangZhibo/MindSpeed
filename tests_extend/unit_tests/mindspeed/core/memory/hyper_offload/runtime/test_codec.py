"""Byte transport contracts on CPU; real SplitZip is tested separately on NPU."""
from dataclasses import replace
import zlib
from unittest.mock import patch

import pytest
import torch

from mindspeed.core.memory.compression import EncodedBuffer, SplitZipCodec, TransferContext, TransferNode
from mindspeed.core.memory.compression.transport import HostSubmission
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.execution.tensor import ShadowTensor
from mindspeed.core.memory.hyper_offload.execution.validation import signature
from mindspeed.core.memory.hyper_offload.execution.warmup.tracker import ActivationTracker
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager
from tests_extend.unit_tests.mindspeed.core.memory.hyper_offload.runtime.test_residency import _mock_accelerator_if_needed, _MockStream


class Event:
    def record(self, stream):
        self.stream = stream

    def wait(self, stream):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


class TestCodec:
    """Variable-length CPU test double, not a production compression fallback."""
    __test__ = False

    def accepts(self, context):
        self.context = context
        return True

    def encode(self, source, stream):
        data = zlib.compress(bytes(source.numpy()))
        return EncodedBuffer(torch.frombuffer(bytearray(data), dtype=torch.uint8), source.numel())

    def decode(self, packet, metadata, stream):
        data = zlib.decompress(bytes(packet.numpy()))
        assert len(data) == metadata
        return torch.frombuffer(bytearray(data), dtype=torch.uint8)


class TestAsyncCodec(TestCodec):
    __test__ = False

    def __init__(self):
        self.drained = self.waited = False

    def host_capacity(self, size):
        return size+128

    def encode_to_host(self, source, host, stream):
        encoded = self.encode(source, stream)
        host[:encoded.data.numel()].copy_(encoded.data)
        return HostSubmission((encoded.metadata, encoded.data.numel()), Event())

    def host_view(self, host, metadata):
        return host[:metadata[1]]

    def decode(self, packet, metadata, stream):
        return super().decode(packet, metadata[0], stream)

    def synchronize(self):
        self.drained = True

    def wait_for_transfers(self, consumer):
        self.waited = True


@pytest.fixture
def manager():
    with _mock_accelerator_if_needed(force=True), patch("torch.Event", Event), patch.object(
        ResidencyManager, "_get_copy_stream", return_value=_MockStream(),
    ):
        yield ResidencyManager(1024**2, codec=TestCodec())


def test_snapshot_decodes_with_original_codec_and_preserves_aliases(manager):
    tensor = torch.zeros(1024, dtype=torch.bfloat16)
    tensor[17] = -3
    buffer = manager.bind(9, tensor)
    shadow = ShadowTensor(tensor[3::2], buffer, 9)
    original_signature = signature((shadow,))
    manager.set_transfer_node(7, "post", "test_node")
    manager.copy_d2h(9)
    assert buffer.host_buffer.numel() < buffer.logical_nbytes
    assert manager._codec.context.node == TransferNode(7, "post", "test_node")
    manager.release_device(9)
    assert signature((shadow,)) == original_signature
    manager._codec = None  # Changing selection must not change snapshot decoding.
    manager.copy_h2d(9)
    torch.testing.assert_close(shadow.resolve(), tensor[3::2], rtol=0, atol=0)
    tracker = ActivationTracker()
    tracker._ensure_id(shadow)
    assert tracker.storage_sizes[9] == 2048
    assert buffer.host_reuse_event is buffer.device_event


def test_demand_restore_and_host_recycling_use_decode_completion(manager):
    tensor = torch.ones(1024, dtype=torch.bfloat16)
    buffer = manager.bind(1, tensor)
    manager.copy_d2h(1)
    manager.release_device(1)
    buffer.prepare_for_write()
    assert buffer.host_encoding is None and buffer.host_buffer is None
    assert buffer.device_event is not None
    assert any(event is buffer.device_event for entries in manager._host_pool._pending.values() for _, event in entries)
    restored = torch.empty(0, dtype=tensor.dtype).set_(buffer.device_storage(), 0, tensor.shape)
    torch.testing.assert_close(restored, tensor)


def test_rebind_invalidates_snapshot_and_reinterpreted_dtype(manager):
    tensor = torch.ones(1024, dtype=torch.bfloat16)
    buffer = manager.bind(1, tensor)
    manager.copy_d2h(1)
    manager.bind(1, tensor.view(torch.int16))
    assert buffer.storage_dtype is None
    other = torch.full_like(tensor, 7)
    manager.bind(1, other)
    assert buffer.host_buffer is None and buffer.host_encoding is None
    assert buffer.storage_dtype == torch.bfloat16


def test_failed_encode_releases_host_lease_and_can_retry(manager):
    manager.bind(1, torch.ones(1024, dtype=torch.bfloat16))
    with patch.object(manager._codec, "encode", side_effect=RuntimeError("encode failed")):
        with pytest.raises(RuntimeError, match="encode failed"):
            manager.copy_d2h(1)
    assert not manager._host_pool._leased
    assert manager._residency[1].host_encoding is None
    manager.copy_d2h(1)
    manager.release_device(1)
    with patch.object(manager._codec, "decode", side_effect=RuntimeError("decode failed")):
        with pytest.raises(RuntimeError, match="decode failed"):
            manager.copy_h2d(1)
    assert manager._residency[1].device_buffer is None
    manager.copy_h2d(1)


def test_clear_restores_live_output_and_retires_encoding(manager):
    tensor = torch.ones(1024, dtype=torch.bfloat16)
    buffer = manager.bind(1, tensor)
    shadow = ShadowTensor(tensor, buffer, 1)
    manager.copy_d2h(1)
    manager.release_device(1)
    manager.clear_runtime()
    assert not manager._residency
    assert buffer.host_encoding is None
    torch.testing.assert_close(shadow.resolve(), tensor)


def test_codec_is_injected_through_public_config():
    codec = TestCodec()
    session = OffloadSession(OffloadConfig(codec=codec))
    assert session.residency_manager._codec is codec


def test_public_config_rejects_invalid_codec():
    with pytest.raises(TypeError, match="codec must provide"):
        OffloadConfig(codec=object())


def test_deferred_host_capacity_is_not_transferred_as_payload(manager):
    codec = TestAsyncCodec()
    manager._codec = codec
    tensor = torch.ones(1024, dtype=torch.bfloat16)
    buffer = manager.bind(1, tensor)
    manager.copy_d2h(1)
    assert buffer.host_buffer.numel() == 2176
    assert buffer.host_encoding.metadata[1] < 2048
    manager.release_device(1)
    manager.copy_h2d(1)
    restored = torch.empty(0, dtype=torch.bfloat16).set_(buffer.device_storage(), 0, tensor.shape)
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
    manager.wait_for_transfers()
    manager.sync_all_transfers()
    assert codec.drained and codec.waited


def test_deferred_submission_failure_drains_before_host_reuse(manager):
    codec = TestAsyncCodec()
    manager._codec = codec
    manager.bind(1, torch.ones(1024, dtype=torch.bfloat16))
    original = codec.encode_to_host

    def fail(*args):
        original(*args)
        raise RuntimeError("after submission")

    with patch.object(codec, "encode_to_host", fail):
        with pytest.raises(RuntimeError, match="after submission"):
            manager.copy_d2h(1)
    assert codec.drained and not manager._host_pool._leased


def test_failed_deferred_drain_quarantines_host_allocation(manager):
    codec = TestAsyncCodec()
    manager._codec = codec
    manager.bind(1, torch.ones(1024, dtype=torch.bfloat16))
    with patch.object(codec, "encode_to_host", side_effect=RuntimeError("submit")), patch.object(
        codec, "synchronize", side_effect=RuntimeError("drain"),
    ):
        with pytest.raises(RuntimeError, match="submit"):
            manager.copy_d2h(1)
    assert len(manager._quarantined) == 1
    assert manager._host_pool._leased


def test_failed_deferred_wait_keeps_writer_fence_for_host_recycling(manager):
    manager._codec = TestAsyncCodec()
    buffer = manager.bind(1, torch.ones(1024, dtype=torch.bfloat16))
    manager.copy_d2h(1)
    host_storage = buffer.host_buffer.untyped_storage()._cdata

    class PendingWriter(Event):
        def wait(self, stream):
            raise RuntimeError("wait before DMA submission")

        def query(self):
            return False

    buffer.host_event = PendingWriter()
    manager.release_device(1)
    with pytest.raises(RuntimeError, match="before DMA submission"):
        manager.copy_h2d(1)
    assert not buffer.host_reuse_event.query()
    manager.release_host(1)
    replacement = manager._host_pool.acquire(2176)
    assert replacement.untyped_storage()._cdata != host_storage
    manager._host_pool.release(replacement)


def test_precise_node_selection_and_support_guards_without_loading_native_library():
    # torch.device('npu') requires torch_npu registration; device.type is the
    # only backend fact needed for pure policy evaluation.
    from types import SimpleNamespace
    context = TransferContext(TransferNode(12, "post"), 3, 1024, torch.bfloat16, SimpleNamespace(type="npu"))
    codec = SplitZipCodec(nodes={(12, "post")})
    assert codec.accepts(context)
    assert not codec.accepts(replace(context, node=TransferNode(12, "pre")))
    assert not codec.accepts(replace(context, dtype=torch.float32))
    assert not codec.accepts(replace(context, original_bytes=0))
    assert not codec.accepts(replace(context, original_bytes=512 * 1024**2 + 2))
    assert codec._native is None
    callback = SplitZipCodec(selector=lambda c: c.storage_id == 3)
    assert callback.accepts(context)
    assert not callback.accepts(replace(context, storage_id=4))
    hook = replace(context, node=TransferNode(12, "pack_ready"))
    assert SplitZipCodec(nodes={(12, "pack_ready")}).accepts(hook)


@pytest.mark.parametrize("kwargs", [dict(nodes={(-1, "post")}), dict(nodes={(1, "")}),
                                  dict(nodes=set(), selector=lambda c: True), dict(sample=torch.ones(2))])
def test_invalid_selection_or_calibration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitZipCodec(**kwargs)
