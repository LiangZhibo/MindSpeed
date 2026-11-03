"""Real SplitZip tests require an explicitly built native library on Ascend."""
import os
import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from mindspeed.core.memory.compression import SplitZipCodec
from mindspeed.core.memory.hyper_offload import OffloadConfig, OffloadSession
from mindspeed.core.memory.hyper_offload.runtime.residency import ResidencyManager


@lru_cache(maxsize=1)
def _trial_module():
    # Other dependencies also export a top-level tools package.
    path = Path(__file__).resolve().parents[3] / "tools/hyper_offload/splitzip_basic_trial.py"
    spec = importlib.util.spec_from_file_location("hyperoffload_splitzip_trial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def library():
    value = os.environ.get("MINDSPEED_SPLITZIP_LIBRARY")
    if not value or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("set MINDSPEED_SPLITZIP_LIBRARY on an Ascend NPU host")
    return value


def test_native_packet_and_lifecycle(library):
    result = _trial_module().codec_checks(library, torch.accelerator.current_accelerator())
    assert len(result["roundtrips"]) == 6
    assert result["stats"]["encode_calls"] > 0
    assert result["stats"]["decode_calls"] > 0


def test_native_encode_failure_is_transactional(library):
    from unittest.mock import patch
    codec = SplitZipCodec(library)
    manager = ResidencyManager(64 * 1024**2, codec=codec)
    tensor = torch.ones(8193, dtype=torch.bfloat16, device=torch.accelerator.current_accelerator())
    buffer = manager.bind(1, tensor)
    method = "encode_to_host" if codec.host_capacity(tensor.untyped_storage().nbytes()) is not None else "encode"
    encode = getattr(codec, method)

    def fail_after_enqueue(*args):
        encode(*args)
        raise RuntimeError("injected post-encode failure")

    with patch.object(codec, method, fail_after_enqueue):
        with pytest.raises(RuntimeError, match="post-encode failure"):
            manager.copy_d2h(1)
    assert buffer.host_buffer is None and buffer.host_encoding is None
    assert not manager._host_pool._leased
    manager.copy_d2h(1)
    manager.release_device(1)
    manager.copy_h2d(1)
    manager.clear_runtime()


def test_native_session_abort_can_retry(library):
    codec = SplitZipCodec(library)
    session = OffloadSession(OffloadConfig(codec=codec, max_resident_activation_mb=3))
    device = torch.accelerator.current_accelerator()

    def forward():
        x = torch.randn(1024, 256, dtype=torch.bfloat16, device=device, requires_grad=True)
        return x, x.sin().cos().tanh().sin().float().square().mean()

    with pytest.raises(ValueError, match="abort"):
        with session:
            x, loss = forward()
            raise ValueError("abort")
    del x, loss
    with session:
        x, loss = forward()
        loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("prepared,asynchronous", [(False, False), (True, False), (True, True)])
def test_native_repeated_updates_and_transfers(library, prepared, asynchronous):
    """Do not rely on incidental Python delay to order the NPU Host task queue."""
    trial = _trial_module()
    from unittest.mock import patch
    def factory(path, **kw):
        return SplitZipCodec(path, use_native_context=prepared, async_transport=asynchronous, **kw)
    with patch.object(trial, "SplitZipCodec", factory):
        for _ in range(3):
            trial.codec_checks(library, torch.accelerator.current_accelerator())


def test_native_snapshot_survives_retired_transport_buffers(library):
    codec = SplitZipCodec(library, max_inflight=1)
    manager = ResidencyManager(64*1024**2, codec=codec)
    tensor = torch.randn(8195, dtype=torch.bfloat16, device=torch.accelerator.current_accelerator())
    buffer = manager.bind(1, tensor)
    manager.copy_d2h(1)
    manager.release_device(1)
    codec.close()
    assert not codec._states
    manager.copy_h2d(1)
    actual = torch.empty(0, dtype=torch.bfloat16, device=tensor.device).set_(buffer.device_storage(), 0, tensor.shape)
    torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    manager.clear_runtime()
    codec.close()


def test_completed_encode_packet_does_not_accumulate_with_restored_storage(library):
    """A completed packet must not remain allocated through backward/H2D."""
    device = torch.accelerator.current_accelerator()
    nbytes = 32 * 1024**2
    tensor = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=device)
    codec = SplitZipCodec(library)
    manager = ResidencyManager(128 * 1024**2, codec=codec)
    buffer = manager.bind(1, tensor)
    manager.copy_d2h(1)
    # Wait for submission before synchronizing the device: the native worker
    # may not have enqueued payload DMA when encode_to_host returns.
    buffer.host_encoding.metadata.native_packet.stored_bytes()
    torch.npu.synchronize()
    manager.release_device(1)
    torch.npu.empty_cache()
    before_restore = torch.npu.memory_allocated(device)
    manager.copy_h2d(1)
    torch.npu.synchronize()
    torch.npu.empty_cache()
    after_restore = torch.npu.memory_allocated(device)
    # Restoring32MiB should replace the completed encode temporary instead of
    # adding another32MiB live allocation. Leave room for allocator bookkeeping.
    assert after_restore - before_restore < nbytes // 2, (before_restore, after_restore, nbytes)
    actual = torch.empty(0, dtype=torch.bfloat16, device=device).set_(
        buffer.device_storage(), 0, tensor.shape,
    )
    torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    manager.clear_runtime()
    codec.close()


@pytest.mark.parametrize("prepared", [False, True])
def test_native_explicit_stream_outside_current_context(library, prepared):
    """Native calls must flush queued producer waits and tail staging copies."""
    codec = SplitZipCodec(library, use_native_context=prepared, async_transport=False)
    device = torch.accelerator.current_accelerator()
    producer = torch.accelerator.current_stream()
    transfer = torch.Stream(device=device)
    codec.prepare(device, transfer)
    for value in (37, 62, 91):
        source = torch.empty(16386, dtype=torch.uint8, device=device)
        source.fill_(value)
        transfer.wait_stream(producer)
        encoded = codec.encode(source, transfer)
        restored = codec.decode(encoded.data, encoded.metadata, transfer)
        producer.wait_stream(transfer)
        assert torch.equal(restored.cpu(), torch.full((16386,), value, dtype=torch.uint8))
    codec.close()


def test_reused_encode_packet_cannot_change_an_older_host_snapshot(library):
    device = torch.accelerator.current_accelerator()
    codec = SplitZipCodec(library, max_inflight=1, packet_pool_bytes=16*1024**2)
    manager = ResidencyManager(64*1024**2, codec=codec)
    originals = [torch.full((4*1024**2,), value, dtype=torch.bfloat16, device=device) for value in (3.,7.)]
    buffers = []
    for sid, tensor in enumerate(originals,1):
        buffers.append(manager.bind(sid,tensor))
        manager.copy_d2h(sid)
        manager.release_device(sid)
        codec.synchronize()
    assert codec.stats['packet_pool_hits'] >= 1
    assert codec._packet_pool.cached_bytes <= codec.packet_pool_bytes
    for sid, (buffer, expected) in enumerate(zip(buffers, originals),1):
        manager.copy_h2d(sid)
        actual = torch.empty(0,dtype=torch.bfloat16,device=device).set_(buffer.device_storage(),0,expected.shape)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    manager.clear_runtime()
    codec.close()
    assert codec._packet_pool.cached_bytes == 0
