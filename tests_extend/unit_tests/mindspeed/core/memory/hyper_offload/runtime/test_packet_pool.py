"""CPU ownership tests for the optional completed-packet pool."""
import torch
import pytest

def test_packet_pool_capacity_counts_headers_and_preserves_device_size_keys():
    from collections import Counter
    from mindspeed.core.memory.compression.splitzip import _PacketPool
    stats = Counter()
    pool = _PacketPool(8192, stats)
    first = (torch.empty(4096, dtype=torch.uint8), torch.empty(2048, dtype=torch.uint8))
    second = (torch.empty(4096, dtype=torch.uint8), torch.empty(2048, dtype=torch.uint8))
    pool.release(first)
    pool.release(second)  # Cannot retain two pairs within the total byte cap.
    assert pool.cached_bytes == 6144
    assert pool.take(torch.device('cpu'), 2048) is None
    pair = pool.take(torch.device('cpu'), 4096)
    assert pair[0] is first[0] and pair[1] is first[1]
    assert pool.cached_bytes == 0
    pool.release(first)
    pool.clear()
    assert pool.cached_bytes == 0 and pool.take(torch.device('cpu'), 4096) is None


def test_packet_pool_recycles_only_after_successful_completion_query():
    import ctypes as ct
    from collections import Counter
    from types import SimpleNamespace
    from mindspeed.core.memory.compression.splitzip import _AsyncOwner, _PacketPool
    readiness = [0]
    def query(pointer, ready):
        ct.cast(ready, ct.POINTER(ct.c_uint32))[0] = readiness[0]
        return 0
    def packet_stats(pointer, stored, raw):
        ct.cast(stored, ct.POINTER(ct.c_uint32))[0] = 2000
        ct.cast(raw, ct.POINTER(ct.c_uint32))[0] = 0
        return 0
    def check(status):
        if status: raise RuntimeError('native failure')
    native = SimpleNamespace(check=check,lib=SimpleNamespace(sz_async_query=query,
        sz_async_stats=packet_stats,sz_async_destroy=lambda pointer: 0))
    stats = Counter();pool = _PacketPool(8192, stats)
    packet, header = torch.empty(4096,dtype=torch.uint8),torch.empty(2048,dtype=torch.uint8)
    owner = _AsyncOwner(native,(packet,header),stats,pool.release)
    owner.pointer.value = 123
    assert not owner.query() and pool.cached_bytes == 0
    readiness[0] = 1
    assert owner.query() and pool.cached_bytes == 6144
    assert owner.query() and pool.cached_bytes == 6144
    assert owner.stored_bytes() == 2000
    owner._finalizer()


@pytest.mark.parametrize('capacity', [-1, True, 0.5])
def test_packet_pool_rejects_invalid_capacity(capacity):
    from mindspeed.core.memory.compression import SplitZipCodec
    with pytest.raises(ValueError, match='packet_pool_bytes'):
        SplitZipCodec(packet_pool_bytes=capacity)
