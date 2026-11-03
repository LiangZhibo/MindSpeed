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
"""Hardware bandwidth profiling utilities."""

import logging
import math
from statistics import median
import time
import torch

from mindspeed.core.memory.hyper_offload.runtime.timer import DeviceTimer

logger = logging.getLogger(__name__)


def profile_transfer_bandwidth(*, diagnostics: dict | None = None, size_bytes: int = 16*1024**2) -> tuple[float, float]:
    """Profile steady host/device bandwidth with a bounded median estimate.

    One roundtrip warms the existing buffers and copy path, then five paired
    samples limit sensitivity to a single cold or delayed copy. No model data,
    identity, or RNG is used. This function runs during trace warmup only.
    An optional diagnostics dictionary receives the raw sample timings.

    Returns:
        A tuple of (d2h_bandwidth_gbps, h2d_bandwidth_gbps).

    """
    if not torch.accelerator.is_available():
        raise RuntimeError(
            "Cannot profile transfer bandwidth: no accelerator available"
        )

    started = time.perf_counter()
    device = torch.accelerator.current_accelerator()
    if isinstance(size_bytes,bool) or not isinstance(size_bytes,int) or size_bytes<1:
        raise ValueError("bandwidth probe requires positive integer bytes")
    size = size_bytes
    try:
        src = torch.empty(size, dtype=torch.uint8, device=device)
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    except (torch.OutOfMemoryError, MemoryError):
        logger.warning("Bandwidth probe allocation failed; using conservative 1 GiB/s estimates")
        if diagnostics is not None:
            diagnostics.update(status='allocation_fallback', bytes=size,
                               elapsed_seconds=time.perf_counter()-started)
        return 1.0, 1.0
    timer = DeviceTimer()

    def roundtrip():
        timer.start()
        host.copy_(src, non_blocking=True)
        outbound = timer.stop()
        timer.start()
        src.copy_(host, non_blocking=True)
        inbound = timer.stop()
        return outbound, inbound

    warmup = roundtrip()
    samples = [roundtrip() for _ in range(5)]

    def rate(timings):
        if any(not math.isfinite(value) or value <= 0 for value in timings):
            logger.warning("Invalid bandwidth timings %r; using conservative 1 GiB/s", timings)
            return 1.0
        elapsed_ms = median(timings)
        return max(size / (elapsed_ms / 1000.0) / 1024**3, 1.0)

    outbound, inbound = tuple(zip(*samples))
    rates = rate(outbound), rate(inbound)
    if diagnostics is not None:
        diagnostics.update(status='measured', bytes=size, warmup_ms=warmup,
                           d2h_samples_ms=outbound, h2d_samples_ms=inbound,
                           d2h_bandwidth_gbps=rates[0], h2d_bandwidth_gbps=rates[1],
                           elapsed_seconds=time.perf_counter()-started)
    return rates
