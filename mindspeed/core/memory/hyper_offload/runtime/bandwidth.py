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
import torch

from mindspeed.core.memory.hyper_offload.runtime.timer import DeviceTimer

logger = logging.getLogger(__name__)


def profile_transfer_bandwidth() -> tuple[float, float]:
    """Profile host-to-device and device-to-host transfer bandwidth.

    Returns:
        A tuple of (d2h_bandwidth_gbps, h2d_bandwidth_gbps).

    """
    if not torch.accelerator.is_available():
        raise RuntimeError(
            "Cannot profile transfer bandwidth: no accelerator available"
        )

    device = torch.accelerator.current_accelerator()
    size = 16 * 1024 * 1024
    try:
        src = torch.empty(size, dtype=torch.uint8, device=device)
        host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    except (torch.OutOfMemoryError, MemoryError):
        logger.warning("Bandwidth probe allocation failed; using conservative 1 GiB/s estimates")
        return 1.0, 1.0
    timer = DeviceTimer()

    # D2H bandwidth
    timer.start()
    host.copy_(src, non_blocking=True)
    d2h_ms = timer.stop()

    # H2D bandwidth
    timer.start()
    src.copy_(host, non_blocking=True)
    h2d_ms = timer.stop()
    def rate(elapsed_ms):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            logger.warning("Invalid bandwidth timing %r; using conservative 1 GiB/s", elapsed_ms)
            return 1.0
        return max(size / (elapsed_ms / 1000.0) / 1024**3, 1.0)

    return rate(d2h_ms), rate(h2d_ms)
