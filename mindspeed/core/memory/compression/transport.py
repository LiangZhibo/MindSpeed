"""Framework-neutral metadata and codec boundary for storage transfers.

The caller owns source/Host/output lifetimes and stream dependencies. Codecs
enqueue native computation on the supplied stream, returning byte buffers.
An optional HostSubmission extension also owns deferred transport completion.
No logical tensor, autograd, schedule or HyperOffload type crosses this boundary.
"""
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class TransferNode:
    """Exact execution position; op_id is local to a validated execution trace."""

    op_id: int
    phase: str
    name: str = ""


@dataclass(frozen=True)
class TransferContext:
    node: TransferNode | None
    storage_id: int
    original_bytes: int
    dtype: torch.dtype | None
    device: torch.device


@dataclass(frozen=True)
class EncodedBuffer:
    data: torch.Tensor
    metadata: Any


@dataclass(frozen=True)
class HostSubmission:
    """Optional whole-storage transport result with a deferred completion fence.

    A completion implements query(), synchronize(), and wait(stream). Its wait
    must not expose an event before the actual DMA has been submitted.
    """

    metadata: Any
    completion: Any


class TransferCodec(Protocol):
    def accepts(self, context: TransferContext) -> bool: ...

    def encode(self, source: torch.Tensor, stream: torch.Stream) -> EncodedBuffer: ...

    def decode(self, packet: torch.Tensor, metadata: Any, stream: torch.Stream) -> torch.Tensor: ...


@dataclass(frozen=True)
class HostEncoding:
    """Immutable recipe attached to this Host snapshot, never reselected on H2D."""

    codec: TransferCodec
    metadata: Any
    original_bytes: int
