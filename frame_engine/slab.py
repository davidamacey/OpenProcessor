"""Offsets inside the engine's one registered region: first fit, coalescing, a bounded wait.

The CUDA shared-memory region is allocated and registered with Triton once (``transport``); every
model input of every photo in flight is a slice of it, addressed by offset in the request
(Triton's ``shared_memory_offset``). This allocator hands out those slices. A request that finds
no room waits at most its ``timeout`` and then gets :class:`SlabTimeoutError`, which the front answers
as overload (a retryable gRPC status): pressure is an answer, never an unbounded wait.
"""

from __future__ import annotations

import threading
import time


ALIGN = 256  # cudaMalloc alignment; every slice starts where a kernel may write vectors


class SlabTimeoutError(TimeoutError):
    """No slice of the asked size became free within the wait."""


class SlabTooLargeError(ValueError):
    """The request is larger than the whole region: it can never fit (a size setting, not load)."""


def aligned(nbytes: int) -> int:
    return (nbytes + ALIGN - 1) // ALIGN * ALIGN


class SlabAllocator:
    def __init__(self, size: int) -> None:
        if size <= 0 or size % ALIGN:
            raise ValueError(f'slab size {size} must be a positive multiple of {ALIGN}')
        self.size = size
        self._free: list[list[int]] = [[0, size]]  # [offset, length], sorted by offset
        self._cond = threading.Condition()
        self.in_use = 0
        self.peak = 0
        self.waits = 0

    def alloc(self, nbytes: int, timeout: float) -> int:
        """The offset of ``nbytes`` (rounded up to :data:`ALIGN`); waits at most ``timeout``.

        Raises:
            SlabTooLargeError: the request is larger than the whole region (it can never fit).
            SlabTimeoutError: no slice became free in time.
        """
        n = aligned(nbytes)
        if n > self.size:
            raise SlabTooLargeError(f'slab request {n} bytes exceeds the region of {self.size}')
        deadline = time.monotonic() + timeout
        with self._cond:
            waited = False
            while True:
                off = self._take(n)
                if off is not None:
                    return off
                if not waited:
                    self.waits += 1
                    waited = True
                left = deadline - time.monotonic()
                if left <= 0:
                    raise SlabTimeoutError(f'{n} bytes of the region not free within {timeout:g} s')
                self._cond.wait(left)

    def _take(self, n: int) -> int | None:
        for k, blk in enumerate(self._free):
            if blk[1] >= n:
                off = blk[0]
                blk[0] += n
                blk[1] -= n
                if blk[1] == 0:
                    del self._free[k]
                self.in_use += n
                self.peak = max(self.peak, self.in_use)
                return off
        return None

    def release(self, offset: int, nbytes: int) -> None:
        n = aligned(nbytes)
        with self._cond:
            self.in_use -= n
            self._free.append([offset, n])
            self._free.sort()
            merged: list[list[int]] = []
            for off, length in self._free:
                if merged and merged[-1][0] + merged[-1][1] == off:
                    merged[-1][1] += length
                else:
                    merged.append([off, length])
            self._free = merged
            self._cond.notify_all()
