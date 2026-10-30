"""Device-memory admission for ``gpu_image_prep`` (host only, so the offline suite covers it).

A photo's decoded frame (and, for a rotated photo, its re-oriented copy) is the one large
allocation of a request. Every request reserves those bytes from a :class:`ByteBudget` before it
decodes and gives them back when its frame is dropped. Memory pressure therefore makes requests
wait in line (first come, first served, so a large photo is not starved by small ones) instead of
failing a CUDA or nvJPEG allocation in the middle of an image. A wait that outlasts its limit, a
photo larger than the whole budget, and an allocation that fails anyway are answered with a
status (``gpu_memory``, ``over_budget``) and the API decodes that photo on the CPU, counted
(``cpu_gpu_memory``, ``cpu_over_budget``): never a silent fallback and never a partial output.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from prep_geometry import JpegHeader

_CUDA_ERROR_MEMORY_ALLOCATION = 2
_NVJPEG_STATUS_ALLOCATOR_FAILURE = 5


class OverBudgetError(Exception):
    """A single request needs more bytes than the whole budget."""


def need_bytes(header: JpegHeader) -> int:
    """Bytes the request's frame holds in the shared arena: the HWC uint8 RGB decode, plus the
    re-oriented copy that exists next to it while an EXIF tag other than 1 is applied."""
    frame = header.width * header.height * 3
    return frame * (2 if header.orientation != 1 else 1)


class ByteBudget:
    """``acquire(n)`` blocks until ``n`` bytes are free and every earlier waiter was served."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.in_use = 0
        self._cond = threading.Condition()
        self._queue: deque[object] = deque()

    def acquire(self, n: int, timeout: float) -> bool:
        """True once ``n`` bytes are reserved, False if ``timeout`` seconds passed first."""
        if n > self.total:
            raise OverBudgetError(f'{n} bytes against a budget of {self.total}')
        deadline = time.monotonic() + timeout
        ticket = object()
        with self._cond:
            self._queue.append(ticket)
            try:
                while self._queue[0] is not ticket or self.in_use + n > self.total:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cond.wait(left)
                self.in_use += n
                return True
            finally:
                self._queue.remove(ticket)
                self._cond.notify_all()

    def release(self, n: int) -> None:
        with self._cond:
            self.in_use -= n
            self._cond.notify_all()


def is_memory_error(exc: BaseException) -> bool:
    """A device allocation failure from CuPy, the CUDA runtime or nvJPEG (matched by class name
    and status so this module needs none of those libraries)."""
    name = type(exc).__name__
    status = getattr(exc, 'status', None)
    if name == 'OutOfMemoryError':
        return True
    if name == 'CUDARuntimeError':
        return status == _CUDA_ERROR_MEMORY_ALLOCATION
    return name == 'NvJpegError' and status == _NVJPEG_STATUS_ALLOCATOR_FAILURE
