"""Per-process thread pool and admission gate for per-image CPU work.

Decode, letterbox, PE preprocessing, blur scoring and crop JPEG encoding are
numpy, OpenCV and Pillow calls that release the GIL. Run directly in a
coroutine they hold the event-loop thread, so one worker cannot overlap two
images and a request waiting on the loop looks like a slow datastore call.
:func:`run_cpu` moves such a call to a small pool owned by the process, with
the caller's ``contextvars`` (the bound project, the request id) copied
across the hop.

The hop costs CPU of its own: every call that releases the GIL while the loop
thread wants it is a hand-over. Measured on 640 x 480 frames it added more
CPU than the work it moved (0.147 against 0.092 CPU-s per image), and on
12-24 MP photos it is what lets a worker use several cores. A call that
declares it touches less than ``OFFLOAD_MIN_BYTES`` of pixels therefore runs
in place: a few milliseconds on the loop are cheaper than the hop.

:class:`InflightGate` bounds how many images a process holds decoded at once.
Past the cap a request queues (first in, first out) before it decodes rather
than all requests decoding together, so more clients cannot lower the
throughput and the resident set stays bounded. The gate counts megabytes of decoded
pixels, not images: what it protects (memory and the host's cores) is spent per pixel, and a
count of images would hold back a GPU-bound stream of small frames to protect a stream of photos.
"""

from __future__ import annotations

import asyncio
import functools
import io
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from PIL import Image

from src.config.env import env_int
from src.config.project_context import run_in_executor_bound


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


DEFAULT_CPU_THREADS = 8
DEFAULT_INFLIGHT_MB = 1024
MIB = 1024 * 1024
# Below this much pixel data (about 1.3 MP of RGB) a call runs in place.
OFFLOAD_MIN_BYTES = 4 * 1024 * 1024

_offload_always: ContextVar[bool] = ContextVar('op_cpu_offload_always', default=False)

# One pool and one gate per process id, so a forked child builds its own.
_executors: dict[int, ThreadPoolExecutor] = {}
_gates: dict[int, InflightGate] = {}
_lock = threading.Lock()


def cpu_threads() -> int:
    """Threads in this process's CPU pool (``OP_CPU_POOL_THREADS``)."""
    return env_int('OP_CPU_POOL_THREADS', DEFAULT_CPU_THREADS, minimum=1)


def inflight_mb() -> int:
    """Decoded RGB megabytes one process holds in the ingest pipeline (``OP_INGEST_INFLIGHT_MB``)."""
    return env_int('OP_INGEST_INFLIGHT_MB', DEFAULT_INFLIGHT_MB, minimum=1)


def mib(nbytes: int | None) -> int:
    """``nbytes`` in whole MiB, at least 1 (an image whose size is unknown counts as 1)."""
    return max(1, -(-(nbytes or 0) // MIB))


def cpu_executor() -> ThreadPoolExecutor:
    """The process's CPU pool, created on first use."""
    pid = os.getpid()
    with _lock:
        if pid not in _executors:
            _executors[pid] = ThreadPoolExecutor(
                max_workers=cpu_threads(), thread_name_prefix='op-cpu-'
            )
        return _executors[pid]


def shutdown_cpu_executor() -> None:
    """Stop the pool at application shutdown (running calls finish first)."""
    with _lock:
        executor = _executors.pop(os.getpid(), None)
    if executor is not None:
        executor.shutdown(wait=True)


def encoded_nbytes(source: bytes | str) -> int | None:
    """RGB bytes of an encoded image (given as bytes or a path) from its header alone.

    ``None`` when the header cannot be read: the call is then offloaded, the safe side
    for work of unknown size.
    """
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as img:
            return img.width * img.height * 3
    except Exception:
        return None


@contextmanager
def offload_always() -> Iterator[None]:
    """Inside this block (and the tasks started in it) every :func:`run_cpu` call goes to the
    pool whatever its ``nbytes``.

    For a pass whose own event-loop thread is the limit (the embed reprocess runs many small
    crops and frames side by side, #259) a hop costs less than a few milliseconds of in-place
    work times every image. Ingest keeps the in-place rule it was measured with.
    """
    token = _offload_always.set(True)
    try:
        yield
    finally:
        _offload_always.reset(token)


async def run_cpu[T](
    fn: Callable[..., T], /, *args: Any, nbytes: int | None = None, **kwargs: Any
) -> T:
    """Run ``fn(*args, **kwargs)`` on the CPU pool and await the result.

    ``nbytes`` is the pixel data the call touches. Under ``OFFLOAD_MIN_BYTES`` the call runs
    in place (see the module docstring) unless :func:`offload_always` is active; unknown
    (``None``) is offloaded.
    """
    call = functools.partial(fn, **kwargs) if kwargs else fn
    if nbytes is not None and nbytes < OFFLOAD_MIN_BYTES and not _offload_always.get():
        return call(*args)
    return await run_in_executor_bound(asyncio.get_running_loop(), cpu_executor(), call, *args)


class InflightGate:
    """A first-in first-out gate that admits at most ``capacity`` units at once.

    A request holds the units of the work it brings (ingest counts MiB of decoded pixels, so
    a 32-frame request of 640 x 480 images is 29 units and one 18 MP photo is 52). One that
    wants more than the whole capacity is clamped to it, so it still runs, alone, instead of
    waiting forever.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f'capacity must be at least 1, got {capacity}')
        self.capacity = capacity
        self._free = capacity
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    @property
    def queued(self) -> int:
        return len(self._waiters)

    @asynccontextmanager
    async def hold(self, units: int) -> AsyncIterator[None]:
        units = min(max(1, units), self.capacity)
        await self._acquire(units)
        try:
            yield
        finally:
            self._free += units
            self._wake()

    async def _acquire(self, units: int) -> None:
        if not self._waiters and self._free >= units:
            self._free -= units
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (units, future)
        self._waiters.append(entry)
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                # Admitted at the moment the wait was cancelled: hand the units back.
                self._free += units
            elif entry in self._waiters:
                self._waiters.remove(entry)
            self._wake()
            raise

    def _wake(self) -> None:
        while self._waiters and self._waiters[0][0] <= self._free:
            units, future = self._waiters.popleft()
            if future.cancelled():
                continue
            self._free -= units
            future.set_result(None)


def ingest_gate() -> InflightGate:
    """The process's ingest admission gate, created on first use."""
    pid = os.getpid()
    if pid not in _gates:
        _gates[pid] = InflightGate(inflight_mb())
    return _gates[pid]


__all__ = [
    'DEFAULT_CPU_THREADS',
    'DEFAULT_INFLIGHT_MB',
    'OFFLOAD_MIN_BYTES',
    'InflightGate',
    'cpu_executor',
    'cpu_threads',
    'encoded_nbytes',
    'inflight_mb',
    'ingest_gate',
    'mib',
    'offload_always',
    'run_cpu',
    'shutdown_cpu_executor',
]
