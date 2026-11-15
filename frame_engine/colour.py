"""The decoded frame in sRGB: a photo's embedded ICC profile applied on the GPU before any branch.

The rule is :mod:`src.utils.icc_srgb`'s, the one the API's CPU decode follows: a profile that
littleCMS maps to sRGB within a level (and an untagged photo) is left alone, so its frame is the
decoded one bit for bit and costs nothing here; an RGB profile that moves colours is converted;
anything else (grey, CMYK, a profile littleCMS cannot open) is ignored as before.

A ``convert`` profile is applied as littleCMS's own answer for every 8-bit input: a table of
``2**24 x 3`` bytes (48 MiB) built once per distinct profile on the host and uploaded, then one
gather per pixel, in place on the frame, on the worker's stream. That is the CPU route's
conversion exactly (the same transform evaluated at the same inputs), where a small 3D grid with
interpolation was measured 8 to 32 levels off near the gamut clip.

Tables: at most ``colour_tables`` (``OP_FRAME_ENGINE_COLOUR_TABLES``) stay on the device, least
recently used first out; none exist until a converted photo arrives. They come from the CuPy pool
(``OP_FRAME_ENGINE_POOL_MIB``) outside the frame budget. A table's memory returns to the pool only
when its last holder drops it, and every holder waits for its own stream after the kernel before
dropping it, so an evicted table is never read after it is freed. A new profile's table is built
by the first photo that needs it, with no lock held while it builds; a photo with the same new
profile waits for that build, every other photo goes on. Untagged and sRGB photos never wait.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Any, Literal

import cupy as cp
import numpy as np

from frame_engine.transport import OverloadError
from src.utils.icc_srgb import ProfileKind, classify


#: What the step did to a frame (``op_frame_engine_colour`` label).
ColourOutcome = Literal['srgb', 'converted', 'ignored']
TABLE_BYTES = (1 << 24) * 3
#: A photo whose new profile another photo is building waits at most this long (a build takes
#: about 2 s of one core), then answers overload (retried by the API).
BUILD_WAIT_SECONDS = 30.0
#: Distinct profiles whose kind is remembered before the memo starts over.
KINDS = 4096

_SRC = r"""
extern "C" __global__ void icc_gather(unsigned char* px, long long n, const unsigned char* table)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        unsigned char* p = px + i * 3;
        const unsigned char* t = table + ((size_t)p[0] * 65536 + (size_t)p[1] * 256 + p[2]) * 3;
        p[0] = t[0]; p[1] = t[1]; p[2] = t[2];
    }
}
"""


def profile_key(icc: bytes) -> str:
    return hashlib.sha256(icc).hexdigest()


class ColourStep:
    def __init__(self, capacity: int, alloc_stream: Any) -> None:
        self.capacity = capacity
        self.alloc_stream = alloc_stream
        self.kernel = cp.RawKernel(_SRC, 'icc_gather')
        self._lock = threading.Lock()
        self._tables: OrderedDict[str, Any] = OrderedDict()
        self._building: dict[str, Future[ProfileKind]] = {}
        #: The kind of every profile seen (a hash and a word each; bounded by KINDS).
        self._kinds: dict[str, ProfileKind] = {}

    @property
    def table_bytes(self) -> int:
        with self._lock:
            return len(self._tables) * TABLE_BYTES

    def apply(self, frame: Any, icc: bytes | None) -> ColourOutcome:
        """Convert ``frame`` (HWC uint8 RGB on the GPU, the current stream's) in place."""
        if not icc:
            return 'srgb'
        with self._lock:
            kind = self._kinds.get(profile_key(icc))
        table = None
        if kind is None or kind == 'convert':
            kind, table = self._lookup(icc)
        if kind == 'srgb':
            return 'srgb'
        if table is None:
            return 'ignored'
        n = int(frame.shape[0]) * int(frame.shape[1])
        self.kernel((min(4096, (n + 255) // 256),), (256,), (frame, np.int64(n), table))
        # an evicted table is freed only when its last holder drops it, after this wait
        cp.cuda.get_current_stream().synchronize()
        return 'converted'

    def _lookup(self, icc: bytes) -> tuple[ProfileKind, Any]:
        """The kind and, for ``convert``, the device table; builds it once per profile."""
        key = profile_key(icc)
        while True:
            with self._lock:
                table = self._tables.get(key)
                if table is not None:
                    self._tables.move_to_end(key)
                    return 'convert', table
                pending = self._building.get(key)
                if pending is None:
                    mine: Future[ProfileKind] = Future()
                    self._building[key] = mine
                    break
            try:
                kind = pending.result(timeout=BUILD_WAIT_SECONDS)
            except FutureTimeout:
                raise OverloadError(
                    'colour', f'profile table not built within {BUILD_WAIT_SECONDS:g} s'
                ) from None
            if kind != 'convert':
                return kind, None
            # built by another photo: take it from the cache (built again if evicted since)
        try:
            kind, host = classify(icc)
            if host is not None:
                with self.alloc_stream:
                    table = cp.asarray(host.reshape(-1))
                self.alloc_stream.synchronize()
        except BaseException as exc:
            with self._lock:
                del self._building[key]
            mine.set_exception(exc)
            raise
        with self._lock:
            del self._building[key]
            if len(self._kinds) >= KINDS:
                self._kinds.clear()
            self._kinds[key] = kind
            if table is not None:
                self._tables[key] = table
                while len(self._tables) > self.capacity:
                    self._tables.popitem(last=False)
        mine.set_result(kind)
        return kind, table
