"""The resampler's filter weights (``frame_engine/prep_kernels.py``) against a CuPy stand-in.

A weight set is built on the host once per distinct size (a process-wide ``lru_cache``), packed
into the one byte string the kernels read, and staged for each resample in the calling worker's
own arena: a pinned host buffer copied to its device twin on the worker's stream. The arena is
reused from its front only after the stream was waited for, because queued kernels still read the
device bytes and queued copies still read the host bytes. Shared between workers, device weights
uploaded on one stream were read by kernels on others, and a cache cleared at its limit freed
arrays other streams' queued kernels still read (an illegal address on coco2k at 8 in flight,
2026-11-14): that is why no device memory here is shared across workers.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from frame_engine.prep_geometry import filter_weights


class _Pinned(bytearray):
    """``cupy.cuda.alloc_pinned_memory``'s result: a buffer with an address."""

    ptr = 0


@pytest.fixture
def kernels(monkeypatch: pytest.MonkeyPatch) -> Any:
    events: list[tuple[str, Any]] = []
    device: dict[int, int] = {}  # device address -> byte value written there by a copy
    pinned: list[_Pinned] = []
    next_address = [0x1000_0000]

    def fresh_address(n: int) -> int:
        at = next_address[0]
        next_address[0] += (n + 0xFFFF) & ~0xFFFF
        return at

    class _Stream:
        ptr = 7

        def synchronize(self) -> None:
            events.append(('sync', threading.get_ident()))

    def alloc_pinned(n: int) -> _Pinned:
        p = _Pinned(n)
        p.ptr = fresh_address(n)
        pinned.append(p)
        return p

    def empty(n: int, _dtype: Any) -> Any:
        return types.SimpleNamespace(data=types.SimpleNamespace(ptr=fresh_address(n)))

    def memcpy_async(dst: int, src: int, n: int, kind: int, stream: int) -> None:
        assert kind == 1
        assert stream == 7
        events.append(('copy', dst))
        host = next(p for p in pinned if p.ptr <= src < p.ptr + len(p))
        device.update({dst + k: host[src - host.ptr + k] for k in range(min(n, 4096))})

    cp: Any = types.ModuleType('cupy')
    cp.RawModule = lambda code: types.SimpleNamespace(get_function=lambda name: name)
    cp.empty = empty
    cp.uint8 = np.uint8
    cp.cuda = types.SimpleNamespace(
        get_current_stream=_Stream,
        alloc_pinned_memory=alloc_pinned,
        runtime=types.SimpleNamespace(memcpyAsync=memcpy_async, memcpyHostToDevice=1),
    )
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.delitem(sys.modules, 'frame_engine.prep_kernels', raising=False)
    mod = importlib.import_module('frame_engine.prep_kernels')
    k = mod.PrepKernels()
    k.events, k.device, k.module = events, device, mod
    return k


def _unpack(w: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First taps, tap counts and weights back out of a packed set (the kernels' layout)."""
    b = w.blob
    return (
        b[: 4 * w.n].view(np.int32),
        b[4 * w.n : 8 * w.n].view(np.int32),
        b[8 * w.n :].view(np.float32).reshape(w.n, w.kmax),
    )


def test_a_packed_set_holds_the_filter_weights_of_its_window(kernels: Any) -> None:
    w = kernels.module._weights(4000, 336, 20, 100, True, False)
    f, c, wt = filter_weights(4000, 336, antialias=True, lanczos=False)
    got_f, got_c, got_w = _unpack(w)
    assert np.array_equal(got_f, f[20:120])
    assert np.array_equal(got_c, c[20:120])
    assert np.array_equal(got_w, wt[20:120])
    assert (w.n, w.kmax) == (100, wt.shape[1])
    assert (w.lo, w.hi) == (int(f[20]), int((f[20:120] + c[20:120]).max()))


def test_a_size_is_computed_once_for_every_worker(kernels: Any) -> None:
    got: dict[str, Any] = {}

    def worker(name: str) -> None:
        got[name] = kernels.module._weights(4000, 336, 0, 336, True, False)

    for name in ('a', 'b'):
        t = threading.Thread(target=worker, args=(name,))
        t.start()
        t.join()
    assert got['a'] is got['b']


def test_the_device_copy_starts_with_the_packed_set(kernels: Any) -> None:
    w = kernels.module._weights(300, 64, 0, 64, False, False)
    address = kernels._arena().put(w.blob)
    head = np.array([kernels.device[address + k] for k in range(64)], np.uint8)
    assert np.array_equal(head, w.blob[:64])
    assert address % 16 == 0


def test_each_worker_stages_in_its_own_arena(kernels: Any) -> None:
    w = kernels.module._weights(300, 64, 0, 64, False, False)
    got: dict[str, int] = {}

    def worker(name: str) -> None:
        got[name] = kernels._arena().put(w.blob)

    for name in ('a', 'b'):
        t = threading.Thread(target=worker, args=(name,))
        t.start()
        t.join()
    assert abs(got['a'] - got['b']) >= kernels.module.ARENA_BYTES  # never the other's bytes


def test_a_full_arena_is_reused_only_after_the_stream_was_waited_for(kernels: Any) -> None:
    arena = kernels._arena()
    quarter = np.ones(kernels.module.ARENA_BYTES // 4, np.uint8)
    first = arena.put(quarter)
    for _ in range(3):
        arena.put(quarter)
    assert not [e for e in kernels.events if e[0] == 'sync']
    again = arena.put(quarter)  # the fifth does not fit behind the other four
    kinds = [e[0] for e in kernels.events]
    assert kinds.index('sync') == len(kinds) - 2  # the wait comes right before the restarting copy
    assert kinds[-1] == 'copy'
    assert again == first  # and starts over at the front


def test_a_set_larger_than_the_arena_is_refused(kernels: Any) -> None:
    with pytest.raises(ValueError, match='does not fit'):
        kernels._arena().put(np.zeros(kernels.module.ARENA_BYTES + 1, np.uint8))
