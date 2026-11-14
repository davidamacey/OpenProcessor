"""The resampler's filter-weight cache (``frame_engine/prep_kernels.py``) against a CuPy stand-in.

The weights are device arrays uploaded on the calling worker's stream and read by kernels queued
on that stream. Shared by every worker, an array uploaded on one stream was read by kernels on
others, and clearing the cache at its key limit freed arrays other streams' queued kernels still
read (an illegal address on coco2k at 8 in flight, 2026-11-14). Each worker keeps its own cache
and empties it only after its stream was waited for.
"""

from __future__ import annotations

import importlib
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def kernels(monkeypatch: pytest.MonkeyPatch) -> Any:
    events: list[tuple[str, Any]] = []

    class _Stream:
        def synchronize(self) -> None:
            events.append(('sync', threading.get_ident()))

    cp: Any = types.ModuleType('cupy')
    cp.RawModule = lambda code: types.SimpleNamespace(get_function=lambda name: name)
    cp.asarray = lambda a: np.array(a)
    cp.cuda = types.SimpleNamespace(get_current_stream=_Stream)
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.delitem(sys.modules, 'frame_engine.prep_kernels', raising=False)
    mod = importlib.import_module('frame_engine.prep_kernels')
    k = mod.PrepKernels()
    k.events = events
    k.module = mod
    return k


def test_each_worker_thread_uploads_and_keeps_its_own_weights(kernels: Any) -> None:
    got: dict[str, Any] = {}

    def worker(name: str) -> None:
        got[name] = kernels._weights(4000, 336, 0, 336, True)
        assert kernels._weights(4000, 336, 0, 336, True) is got[name]  # a hit on this worker

    for name in ('a', 'b'):
        t = threading.Thread(target=worker, args=(name,))
        t.start()
        t.join()
    assert got['a'][0] is not got['b'][0]  # never an array another stream uploaded


def test_a_full_cache_is_emptied_only_after_the_stream_was_waited_for(kernels: Any) -> None:
    limit = kernels.module.CACHE_KEYS
    for n in range(1, limit + 1):
        kernels._weights(n + 10, 8, 0, 8, False)
    assert kernels.events == []
    kept = kernels._weights(11, 8, 0, 8, False)
    kernels._weights(limit + 50, 8, 0, 8, False)  # one key over the limit
    assert kernels.events == [('sync', threading.get_ident())]
    assert kernels._weights(11, 8, 0, 8, False) is not kept  # uploaded again after the clear
