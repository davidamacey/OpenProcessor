"""The frame engine's colour step (``frame_engine/colour.py``): the table cache and the gather.

Ground truth is littleCMS through Pillow (``ImageCms``): a converted frame is the CPU route's
conversion. CuPy is stood in for by numpy (the gather runs on the host with the kernel's
indexing); the CUDA source itself is checked by the GPU tier. CC0 profiles from
``tests/fixtures/icc`` (ATTRIBUTION.md).
"""

from __future__ import annotations

import io
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageCms


ICC = Path(__file__).resolve().parents[1] / 'fixtures' / 'icc'
P3 = (ICC / 'DisplayP3-v4.icc').read_bytes()
ADOBE = (ICC / 'AdobeCompat-v4.icc').read_bytes()
COLORD_SRGB = (ICC / 'colord-sRGB.icc').read_bytes()
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))


class _Stream:
    """A stand-in stream: ``alloc`` is the engine's arena stream, ``worker`` a photo's stream."""

    current: dict[int, str] = {}

    def __init__(self, name: str = 'worker') -> None:
        self.name = name

    def __enter__(self) -> _Stream:
        _Stream.current[threading.get_ident()] = self.name
        return self

    def __exit__(self, *_a: Any) -> None:
        _Stream.current.pop(threading.get_ident(), None)

    def synchronize(self) -> None:
        _Cp.syncs += 1
        _Cp.events.append(('sync', self.name, threading.get_ident()))


class _Cp:
    syncs = 0
    uploads = 0
    events: list[tuple[str, str, int]] = []

    @staticmethod
    def asarray(a: Any) -> np.ndarray:
        _Cp.uploads += 1
        stream = _Stream.current.get(threading.get_ident(), 'worker')
        _Cp.events.append(('upload', stream, threading.get_ident()))
        return np.asarray(a).copy()

    class RawKernel:
        def __init__(self, _src: str, _name: str) -> None:
            pass

        def __call__(self, _grid: Any, _block: Any, args: tuple[Any, ...]) -> None:
            _Cp.events.append(('gather', 'worker', threading.get_ident()))
            frame, n, table = args
            px = frame.reshape(int(n), 3)
            idx = px[:, 0].astype(np.int64) * 65536 + px[:, 1].astype(np.int64) * 256 + px[:, 2]
            px[...] = table.reshape(-1, 3)[idx]

    cuda = types.SimpleNamespace(get_current_stream=_Stream)


@pytest.fixture
def step(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setitem(sys.modules, 'cupy', _Cp)
    monkeypatch.delitem(sys.modules, 'frame_engine.colour', raising=False)
    from frame_engine import colour

    _Cp.syncs = _Cp.uploads = 0
    _Cp.events = []
    return lambda capacity=2: colour.ColourStep(capacity, _Stream('alloc'))


def _frame() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 256, (24, 40, 3), dtype=np.uint8)


class TestColourStep:
    def test_an_srgb_frame_is_untouched_and_costs_no_table(self, step: Any) -> None:
        s, f = step(), _frame()
        before = f.copy()
        assert s.apply(f, None) == 'srgb'
        assert s.apply(f, COLORD_SRGB) == 'srgb'
        assert np.array_equal(f, before)
        assert (s.table_bytes, _Cp.uploads, _Cp.syncs) == (0, 0, 0)

    def test_a_p3_frame_becomes_the_littlecms_conversion(self, step: Any) -> None:
        s, f = step(), _frame()
        ref = np.asarray(ImageCms.profileToProfile(
            Image.fromarray(f), ImageCms.ImageCmsProfile(io.BytesIO(P3)), SRGB, outputMode='RGB'
        ))  # fmt: skip
        assert s.apply(f, P3) == 'converted'
        assert np.array_equal(f, ref)
        assert s.table_bytes == 48 * 1024 * 1024

    def test_a_profile_that_is_not_rgb_or_does_not_open_is_ignored(self, step: Any) -> None:
        s, f = step(), _frame()
        before = f.copy()
        assert s.apply(f, b'junk' * 40) == 'ignored'
        assert np.array_equal(f, before)

    def test_tables_are_built_once_and_the_oldest_leaves_first(self, step: Any) -> None:
        s = step(capacity=1)
        s.apply(_frame(), P3)
        s.apply(_frame(), P3)
        assert _Cp.uploads == 1
        s.apply(_frame(), ADOBE)
        assert (_Cp.uploads, s.table_bytes) == (2, 48 * 1024 * 1024)
        s.apply(_frame(), P3)  # evicted by the Adobe table: built again
        assert _Cp.uploads == 3

    def test_photos_of_one_new_profile_share_one_build(
        self, step: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from frame_engine import colour

        s = step()
        gate = threading.Event()
        real = colour.classify
        builds: list[int] = []

        def slow(icc: bytes) -> Any:
            builds.append(1)
            gate.wait(10)
            return real(icc)

        monkeypatch.setattr(colour, 'classify', slow)
        frames = [_frame() for _ in range(4)]
        threads = [threading.Thread(target=s.apply, args=(f, P3)) for f in frames]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(20)
        assert builds == [1]
        assert all(np.array_equal(f, frames[0]) for f in frames)

    def test_a_wait_for_another_photos_build_is_bounded(
        self, step: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from frame_engine import colour
        from frame_engine.transport import OverloadError

        s = step()
        gate = threading.Event()
        monkeypatch.setattr(colour, 'BUILD_WAIT_SECONDS', 0.05)
        monkeypatch.setattr(colour, 'classify', lambda icc: (gate.wait(10), ('convert', None))[1])
        builder = threading.Thread(target=s.apply, args=(_frame(), ADOBE))
        builder.start()
        for _ in range(10_000):  # until the builder has claimed the profile (bounded)
            if s._building:
                break
            gate.wait(0.001)
        with pytest.raises(OverloadError):
            s.apply(_frame(), ADOBE)
        gate.set()
        builder.join(10)

    def test_a_table_is_uploaded_on_the_arena_stream_and_waited_for_before_any_gather(
        self, step: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another worker's stream must never read a table whose upload may still be queued on a
        different stream (the filter-weight cache crash of 2026-11-14)."""
        from frame_engine import colour

        s = step()
        gate = threading.Event()
        real = colour.classify

        def slow(icc: bytes) -> Any:
            gate.wait(10)
            return real(icc)

        monkeypatch.setattr(colour, 'classify', slow)
        threads = [threading.Thread(target=s.apply, args=(_frame(), P3)) for _ in range(3)]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(20)
        kinds = [(e[0], e[1]) for e in _Cp.events]
        assert kinds.count(('upload', 'alloc')) == 1
        assert ('upload', 'worker') not in kinds
        waited = kinds.index(('sync', 'alloc'))
        assert kinds.index(('upload', 'alloc')) < waited
        assert all(i > waited for i, k in enumerate(kinds) if k[0] == 'gather')

    def test_every_reader_waits_for_its_stream_before_it_lets_the_table_go(self, step: Any) -> None:
        """An evicted table is freed only when its last holder drops it; each holder must have
        waited for its own stream (the gather queued there) before ``apply`` returns."""
        s = step(capacity=1)
        for icc in (P3, ADOBE, P3):  # every apply evicts the other profile's table
            before = len(_Cp.events)
            assert s.apply(_frame(), icc) == 'converted'
            mine = [e for e in _Cp.events[before:] if e[2] == threading.get_ident()]
            gathers = [i for i, e in enumerate(mine) if e[0] == 'gather']
            assert gathers
            assert mine[-1][:2] == ('sync', 'worker')
            assert all(i < len(mine) - 1 for i in gathers)
