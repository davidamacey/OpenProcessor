"""The class of an embedded profile (``icc_srgb.profile_kind``): decided from the profile's own tags
when they settle it, by the transform otherwise, and computed once however many threads meet a
new profile at the same moment.

Ground truth for the class is the rule itself, written out here independently of the module: the
profile-to-sRGB transform of littleCMS evaluated at every 8-bit input, ``srgb`` when no input moves
by more than one level.
"""

from __future__ import annotations

import io
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageCms

from src.utils import icc_srgb


ICC = Path(__file__).resolve().parent / 'fixtures' / 'icc'
FIXTURES = sorted(ICC.glob('*.icc'))
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
BUILTIN_SRGB = SRGB.tobytes()
COLORD = (ICC / 'colord-sRGB.icc').read_bytes()
P3 = (ICC / 'DisplayP3-v4.icc').read_bytes()


def oracle(icc: bytes) -> str:
    """The rule, from its definition (every 8-bit input through littleCMS)."""
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if prof.profile.xcolor_space != 'RGB ':
            return 'other'
        transform = ImageCms.buildTransform(prof, SRGB, 'RGB', 'RGB')
    except (OSError, ImageCms.PyCMSError):
        return 'other'
    v = np.arange(1 << 24, dtype=np.uint32)
    rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb.reshape(4096, 4096, 3))
    out = np.asarray(transform.apply(img)).reshape(-1, 3).astype(np.int16)
    return 'srgb' if int(np.abs(out - rgb.astype(np.int16)).max()) <= 1 else 'convert'


def fresh(icc: bytes, tag: int) -> bytes:
    """The same profile under other bytes (the creation date field), so the memo has not seen it."""
    return icc[:24] + struct.pack('>I', tag) + icc[28:]


def with_red_x(icc: bytes, x: float) -> bytes:
    """``icc`` with the x of its red colorant (``rXYZ``) replaced."""
    (count,) = struct.unpack('>I', icc[128:132])
    for i in range(count):
        at = 132 + 12 * i
        if icc[at : at + 4] == b'rXYZ':
            (offset,) = struct.unpack('>I', icc[at + 4 : at + 8])
            return icc[: offset + 8] + struct.pack('>i', round(x * 65536)) + icc[offset + 12 :]
    raise AssertionError('no rXYZ')


def red_x(icc: bytes) -> float:
    red = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile.red_colorant
    assert red is not None
    return float(red[0][0])


class Counting:
    """Stands in for littleCMS's profile reader (``ImageCms.core.profile_frombytes``, which every
    ``ImageCmsProfile(...)`` calls) and counts how often this profile's bytes are opened (every
    classification of it opens it at least once), each open lasting ``pause`` seconds so that
    threads arriving together overlap. ``fail_first`` makes the first open raise."""

    def __init__(self, icc: bytes, pause: float = 0.0, *, fail_first: bool = False) -> None:
        self.real = ImageCms.core.profile_frombytes
        self.icc, self.pause, self.fail_first = icc, pause, fail_first
        self.opened = 0
        self._lock = threading.Lock()

    def __call__(self, data: bytes) -> Any:
        if data == self.icc:
            with self._lock:
                self.opened += 1
                first = self.opened == 1
            time.sleep(self.pause)
            if first and self.fail_first:
                raise RuntimeError('boom')
        return self.real(data)


@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.name)
def test_the_class_is_the_rules_for_every_fixture_profile(path: Path) -> None:
    icc = path.read_bytes()
    assert icc_srgb.profile_kind(fresh(icc, 1)) == oracle(icc)
    assert icc_srgb.classify(fresh(icc, 2))[0] == oracle(icc)


def test_the_built_in_srgb_profile_is_srgb() -> None:
    assert icc_srgb.profile_kind(BUILTIN_SRGB) == 'srgb'
    assert oracle(BUILTIN_SRGB) == 'srgb'


@pytest.mark.parametrize(('icc', 'kind'), [(COLORD, 'srgb'), (P3, 'convert')])
def test_a_profile_the_tags_settle_builds_no_transform(
    monkeypatch: pytest.MonkeyPatch, icc: bytes, kind: str
) -> None:
    def refuse(_icc: bytes) -> None:
        raise AssertionError('a transform was built for a profile whose tags settle its class')

    monkeypatch.setattr(icc_srgb, '_transform', refuse)
    assert icc_srgb.profile_kind(fresh(icc, 3)) == kind


def test_a_profile_the_tags_cannot_settle_goes_to_the_transform() -> None:
    """A red colorant 0.004 off sRGB's is too near to call from the matrix alone."""
    icc = with_red_x(COLORD, red_x(COLORD) + 0.004)
    assert icc_srgb.profile_kind(fresh(icc, 4)) == oracle(icc)


def test_threads_that_meet_a_new_profile_together_compute_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    icc = fresh(P3, 5)
    counting = Counting(icc, pause=0.2)
    monkeypatch.setattr(ImageCms.core, 'profile_frombytes', counting)
    start = threading.Barrier(8)
    kinds: list[str] = []

    def call() -> None:
        start.wait()
        kinds.append(icc_srgb.profile_kind(icc))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert kinds == ['convert'] * 8
    assert counting.opened == 1


def test_a_failed_computation_is_not_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    icc = fresh(P3, 6)
    monkeypatch.setattr(ImageCms.core, 'profile_frombytes', Counting(icc, fail_first=True))
    with pytest.raises(RuntimeError):
        icc_srgb.profile_kind(icc)
    assert icc_srgb.profile_kind(icc) == 'convert'


def test_untagged_and_unopenable_profiles() -> None:
    assert icc_srgb.profile_kind(None) == 'srgb'
    assert icc_srgb.profile_kind(b'\x00' * 200) == 'other'
