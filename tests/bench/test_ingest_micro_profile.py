"""Tests for scripts/bench/ingest_micro_profile.py on synthetic JPEGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from scripts.bench import ingest_micro_profile as m


if TYPE_CHECKING:
    from pathlib import Path


def write_jpegs(directory: Path, size: tuple[int, int], n: int) -> None:
    directory.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        pixels = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f'{i}.jpg', quality=90)


def test_profile_directory_reports_every_operation_and_scales_with_pixels(tmp_path: Path) -> None:
    write_jpegs(tmp_path / 'small', (160, 120), 3)
    write_jpegs(tmp_path / 'large', (2400, 1800), 3)
    small = m.profile_directory(tmp_path / 'small', limit=3)
    large = m.profile_directory(tmp_path / 'large', limit=3)
    assert set(small) == {*m.OPERATIONS, 'megapixels'}
    assert small['megapixels'] == pytest.approx(0.0192)
    assert large['megapixels'] == pytest.approx(4.32)
    assert all(v >= 0 for v in small.values())
    # decode and the full-frame letterbox must cost more on 225x the pixels
    assert large['decode'] > small['decode']
    assert large['letterbox'] > small['letterbox']


def test_profile_directory_honors_limit_and_rejects_an_empty_directory(tmp_path: Path) -> None:
    write_jpegs(tmp_path / 'd', (64, 48), 5)
    assert m.profile_directory(tmp_path / 'd', limit=2)['megapixels'] > 0
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        m.profile_directory(tmp_path / 'empty', limit=2)
