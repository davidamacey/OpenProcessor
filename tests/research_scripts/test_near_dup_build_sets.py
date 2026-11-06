"""nd:P0: the constructed copy set builders (sizes and determinism checked against arithmetic)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.research.near_dup import build_sets


def _jpeg(path: Path, w: int, h: int, seed: int) -> None:
    arr = np.random.default_rng(seed).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, 'JPEG', quality=95)


def test_every_transform_has_the_documented_size() -> None:
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, size=(200, 400, 3), dtype=np.uint8)
    )
    sizes = {}
    for name in build_sets.TRANSFORMS:
        data, ext = build_sets.apply_transform(name, img)
        with Image.open(io.BytesIO(data)) as out:
            sizes[name] = out.size
        assert ext in {'jpg', 'png'}
    assert sizes['T-half'] == (200, 100)
    assert sizes['T-quarter'] == (100, 50)
    assert sizes['T-crop90'] == (360, 180)  # 400 * 0.9, 200 * 0.9
    assert sizes['T-crop80'] == (320, 160)
    assert sizes['CB-crop90'] == (360, 180)  # 5 % off each side: 400 - 40, 200 - 20
    assert sizes['T-q60'] == (400, 200)
    wr, hr = sizes['T-rot']
    assert wr < 400
    assert hr < 200


def test_inscribed_rect_is_inside_the_rotated_frame() -> None:
    # A square rotated 45 degrees: the inscribed square has side w / sqrt(2) = 70.7 for w = 100.
    wr, hr = build_sets.inscribed_rect(100, 100, 45.0)
    assert wr == pytest.approx(70, abs=1)
    assert hr == pytest.approx(70, abs=1)
    assert build_sets.inscribed_rect(100, 100, 0.0) == (100, 100)


def test_sample_is_seeded_and_honours_the_exclusion(tmp_path: Path) -> None:
    files = []
    for i in range(12):
        p = tmp_path / f'{i}.jpg'
        _jpeg(p, 40, 30, i)
        files.append(p)
    first = build_sets.sample_originals({'a': files}, 5, seed=7, exclude=set())
    again = build_sets.sample_originals({'a': files}, 5, seed=7, exclude=set())
    assert first == again
    assert len(first) == 5
    banned = {first[0]['sha256']}
    third = build_sets.sample_originals({'a': files}, 5, seed=7, exclude=banned)
    assert first[0]['sha256'] not in {r['sha256'] for r in third}
    assert len(third) == 5


def test_build_copies_writes_hashed_files(tmp_path: Path) -> None:
    src = tmp_path / 'o.jpg'
    _jpeg(src, 64, 48, 1)
    rows = build_sets.build_copies(
        [{'source': 'a', 'path': str(src), 'sha256': ''}],
        tmp_path / 'out',
        ('T-q60', 'T-half'),
        workers=1,
    )
    assert {r['transform'] for r in rows} == {'T-q60', 'T-half'}
    for r in rows:
        assert build_sets.sha256_file(Path(str(r['path']))) == r['sha256']


def test_read_copydays_pairs_variants_with_originals(tmp_path: Path) -> None:
    for kind, names in {
        'original': ['200000', '200100'],
        'strong': ['200001', '200101', '999901'],
    }.items():
        (tmp_path / kind).mkdir()
        for k, name in enumerate(names):
            _jpeg(tmp_path / kind / f'{name}.jpg', 16, 16, k)
    rows = build_sets.read_copydays(tmp_path)
    assert sum(r['transform'] == 'orig' for r in rows) == 2
    strong = [r for r in rows if r['transform'] == 'copydays_strong']
    assert len(strong) == 2  # 999901 has no original and is dropped
    assert {r['family'] for r in strong} == {0, 1}
