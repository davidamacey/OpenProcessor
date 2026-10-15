"""Selection and pin logic of scripts/datasets/fetch_openimages_bench.py (no network)."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from scripts.datasets import fetch_openimages_bench as fb
from scripts.datasets._common import FetchError


if TYPE_CHECKING:
    from pathlib import Path


BY = fb.ALLOWED_LICENSE_URL
CSV = (
    'ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author\n'
    f'b,train,http://x/b,l,{BY},p,Ann\n'
    'z,train,http://x/z,l,https://creativecommons.org/licenses/by-nc/2.0/,p,Bob\n'
    f'a,train,http://x/a,l,{BY},p,Cy\n'
)


def _rows(ids: str) -> list[dict[str, object]]:
    return [
        {
            'image_id': i,
            'license': BY,
            'author': 'x',
            'original_url': 'u',
            'bytes': 3,
            'width': 4,
            'height': 5,
            'sha256': i * 2,
        }
        for i in ids
    ]


def test_eligible_rows_filters_licence_and_sorts() -> None:
    assert [r['image_id'] for r in fb.eligible_rows(CSV)] == ['a', 'b']


def test_candidate_order_is_seeded_and_complete() -> None:
    pool = [{'image_id': str(i)} for i in range(50)]
    a = fb.candidate_order(pool, 7)
    assert a == fb.candidate_order(pool, 7)
    assert a != fb.candidate_order(pool, 8)
    assert sorted(r['image_id'] for r in a) == sorted(r['image_id'] for r in pool)


def test_pin_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    pin = fb.build_pin(_rows('cab'), 1)
    assert [r['image_id'] for r in pin['images']] == ['a', 'b', 'c']
    path = tmp_path / 'p.json'
    path.write_text(json.dumps(pin))
    assert fb.load_pin(path)['count'] == 3
    pin['images'][0]['sha256'] = 'tampered'
    path.write_text(json.dumps(pin))
    with pytest.raises(FetchError, match='manifest_sha256'):
        fb.load_pin(path)


def test_verify_images_reports_missing_size_and_hash(tmp_path: Path) -> None:
    good = b'abc'
    rows = [{**r, 'sha256': hashlib.sha256(good).hexdigest()} for r in _rows('abc')]
    pin = fb.build_pin(rows, 1)
    (tmp_path / 'a.jpg').write_bytes(good)
    (tmp_path / 'b.jpg').write_bytes(b'abcd')
    problems = fb.verify_images(pin, tmp_path)
    assert problems == ['b.jpg: 4 bytes, pinned 3', 'c.jpg: missing']
    (tmp_path / 'b.jpg').write_bytes(b'xyz')
    assert 'b.jpg: sha256 differs from the pin' in fb.verify_images(pin, tmp_path)
