"""Tests for the pure parts of scripts/datasets/fetch_hires_set.py (no network)."""

from __future__ import annotations

from typing import Any

from scripts.datasets import fetch_hires_set as f
from scripts.datasets.bench_set import build_pin, load_pin


def page(
    page_id: int,
    width: int,
    height: int,
    *,
    mime: str = 'image/jpeg',
    license_name: str = 'CC BY-SA 4.0',
    artist: str = '<a href="x">Jane &amp; Co</a>',
) -> dict[str, Any]:
    return {
        'pageid': page_id,
        'title': f'File:Photo {page_id}.jpg',
        'imageinfo': [
            {
                'url': f'https://upload.example/p{page_id}.jpg?utm_source=x',
                'descriptionurl': f'https://commons.example/wiki/{page_id}',
                'width': width,
                'height': height,
                'mime': mime,
                'extmetadata': {
                    'LicenseShortName': {'value': license_name},
                    'Artist': {'value': artist},
                },
            }
        ],
    }


def test_candidate_rows_filters_size_mime_and_license_and_sorts() -> None:
    pages = [
        page(30, 4000, 3000),  # 12 MP, kept
        page(10, 5472, 3648),  # 20 MP, kept
        page(20, 640, 480),  # too small
        page(21, 8000, 6000),  # 48 MP, too big
        page(22, 4000, 3000, mime='image/png'),
        page(23, 4000, 3000, license_name='All rights reserved'),
        page(24, 4000, 3000, license_name='Public domain'),
        page(25, 4000, 3000, license_name='CC0'),
        {'pageid': 99, 'title': 'File:no info.jpg'},
    ]
    rows = f.candidate_rows(pages, 12_000_000, 24_000_000)
    assert [r['page_id'] for r in rows] == [10, 24, 25, 30]
    first = rows[0]
    assert first['url'] == 'https://upload.example/p10.jpg'
    assert first['author'] == 'Jane & Co'
    assert first['title'] == 'Photo 10.jpg'


def test_license_pattern_does_not_accept_a_nonfree_variant() -> None:
    assert f.ALLOWED_LICENSE.match('CC BY 2.0')
    assert f.ALLOWED_LICENSE.match('CC BY-SA 3.0')
    assert not f.ALLOWED_LICENSE.match('CC BY-NC 2.0')
    assert not f.ALLOWED_LICENSE.match('CC BY-ND 2.0')


def test_pin_rows_load_as_a_bench_pin() -> None:
    cand = f.candidate_rows([page(7, 4000, 3000)], 1, 10**9)[0]
    row = f.pin_row(cand, 5_000_000, 'ab' * 32)
    pin = build_pin([row], seed=1, licenses=['CC BY-SA 4.0'], annotations_sha256='n/a')
    # load_pin accepts the same shape as the COCO pin
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'pin.json'
        path.write_text(json.dumps(pin))
        loaded = load_pin(path)
    assert loaded['images'][0]['file_name'] == '7.jpg'
    assert loaded['images'][0]['width'] * loaded['images'][0]['height'] == 12_000_000


def test_candidate_rows_empty_window() -> None:
    assert f.candidate_rows([page(1, 4000, 3000)], 20_000_000, 24_000_000) == []
