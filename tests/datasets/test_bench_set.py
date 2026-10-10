"""Tests for scripts/datasets/bench_set.py: seeded selection, pin format, checksum verification.

Offline: the "downloads" are tiny files served through a monkeypatched downloader.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.datasets import bench_set as b
from scripts.datasets._common import FetchError


if TYPE_CHECKING:
    from pathlib import Path

LICENSES = {1: 'Attribution License', 2: 'Attribution-NonCommercial License'}


def _images(n: int, start: int = 1) -> list[dict[str, Any]]:
    return [
        {
            'id': i,
            'file_name': f'{i:012d}.jpg',
            'license': 1 if i % 4 else 2,
            'width': 640,
            'height': 480,
            'flickr_url': f'http://example.invalid/{i}',
        }
        for i in range(start, start + n)
    ]


def _pools() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = {LICENSES[1]}
    return (
        b.pool_rows(_images(40), LICENSES, allowed, 'val2017'),
        b.pool_rows(_images(40, 1000), LICENSES, allowed, 'train2017'),
    )


def test_pool_drops_disallowed_licenses_and_sorts() -> None:
    val, _ = _pools()
    assert all(r['license_name'] == LICENSES[1] for r in val)
    assert [r['image_id'] for r in val] == sorted(r['image_id'] for r in val)
    assert len(val) == 30  # every fourth image carries the NonCommercial license


def test_selection_is_deterministic_and_seed_sensitive() -> None:
    val, train = _pools()
    first = b.select_uniform(val, train, 10, seed=7)
    assert first == b.select_uniform(val, train, 10, seed=7)
    assert [r['image_id'] for r in first] != [
        r['image_id'] for r in b.select_uniform(val, train, 10, seed=8)
    ]
    assert len({r['image_id'] for r in first}) == 10


def test_selection_prefers_val_and_tops_up_from_train() -> None:
    val, train = _pools()
    small = b.select_uniform(val, train, 10, seed=1)
    assert {r['split'] for r in small} == {'val2017'}
    big = b.select_uniform(val, train, 40, seed=1)
    assert sum(r['split'] == 'val2017' for r in big) == 30
    assert sum(r['split'] == 'train2017' for r in big) == 10


def test_selection_refuses_to_shrink() -> None:
    val, train = _pools()
    with pytest.raises(FetchError, match='only 60 eligible'):
        b.select_uniform(val, train, 61, seed=1)


def _filled(rows: list[dict[str, Any]], payload: bytes = b'x') -> list[dict[str, Any]]:
    return [
        {**r, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()} for r in rows
    ]


def test_manifest_hash_changes_with_any_row_field() -> None:
    val, _ = _pools()
    rows = _filled(val[:3])
    changed = [dict(r) for r in rows]
    changed[1]['sha256'] = '0' * 64
    assert b.manifest_hash(rows) != b.manifest_hash(changed)
    assert b.manifest_hash(rows) == b.manifest_hash([dict(r) for r in rows])


def test_pin_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    val, _ = _pools()
    pin = b.build_pin(_filled(val[:3]), seed=5, licenses=['b', 'a'], annotations_sha256='f' * 64)
    path = tmp_path / 'pin.json'
    path.write_text(json.dumps(pin), encoding='utf-8')
    assert b.load_pin(path)['licenses'] == ['a', 'b']
    pin['images'][0]['sha256'] = '1' * 64
    path.write_text(json.dumps(pin), encoding='utf-8')
    with pytest.raises(FetchError, match='manifest_sha256'):
        b.load_pin(path)


def test_verify_images_reports_missing_size_and_checksum(tmp_path: Path) -> None:
    val, _ = _pools()
    rows = _filled(val[:3])
    pin = b.build_pin(rows, seed=1, licenses=[], annotations_sha256='')
    images = tmp_path / 'images'
    images.mkdir()
    assert len(b.verify_images(pin, images)) == 3  # all missing
    (images / rows[0]['file_name']).write_bytes(b'x')
    (images / rows[1]['file_name']).write_bytes(b'xx')  # wrong size
    (images / rows[2]['file_name']).write_bytes(b'y')  # right size, wrong bytes
    problems = b.verify_images(pin, images)
    assert len(problems) == 2
    assert 'pinned 1' in problems[0]
    assert 'sha256 differs' in problems[1]


def test_first_run_writes_pin_then_later_run_enforces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val, _ = _pools()
    rows = val[:3]
    served = {'body': b'pixels'}

    def fake_download(url: str, dest: Path, *, expected_sha256: str | None = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(served['body'])
        if expected_sha256 and hashlib.sha256(served['body']).hexdigest() != expected_sha256:
            raise FetchError('sha256 mismatch')
        return dest

    monkeypatch.setattr(b, 'download', fake_download)
    pin_path = tmp_path / 'pin.json'
    args = {'seed': 5, 'licenses': ['a'], 'annotations_sha256': 'f' * 64}
    first = b.write_or_check_pin(pin_path, [dict(r) for r in rows], args, tmp_path / 'img')
    assert first['count'] == 3
    assert pin_path.is_file()
    assert all(r['bytes'] == len(b'pixels') for r in first['images'])

    again = b.write_or_check_pin(pin_path, [dict(r) for r in rows], args, tmp_path / 'img2')
    assert again['manifest_sha256'] == first['manifest_sha256']

    served['body'] = b'tampered'
    with pytest.raises(FetchError, match='sha256 mismatch'):
        b.write_or_check_pin(pin_path, [dict(r) for r in rows], args, tmp_path / 'img3')

    with pytest.raises(FetchError, match='selection drift'):
        b.write_or_check_pin(pin_path, [dict(r) for r in rows[:2]], args, tmp_path / 'img4')
    with pytest.raises(FetchError, match='annotation archive'):
        b.write_or_check_pin(
            pin_path, [dict(r) for r in rows], {**args, 'annotations_sha256': 'e' * 64}, tmp_path
        )


def test_cli_verify_only_exit_codes(tmp_path: Path) -> None:
    from scripts.datasets import fetch_coco_subset as f

    val, _ = _pools()
    rows = _filled(val[:2])
    pin_path = tmp_path / 'pin.json'
    pin_path.write_text(
        json.dumps(b.build_pin(rows, seed=1, licenses=[], annotations_sha256='')), encoding='utf-8'
    )
    out = tmp_path / 'set'
    (out / 'images').mkdir(parents=True)
    argv = ['--verify-only', '--out', str(out), '--manifest', str(pin_path)]
    assert f.main(argv) == 1  # files missing
    for r in rows:
        (out / 'images' / r['file_name']).write_bytes(b'x')
    assert f.main(argv) == 0
    (out / 'images' / rows[0]['file_name']).write_bytes(b'z')
    assert f.main(argv) == 1  # same size, different bytes


def test_cli_bench_set_requires_a_manifest(tmp_path: Path) -> None:
    from scripts.datasets import fetch_coco_subset as f

    assert f.main(['--bench-set', '5', '--out', str(tmp_path)]) == 1
