"""Full-COCO manifest mode: ``--bench-set all`` lists every image id once."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.datasets import bench_set as b, fetch_coco_subset as f
from scripts.datasets._common import FetchError


if TYPE_CHECKING:
    from pathlib import Path

LICENSES = [{'id': 1, 'name': 'Attribution License'}, {'id': 2, 'name': 'NC License'}]


def _fake_fetch(rows: list[dict[str, Any]], _images_dir: Any, *, pinned: bool) -> None:
    assert not pinned
    for r in rows:
        r.update(bytes=1, sha256='0')


def _images(ids: range) -> list[dict[str, Any]]:
    return [
        {'id': i, 'file_name': f'{i:012d}.jpg', 'license': 1 + i % 2, 'width': 4, 'height': 3}
        for i in ids
    ]


def _doc(ids: range) -> dict[str, Any]:
    return {'images': _images(ids), 'licenses': LICENSES, 'annotations': [], 'categories': []}


def test_all_mode_lists_every_image_once_with_every_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        f, 'ensure_annotations', lambda _c: (_doc(range(1, 6)), _doc(range(10, 18)))
    )
    monkeypatch.setattr(
        b,
        'fetch_images',
        _fake_fetch,
    )
    pin_path = tmp_path / 'pin.json'
    args = ['--bench-set', 'all', '--out', str(tmp_path / 'o'), '--manifest', str(pin_path)]
    assert f.main(args) == 0
    pin = json.loads(pin_path.read_text())
    ids = [r['image_id'] for r in pin['images']]
    assert sorted(ids) == [*range(1, 6), *range(10, 18)]
    assert len(ids) == len(set(ids)) == pin['count'] == 13
    assert {r['split'] for r in pin['images']} == {'val2017', 'train2017'}


def test_with_test_adds_the_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f, 'ensure_annotations', lambda _c: (_doc(range(1, 3)), _doc(range(10, 12)))
    )
    monkeypatch.setattr(
        b,
        'fetch_images',
        _fake_fetch,
    )
    info = tmp_path / 'info.json'
    info.write_text(json.dumps({'images': _images(range(100, 103)), 'licenses': LICENSES}))
    pin_path = tmp_path / 'pin.json'
    args = [
        '--bench-set',
        'all',
        '--with-test',
        str(info),
        '--out',
        str(tmp_path / 'o'),
        '--manifest',
        str(pin_path),
    ]
    assert f.main(args) == 0
    assert json.loads(pin_path.read_text())['count'] == 7


def test_with_test_needs_all_mode(tmp_path: Path) -> None:
    args = [
        '--bench-set',
        '5',
        '--with-test',
        'x',
        '--out',
        str(tmp_path),
        '--manifest',
        str(tmp_path / 'p'),
    ]
    assert f.main(args) == 1


def test_duplicate_id_across_splits_stops() -> None:
    pool = [{'image_id': 1}]
    with pytest.raises(FetchError):
        b.select_all(pool, pool)


def test_numeric_bench_set_is_unchanged() -> None:
    assert f.build_parser().parse_args(['--out', 'x', '--bench-set', '2000']).bench_set == 2000
    with pytest.raises(SystemExit):
        f.build_parser().parse_args(['--out', 'x', '--bench-set', 'some'])
