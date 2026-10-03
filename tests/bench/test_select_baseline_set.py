"""Deterministic baseline-set selection, manifest checksum and COCO license sidecar."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from scripts.bench import select_baseline_set as sel


if TYPE_CHECKING:
    from pathlib import Path


def _img(path: Path, size: tuple[int, int] = (400, 360)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, (10, 20, 30)).save(path, 'JPEG')


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / 'arch'
    for sub, n in (('a', 12), ('b', 3), ('c', 12)):
        for i in range(n):
            _img(root / sub / f'{i:03d}.jpg')
    _img(root / 'a' / 'tiny.jpg', (100, 400))
    (root / 'a' / 'broken.jpg').write_bytes(b'not a jpeg at all')
    (root / 'a' / 'notes.txt').write_text('x')
    return root


def test_same_seed_same_selection_and_other_seed_differs(archive: Path) -> None:
    one = sel.select_images([archive], 10, seed=42)
    two = sel.select_images([archive], 10, seed=42)
    other = sel.select_images([archive], 10, seed=7)
    assert [c.path for c in one] == [c.path for c in two]
    assert [c.path for c in one] != [c.path for c in other]


def test_filters_drop_tiny_broken_and_non_jpeg(archive: Path) -> None:
    picked = sel.select_images([archive], 1000, seed=1)
    names = {c.path.name for c in picked}
    assert 'tiny.jpg' not in names
    assert 'broken.jpg' not in names
    assert 'notes.txt' not in names
    assert len(picked) == 27


def test_size_ceiling_is_enforced(archive: Path) -> None:
    got = sel.select_images([archive], 1000, seed=1, max_bytes=1)
    assert got == []


def test_round_robin_across_first_level_subfolders(archive: Path) -> None:
    picked = sel.select_images([archive], 9, seed=42)
    per_folder = {f: sum(1 for c in picked if c.path.parent.name == f) for f in 'abc'}
    assert per_folder == {'a': 3, 'b': 3, 'c': 3}


def test_round_robin_continues_when_a_folder_runs_dry(archive: Path) -> None:
    picked = sel.select_images([archive], 20, seed=42)
    per_folder = {f: sum(1 for c in picked if c.path.parent.name == f) for f in 'abc'}
    assert per_folder['b'] == 3
    assert per_folder['a'] + per_folder['c'] == 17


def test_multiple_roots_count_per_source(tmp_path: Path) -> None:
    r1, r2 = tmp_path / 'one', tmp_path / 'two'
    for i in range(5):
        _img(r1 / f'{i}.jpg')
        _img(r2 / f'{i}.jpeg')
    picked = sel.select_images([r1, r2], 6, seed=3)
    assert sel.per_source_counts(picked) == {'one': 3, 'two': 3}


def test_manifest_header_body_and_checksum_roundtrip(archive: Path, tmp_path: Path) -> None:
    picked = sel.select_images([archive], 6, seed=42)
    out = tmp_path / 'manifest.txt'
    digest = sel.write_manifest(out, picked, seed=42, today='2026-10-03')
    text = out.read_text().splitlines()
    assert text[0].startswith('# seed: 42')
    assert any(line == f'# count: {len(picked)}' for line in text)
    assert any(line.startswith('# bytes: ') for line in text)
    assert any(line.startswith('# sources: arch=6') for line in text)
    assert '# date: 2026-10-03' in text
    assert f'# sha256: {digest}' in text
    assert sel.read_manifest(out) == [str(c.path) for c in picked]


def test_checksum_ignores_header_but_detects_body_edits(archive: Path, tmp_path: Path) -> None:
    picked = sel.select_images([archive], 6, seed=42)
    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    da = sel.write_manifest(a, picked, seed=42, today='2026-01-01')
    db = sel.write_manifest(b, picked, seed=42, today='2030-01-01')
    assert da == db
    lines = a.read_text().splitlines()
    idx = next(i for i, line in enumerate(lines) if not line.startswith('#'))
    lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]
    a.write_text('\n'.join(lines) + '\n')
    with pytest.raises(sel.ManifestError, match='checksum'):
        sel.read_manifest(a)


def _coco_fixture(tmp_path: Path, n: int = 20) -> tuple[Path, Path]:
    images_dir = tmp_path / 'val2017'
    images = []
    for i in range(1, n + 1):
        name = f'{i:012d}.jpg'
        _img(images_dir / name, (640, 480))
        images.append(
            {
                'id': i,
                'file_name': name,
                'width': 640,
                'height': 480,
                'license': 1 if i % 2 else 4,
            }
        )
    ann = tmp_path / 'instances_val2017.json'
    ann.write_text(
        json.dumps(
            {
                'images': images,
                'licenses': [
                    {'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'u1'},
                    {'id': 4, 'name': 'Attribution License', 'url': 'u4'},
                ],
            }
        )
    )
    return ann, images_dir


def test_coco_local_selection_records_license_per_image(tmp_path: Path) -> None:
    ann, images_dir = _coco_fixture(tmp_path)
    picked = sel.select_coco(ann, images_dir, 8, seed=42)
    again = sel.select_coco(ann, images_dir, 8, seed=42)
    assert [c.path for c in picked] == [c.path for c in again]
    out = tmp_path / 'coco.txt'
    sel.write_manifest(out, picked, seed=42, today='2026-10-03')
    sidecar = sel.write_license_sidecar(out, picked)
    assert sidecar == tmp_path / 'coco.txt.licenses.csv'
    rows = list(csv.DictReader(sidecar.open()))
    assert [r['path'] for r in rows] == [str(c.path) for c in picked]
    for row in rows:
        want = (
            'Attribution License'
            if int(row['coco_id']) % 2 == 0
            else ('Attribution-NonCommercial-ShareAlike License')
        )
        assert row['license_name'] == want
        assert row['license_id'] in {'1', '4'}


def test_coco_license_filter(tmp_path: Path) -> None:
    ann, images_dir = _coco_fixture(tmp_path)
    picked = sel.select_coco(ann, images_dir, 100, seed=1, allowed_licenses={'Attribution License'})
    assert len(picked) == 10
    assert {c.license_name for c in picked} == {'Attribution License'}


def test_coco_url_mode_downloads_missing_images_with_injected_fetcher(tmp_path: Path) -> None:
    ann, _ = _coco_fixture(tmp_path)
    calls: list[str] = []

    def fake_download(url: str, dest: Path, **_kw: object) -> Path:
        calls.append(url)
        _img(dest, (640, 480))
        return dest

    picked = sel.select_coco(
        ann,
        tmp_path / 'dl',
        5,
        seed=42,
        image_url_template='https://example.invalid/val2017/{file_name}',
        downloader=fake_download,
    )
    assert len(picked) == 5
    assert len(calls) == 5
    assert all(c.startswith('https://example.invalid/val2017/') for c in calls)
    assert all(c.path.is_file() for c in picked)
