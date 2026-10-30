"""Tests for scripts/datasets/prepare_labelled_benchmark.py: label rows, archive safety, manifest verify."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from scripts.datasets import prepare_labelled_benchmark as prep


if TYPE_CHECKING:
    from pathlib import Path


def _rec(label: int) -> dict[str, Any]:
    return {'label': label, 'image': {'bytes': b'jpg%d' % label}}


def test_cars_split_is_by_class_not_by_mirror_split(tmp_path: Path) -> None:
    names = [f'make model {i}' for i in range(196)]
    # class 97 is the last train class, class 98 the first test class; the mirror's own split is the
    # opposite for one of them, which must not matter
    data = {'train': [_rec(0), _rec(98)], 'test': [_rec(97), _rec(195)]}
    rows = prep.cars_rows(tmp_path, names, lambda s: data[s])
    by = {(r['class_id']): r for r in rows}
    assert [by[c]['dml_split'] for c in (0, 97, 98, 195)] == ['train', 'train', 'test', 'test']
    assert by[98]['orig_split'] == 'train'
    assert (tmp_path / by[195]['path']).read_bytes() == b'jpg195'
    assert len({r['path'] for r in rows}) == 4


def test_sop_flattens_paths_and_refuses_traversal(tmp_path: Path) -> None:
    rec = {
        'image_path': 'Stanford_Online_Products/bike_final/11_0.JPG',
        'item_id': '11',
        'category': 'bicycle',
        'image': {'bytes': b'x'},
    }
    rows = prep.sop_rows(tmp_path, lambda s: [rec] if s == 'test' else [])
    assert rows[0]['split'] == 'test'
    assert '/' not in rows[0]['path'].removeprefix('images/')
    assert (tmp_path / rows[0]['path']).exists()
    bad = {**rec, 'image_path': '../../etc/passwd'}
    with pytest.raises(prep.PrepareError):
        prep.sop_rows(tmp_path, lambda s: [bad] if s == 'train' else [])


def test_cub_split_boundary_is_class_100(tmp_path: Path) -> None:
    (tmp_path / 'classes.txt').write_text('100 100.a\n101 101.b\n')
    (tmp_path / 'images.txt').write_text('1 100.a/x.jpg\n2 101.b/y.jpg\n')
    (tmp_path / 'image_class_labels.txt').write_text('1 100\n2 101\n')
    (tmp_path / 'train_test_split.txt').write_text('1 0\n2 1\n')
    rows = prep.cub_rows(tmp_path)
    assert [r['dml_split'] for r in rows] == ['train', 'test']
    assert [r['orig_split'] for r in rows] == ['test', 'train']
    assert rows[1]['class_name'] == '101.b'


def test_vmmr_model_keeps_underscores(tmp_path: Path) -> None:
    p = tmp_path / 'extracted' / 'ford_f_150_2004' / 'a.jpg'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'x')
    row = prep.vmmr_row(p, tmp_path)
    assert (row['make'], row['model'], row['year']) == ('ford', 'f_150', '2004')


def test_lfw_identity_strips_only_the_trailing_index(tmp_path: Path) -> None:
    d = tmp_path / 'imgs'
    d.mkdir()
    (d / 'Ana_Maria_0002.jpg').write_bytes(b'x')
    rows = prep.lfw_variant_rows(d, tmp_path)
    assert rows[0]['identity'] == 'Ana_Maria'


def test_zip_slip_member_is_refused(tmp_path: Path) -> None:
    z = tmp_path / 'a.zip'
    with zipfile.ZipFile(z, 'w') as f:
        f.writestr('../evil.jpg', b'x')
    with pytest.raises(prep.PrepareError):
        prep.extract_zip(z, tmp_path / 'out')
    assert not (tmp_path / 'evil.jpg').exists()


def _prepared(tmp_path: Path) -> Path:
    root = tmp_path / 'ds'
    root.mkdir()
    (root / 'img1.jpg').write_bytes(b'one')
    (root / 'img2.jpg').write_bytes(b'two!')
    prep.write_jsonl([{'path': 'img1.jpg'}, {'path': 'img2.jpg'}], root / 'labels.jsonl')
    prep.write_manifest('calfw', root, {'images': 2})
    return root


def test_verify_accepts_a_consistent_directory(tmp_path: Path) -> None:
    root = _prepared(tmp_path)
    assert prep.verify(root, check_hash=True) == []
    assert (
        json.loads((root / 'manifest.json').read_text())['file_count'] == 3
    )  # two images + labels


def test_verify_reports_missing_unlisted_and_changed_files(tmp_path: Path) -> None:
    root = _prepared(tmp_path)
    (root / 'img1.jpg').unlink()
    (root / 'extra.jpg').write_bytes(b'x')
    (root / 'img2.jpg').write_bytes(b'two?')  # same size, different content
    problems = prep.verify(root)
    (root / 'extra2.jpg').unlink(missing_ok=True)
    assert any(p == 'missing: img1.jpg' for p in problems)
    assert any(p == 'unlisted: extra.jpg' for p in problems)
    assert not any('img2' in p for p in problems if p.startswith('size'))
    assert any(p == 'sha256 differs: img2.jpg' for p in prep.verify(root, check_hash=True))


def test_verify_flags_label_rows_without_a_listed_file(tmp_path: Path) -> None:
    root = _prepared(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    prep.write_jsonl([{'path': 'img1.jpg'}, {'path': 'ghost.jpg'}], root / 'labels.jsonl')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    problems = prep.verify(root)
    assert 'label row without file: ghost.jpg' in problems


def test_verify_flags_a_changed_size_without_hashing(tmp_path: Path) -> None:
    root = _prepared(tmp_path)
    (root / 'img2.jpg').write_bytes(b'much longer than before')
    assert 'size differs: img2.jpg' in prep.verify(root)
