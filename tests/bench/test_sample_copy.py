"""Seeded NVMe sample copies for the ingest profile (``scripts/bench/sample_copy.py``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from scripts.bench import sample_copy as sc, select_baseline_set as sel


@pytest.fixture
def source_manifest(tmp_path: Path) -> Path:
    src = tmp_path / 'private-folder-name'
    src.mkdir()
    cands = []
    for i in range(6):
        p = src / f'IMG_{i}.jpg'
        Image.new('RGB', (40 + i, 30), (i, i, i)).save(p, 'JPEG')
        cands.append(sel.Candidate(p, p.stat().st_size, 'unit'))
    out = tmp_path / 'src.txt'
    sel.write_manifest(out, cands, seed=7, today='2026-10-29')
    return out


def _digest(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_files_are_copied_with_neutral_names_and_identical_bytes(
    source_manifest: Path, tmp_path: Path
) -> None:
    dest, out = tmp_path / 'nvme', tmp_path / 'copy.txt'
    result = sc.copy_sample(source_manifest, dest, out, seed=7)

    originals = sel.read_manifest(source_manifest)
    copies = sel.read_manifest(out)
    assert len(copies) == 6
    assert [Path(c).name for c in copies] == [f'{i:06d}.jpg' for i in range(6)]
    assert [_digest(c) for c in copies] == [_digest(o) for o in originals]
    assert result == {'count': 6, 'bytes': sum(Path(c).stat().st_size for c in copies)}
    for c in copies:
        assert Path(c).stat().st_nlink == 1  # a copy, never a hard link
        assert Path(c).stat().st_ino not in {Path(o).stat().st_ino for o in originals}


def test_the_sources_are_left_in_place(source_manifest: Path, tmp_path: Path) -> None:
    originals = sel.read_manifest(source_manifest)
    sc.copy_sample(source_manifest, tmp_path / 'd', tmp_path / 'c.txt', seed=7)
    assert all(Path(o).is_file() for o in originals)


def test_the_new_manifest_names_no_source_path(source_manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / 'c.txt'
    sc.copy_sample(source_manifest, tmp_path / 'd', out, seed=7)
    text = out.read_text()
    assert 'private-folder-name' not in text
    assert 'IMG_' not in text
    assert '# seed: 7' in text


def test_a_non_empty_destination_is_refused(source_manifest: Path, tmp_path: Path) -> None:
    dest = tmp_path / 'd'
    dest.mkdir()
    (dest / 'old.jpg').write_bytes(b'x')
    with pytest.raises(FileExistsError, match='not empty'):
        sc.copy_sample(source_manifest, dest, tmp_path / 'c.txt', seed=7)


def test_a_corrupted_copy_fails_the_run(
    source_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = sc.shutil.copyfile

    def flaky(src: str, dst: str) -> str:
        real(src, dst)
        with Path(dst).open('ab') as fh:
            fh.write(b'\0')
        return dst

    monkeypatch.setattr(sc.shutil, 'copyfile', flaky)
    with pytest.raises(OSError, match='differs from its source'):
        sc.copy_sample(source_manifest, tmp_path / 'd', tmp_path / 'c.txt', seed=7)
