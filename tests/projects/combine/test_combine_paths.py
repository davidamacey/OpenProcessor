"""Where a source image lands in the target: upload-root files move to the
same relative place under the target's upload root, and nothing climbs out."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.projects.combine.copy_docs import target_image_path


if TYPE_CHECKING:
    from pathlib import Path


def test_an_upload_root_file_moves_under_the_target_root(tmp_path: Path) -> None:
    src, tgt = tmp_path / 'up' / 'a', tmp_path / 'up' / 'b'
    image = src / 'ab' / 'x.jpg'
    path, link = target_image_path(str(image), source_upload_root=src, target_upload_root=tgt)
    assert path == str(tgt / 'ab' / 'x.jpg')
    assert link == image.resolve()


def test_a_dotdot_path_cannot_escape_the_target_root(tmp_path: Path) -> None:
    src, tgt = tmp_path / 'up' / 'a', tmp_path / 'up' / 'b'
    (tmp_path / 'up' / 'elsewhere').mkdir(parents=True)
    src.mkdir()
    tgt.mkdir()
    (tmp_path / 'up' / 'elsewhere' / 'x.jpg').write_bytes(b'x')
    raw = f'{src}/../elsewhere/x.jpg'
    path, link = target_image_path(raw, source_upload_root=src, target_upload_root=tgt)
    assert link is None
    assert path == raw


def test_a_dotdot_path_that_stays_inside_the_root_is_judged_by_where_it_lands(
    tmp_path: Path,
) -> None:
    src, tgt = tmp_path / 'up' / 'a', tmp_path / 'up' / 'b'
    raw = f'{src}/ab/../cd/x.jpg'
    path, link = target_image_path(raw, source_upload_root=src, target_upload_root=tgt)
    assert path == str(tgt / 'cd' / 'x.jpg')
    assert link == (src / 'cd' / 'x.jpg').resolve()
