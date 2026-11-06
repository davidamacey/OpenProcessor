"""Deleting a project removes every area under it, including one no code lists yet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.storage.roots import build_rooted_store
from src.config.storage import StorageRootPaths


if TYPE_CHECKING:
    from pathlib import Path


LISTED_AREAS = ('uploads', 'exports', 'bakeoff_eval', 'umap', 'holdout', 'training', 'label_ops')


def _store(tmp_path: Path):
    return build_rooted_store(
        StorageRootPaths(
            source=tmp_path / 's',
            projects=tmp_path / 'p',
            models=tmp_path / 'm',
            cache=tmp_path / 'c',
            upgrade=tmp_path / 'u',
        )
    )


def test_unlisted_spatial_area_is_removed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_atomic('projects/cats/spatial/colmap/db.bin', b'1')
    store.write_atomic('projects/cats/spatial/scene/1.json', b'1')
    store.write_atomic('projects/dogs/spatial/colmap/db.bin', b'1')
    assert store.delete_prefix('projects/cats/') == 2
    assert store.list_prefix('projects/cats') == []
    assert store.list_prefix('projects/dogs') == ['projects/dogs/spatial/colmap/db.bin']


def test_listed_areas_are_removed_as_before(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for area in LISTED_AREAS:
        store.write_atomic(f'projects/cats/{area}/f.bin', b'1')
    assert store.delete_prefix('projects/cats') == len(LISTED_AREAS)
    assert not (tmp_path / 'p' / 'cats').exists()
