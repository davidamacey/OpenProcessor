"""RootedStore routes keys to the physical roots; the source root is read only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.clients.storage.base import BlobStore, ReadOnlyRootError
from src.clients.storage.roots import build_rooted_store
from src.config import curation_accessors
from src.config.storage import StorageRootPaths, storage_root_paths


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def paths(tmp_path: Path) -> StorageRootPaths:
    return StorageRootPaths(
        source=tmp_path / 'src',
        projects=tmp_path / 'proj',
        models=tmp_path / 'models',
        cache=tmp_path / 'cache',
        upgrade=tmp_path / 'upgrade',
    )


def test_routes_by_first_segment(paths: StorageRootPaths) -> None:
    store = build_rooted_store(paths)
    assert isinstance(store, BlobStore)
    store.write_atomic('projects/cats/uploads/a.jpg', b'1')
    assert (paths.projects / 'cats' / 'uploads' / 'a.jpg').read_bytes() == b'1'
    assert store.list_prefix('projects/cats') == ['projects/cats/uploads/a.jpg']
    assert store.local_path('projects/cats/uploads/a.jpg') == paths.projects / 'cats/uploads/a.jpg'
    assert store.delete_prefix('projects/cats') == 1


def test_source_root_is_read_only(paths: StorageRootPaths) -> None:
    paths.source.mkdir()
    (paths.source / 'x.jpg').write_bytes(b'z')
    store = build_rooted_store(paths)
    assert store.exists('source/x.jpg')
    with pytest.raises(ReadOnlyRootError):
        store.write_atomic('source/y.jpg', b'1')
    with pytest.raises(ReadOnlyRootError):
        store.delete_prefix('source/x.jpg')
    assert store.exists('source/x.jpg')


def test_a_bare_root_is_not_an_object(paths: StorageRootPaths) -> None:
    with pytest.raises(ValueError, match='names a root'):
        build_rooted_store(paths).exists('projects')


def test_paths_come_from_the_existing_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(curation_accessors, '_default_curation_config', None)
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'p'))
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path / 'm'))
    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(tmp_path / 'c'))
    monkeypatch.setenv('OP_SOURCE_ROOT', str(tmp_path / 's'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'st'))
    got = storage_root_paths()
    assert got.projects == tmp_path / 'p'
    assert got.models == tmp_path / 'm'
    assert got.cache == tmp_path / 'c'
    assert got.source == tmp_path / 's'
    assert got.upgrade == tmp_path / 'st' / 'upgrade'
