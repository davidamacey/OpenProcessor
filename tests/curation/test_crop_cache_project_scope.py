"""The tmpfs crop cache is keyed per project (#359): content-derived crop ids repeat across
projects, so a project's cache files live under ``<crop_cache_dir>/<slug>/`` and deleting a
project removes exactly that directory."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

import src.config.curation_accessors as curation_mod
from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.curation import crop_bytes
from src.services.curation.item_delete import delete_items
from src.services.curation.source_image_cache import (
    project_crop_cache_dir,
    prune_crop_cache,
    write_crop_cache,
)
from src.services.projects.delete_paths import _delete_dirs, _validate_delete_paths


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'crop_cache'
    root.mkdir()
    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(root))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(tmp_path / 'jobs'))
    monkeypatch.setattr(curation_mod, '_default_curation_config', None)
    yield root
    monkeypatch.setattr(curation_mod, '_default_curation_config', None)


def _record(slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


def _cache_crop(root: Path, slug: str, crop_id: str, data: bytes) -> None:
    with bind_project(_record(slug)):
        project_dir = project_crop_cache_dir(root)
        assert project_dir is not None
        write_crop_cache(crop_id, data, project_dir)


def test_the_same_crop_id_is_cached_separately_per_project(cache_root: Path) -> None:
    _cache_crop(cache_root, 'alpha', 'shared-id', b'alpha-bytes')
    _cache_crop(cache_root, 'beta', 'shared-id', b'beta-bytes')

    with bind_project(_record('alpha')):
        assert crop_bytes.crop_jpeg_from_cache('shared-id') == b'alpha-bytes'
    with bind_project(_record('beta')):
        assert crop_bytes.crop_jpeg_from_cache('shared-id') == b'beta-bytes'
    with bind_project(_record('gamma')):
        assert crop_bytes.crop_jpeg_from_cache('shared-id') is None  # a miss re-crops


def test_deleting_a_project_removes_its_cache_files_and_only_those(cache_root: Path) -> None:
    for crop_id in ('c1', 'c2', 'shared-id'):
        _cache_crop(cache_root, 'alpha', crop_id, b'alpha')
    _cache_crop(cache_root, 'beta', 'shared-id', b'beta')

    alpha = _record('alpha')
    _validate_delete_paths(alpha)
    asyncio.run(_delete_dirs(alpha))

    with bind_project(_record('alpha')):
        assert [p for p in cache_root.rglob('*.jpg') if p.parent.name == 'alpha'] == []
        assert crop_bytes.crop_jpeg_from_cache('shared-id') is None
    with bind_project(_record('beta')):
        assert crop_bytes.crop_jpeg_from_cache('shared-id') == b'beta'


def test_deleting_items_drops_only_the_bound_projects_cached_crop(cache_root: Path) -> None:
    _cache_crop(cache_root, 'alpha', 'shared-id', b'alpha')
    _cache_crop(cache_root, 'beta', 'shared-id', b'beta')

    class _Gone:
        async def mget(self, **_kw: object) -> dict[str, object]:
            return {'docs': []}  # already absent: the cached crop is dropped

    async def _deletable(_cid: str, _src: dict[str, object]) -> bool:
        return True

    with bind_project(_record('alpha')):
        asyncio.run(
            delete_items(
                _Gone(),  # type: ignore[arg-type]
                ['shared-id'],
                items_index='items',
                crop_cache_dir=cache_root,
                deletable=_deletable,
            )
        )
        assert crop_bytes.crop_jpeg_from_cache('shared-id') is None
    with bind_project(_record('beta')):
        assert crop_bytes.crop_jpeg_from_cache('shared-id') == b'beta'


def test_the_size_cap_prunes_across_every_project_directory(cache_root: Path) -> None:
    for slug in ('alpha', 'beta'):
        for i in range(10):
            _cache_crop(cache_root, slug, f'c{i}', b'x' * 1024)

    removed, _ = prune_crop_cache(cache_root, 10 * 1024)

    assert removed > 0
    assert sum(p.stat().st_size for p in cache_root.rglob('*.jpg')) <= 9 * 1024
