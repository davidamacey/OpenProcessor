"""P3F item 6: the registry refresh must not silently truncate past
OpenSearch's default 1000-hit result window -- ``_refresh`` pages via
``search_after`` on ``slug`` (``_REFRESH_PAGE_SIZE`` well under 1000),
so seeding >1000 fake project docs and refreshing must return every one.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config.curation import base_curation_config
from src.config.projects import resources_for_new
from src.services.projects.registry import ProjectRegistry, record_to_doc, set_project_registry

from .conftest import FakeRegistryOpenSearch


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import src.config.curation as curation_mod

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    curation_mod._default_curation_config = None
    set_project_registry(None)
    yield
    curation_mod._default_curation_config = None
    set_project_registry(None)


def _seed_record(client: FakeRegistryOpenSearch, slug: str) -> None:
    from datetime import UTC, datetime

    from src.config.projects import ProjectRecord

    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
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
    client.docs[f'project:{slug}'] = record_to_doc(record)
    client.seq[f'project:{slug}'] = 0


def test_refresh_returns_more_than_the_1000_hit_default_window() -> None:
    async def _run() -> int:
        client = FakeRegistryOpenSearch()
        n = 1200
        for i in range(n):
            _seed_record(client, f'proj-{i:05d}')
        # Bump the counter doc so ensure_fresh() sees a change and refreshes.
        client.docs['meta:projects_revision'] = {'revision': 1}
        client.seq['meta:projects_revision'] = 0
        client._refresh_all()  # test seeds client.docs directly, bypassing index()

        registry = ProjectRegistry(lambda: client)
        await registry.refresh_strict()
        return len(registry.snapshot())

    assert asyncio.run(_run()) == 1200
