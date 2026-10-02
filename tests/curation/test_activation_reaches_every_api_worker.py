"""Found live (32 API workers): after activating a region profile, ingest on
the other workers kept seeding no region work for the whole run -- those
workers only refreshed their config snapshot when a config route landed on
them, and no ingest route does."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.dependencies import get_curation_opensearch
from src.services.config_store.index import activate, save_config
from src.services.config_store.store import get_config_store, reset_config_stores


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    reset_config_stores()
    yield
    reset_config_stores()


@pytest.mark.asyncio
async def test_a_project_route_sees_an_activation_made_through_another_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeConfigOpenSearch()
    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug='alpha',
        display_name='alpha',
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new('alpha', base_curation_config()),
    )

    async def _client() -> FakeConfigOpenSearch:
        return client

    monkeypatch.setattr('src.services.projects.guard.make_curation_opensearch', _client)

    with bind_project(record):
        store = get_config_store()
        await store.refresh(client)  # this worker's snapshot, taken before the activation
        doc = await save_config(
            client,
            store.index,
            kind='region_profile',
            name='wheel',
            body={},
            expected_revision=None,
        )
        await activate(
            client,
            store.index,
            axis='detection_profile',
            name='wheel',
            revision=doc['revision'],
            expected_active=None,
        )
        # ...and has been idle for a while, as an unlucky worker's would be.
        store.current = replace(store.current, loaded_at=time.monotonic() - 30)
        assert store.current.active_profile is None

        await get_curation_opensearch()

        assert store.current.active_profile == ('wheel', doc['revision'])
