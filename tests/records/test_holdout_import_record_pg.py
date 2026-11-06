"""An import freezes its test split as its own record; the curated freeze is never replaced,
and an imported holdout counts as an existing holdout for the curated freeze."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.project_context import current_project_id
from src.db.repos.records import current_holdout_freeze, freeze_holdout, list_holdout_freezes
from src.db.session import project_tx
from src.services.curation.holdout import build_cohort_query, forget_freeze, record_standing_freeze


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


pytest_plugins = ['_records_world']


async def _curated(session: AsyncSession) -> Any:
    project_id = current_project_id()
    async with project_tx(session, project_id):
        return await freeze_holdout(
            session, project_id, source='manual', item_count=2, frozen_by='api'
        )


async def _import_freeze(session: AsyncSession, ids: list[str]) -> Any:
    return await record_standing_freeze(
        session,
        crop_ids=ids,
        cohort_spec={'kind': 'import'},
        per_class_counts={'0': len(ids)},
        source='import:imp1',
        frozen_by='import',
    )


async def _state(session: AsyncSession) -> tuple[Any, list[Any]]:
    project_id = current_project_id()
    async with project_tx(session, project_id):
        return (
            await current_holdout_freeze(session, project_id),
            await list_holdout_freezes(session, project_id),
        )


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_import_freeze_is_recorded_and_leaves_the_current_freeze_alone(session) -> None:
    curated = await _curated(session)

    row = await _import_freeze(session, ['c', 'd', 'e'])

    current, history = await _state(session)
    assert current.id == curated.id
    assert {h.id for h in history} == {curated.id, row.id}
    assert (row.is_current, row.item_count, row.source) == (False, 3, 'import:imp1')
    assert '/holdout/imports/' in row.snapshot_storage_key


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_import_freeze_with_no_prior_curated_freeze_creates_no_current_row(
    session,
) -> None:
    await _import_freeze(session, ['a'])
    current, history = await _state(session)
    assert current is None
    assert len(history) == 1


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_snapshot_blob_holds_the_frozen_ids_and_sha(session) -> None:
    from src.clients.storage.roots import build_rooted_store
    from src.config.storage import storage_root_paths

    row = await _import_freeze(session, ['b', 'a'])
    with build_rooted_store(storage_root_paths()).open_read(row.snapshot_storage_key) as handle:
        record = json.load(handle)
    assert (record['kind'], record['n_frozen'], record['crop_ids']) == ('import', 2, ['a', 'b'])


@pytest.mark.db
@pytest.mark.asyncio
async def test_undoing_an_import_drops_its_row_and_never_the_current_freeze(session) -> None:
    curated = await _curated(session)
    row = await _import_freeze(session, ['a', 'b'])

    assert await forget_freeze(session, row.id) is True
    assert await forget_freeze(session, curated.id) is False
    current, history = await _state(session)
    assert current.id == curated.id
    assert [h.id for h in history] == [curated.id]


def test_the_curated_cohort_never_samples_imported_labels() -> None:
    filters = build_cohort_query()['bool']['filter']
    assert {'term': {'class_source': 'human'}} in filters


@pytest.fixture
def app_client(world):
    from _curation_app import mount_curation_routers

    from src.core.dependencies import get_opensearch
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = AsyncMock()
    fake.indices = AsyncMock()
    fake.indices.exists = AsyncMock(return_value=True)
    fake.indices.create = AsyncMock(return_value={'acknowledged': True})
    fake.indices.refresh = AsyncMock(return_value={'_shards': {}})
    fake.indices.put_mapping = AsyncMock(return_value={'acknowledged': True})
    fake.search = AsyncMock(return_value={'hits': {'hits': [], 'total': {'value': 0}}})
    fake.bulk = AsyncMock(return_value={'errors': False, 'items': []})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[get_opensearch] = lambda: fake
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        client.fake_os = fake  # type: ignore[attr-defined]
        yield client


@pytest.mark.db
def test_the_curated_freeze_refuses_while_an_imported_holdout_exists(app_client: Any) -> None:
    app_client.fake_os.count = AsyncMock(return_value={'count': 41})
    url = '/curation/projects/default/test_holdout/freeze'
    refused = app_client.post(url, json={'percent': 20})
    assert refused.status_code == 409
    # Any flagged item counts, whatever wrote the flag.
    query = app_client.fake_os.count.call_args.kwargs['body']['query']
    assert query == {'term': {'test_holdout': True}}
    # force=true gets past the guard (this cohort is empty, so 422, not 409).
    forced = app_client.post(url + '?force=true', json={'percent': 20})
    assert forced.status_code == 422
