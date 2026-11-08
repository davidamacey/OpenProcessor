"""The partition routes: start a run, list partitions, activate with OCC (409 on a stale view)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from fixtures.job_rows import make_project

from src.config.curation_accessors import base_curation_config
from src.config.project_context import set_bound_project
from src.config.projects import ProjectRecord, resources_for_new
from src.core.db_session import install_database_error_handlers
from src.db.session import async_session_factory, project_tx
from src.routers.curation._project_deps import install_project_exception_handlers
from src.services.clustering.partitions import (
    PartitionMember,
    PartitionSpace,
    get_partition,
    list_partitions,
    mint_uids,
    record_partition,
)


pytestmark = [pytest.mark.db, pytest.mark.asyncio]
SPACE = PartitionSpace('pe_embedding', 'pe_core@v1', 'pe_core_v1')


async def _partition(project) -> uuid.UUID:
    async with async_session_factory()() as session, project_tx(session, project):
        (uid,) = await mint_uids(session, 1)
        return await record_partition(
            session,
            project=project,
            target='item',
            level='class',
            space=SPACE,
            method='ivf',
            params={},
            members=[PartitionMember('a', uid), PartitionMember('b', None, 'unassigned')],
        )


async def test_partitions_list_newest_first_with_counts_and_scope(pg_env, pg) -> None:
    mine, other = make_project(pg), make_project(pg)
    first, second = await _partition(mine), await _partition(mine)
    await _partition(other)
    async with async_session_factory()() as session, project_tx(session, mine):
        rows = await list_partitions(session, target='item')
        one = await get_partition(session, first)
        foreign = await get_partition(session, uuid.uuid4())
    assert [r.id for r in rows] == [second, first]
    assert one is not None
    assert (one.n_items, one.n_clusters, one.n_unassigned, one.serving) == (2, 1, 1, False)
    assert foreign is None


# --- the HTTP routes -------------------------------------------------------------------------


def _record(project: uuid.UUID, slug: str) -> ProjectRecord:
    now = dt.datetime.now(dt.UTC).isoformat()
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
    return dataclasses.replace(record, id=project)


@pytest_asyncio.fixture
async def http(pg, pg_env, job_env, producer, wakes, monkeypatch):
    from src.db.engine import dispose_engines
    from src.routers import curation as curation_package
    from src.routers.curation import cluster_partitions
    from src.routers.curation._project_deps import bind_path_project

    records: dict[str, ProjectRecord] = {}
    app = FastAPI()

    async def bind(project: str) -> None:
        set_bound_project(records[project])

    async def fake_space(target):
        return _StubSpace()

    monkeypatch.setattr(cluster_partitions, 'embedding_space', fake_space)
    app.dependency_overrides[bind_path_project] = bind
    app.include_router(
        curation_package.router,
        prefix='/curation/projects/{project}',
        dependencies=[Depends(bind_path_project)],
    )
    install_database_error_handlers(app)
    install_project_exception_handlers(app)
    app.state.records = records
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
        c.records = records  # type: ignore[attr-defined]
        yield c
    await dispose_engines()


class _StubSpace:
    vector_field = 'pe_embedding'


def _project(http, pg, slug: str) -> uuid.UUID:
    pid = make_project(pg, slug)
    http.records[slug] = _record(pid, slug)
    return pid


async def test_run_start_is_202_then_409_job_busy(http, pg) -> None:
    _project(http, pg, 'rt-run')
    url = '/curation/projects/rt-run/clusters/runs'
    first = await http.post(url, json={'target': 'item'})
    assert first.status_code == 202
    assert uuid.UUID(first.json()['job_id'])
    second = await http.post(url, json={'target': 'item'})
    assert second.status_code == 409
    assert second.json()['detail']['error'] == 'job_busy'


async def test_activation_is_202_and_a_stale_view_is_409(http, pg) -> None:
    project = _project(http, pg, 'rt-act')
    first, second = await _partition(project), await _partition(project)
    url = '/curation/projects/rt-act/clusters/activation'
    ok = await http.put(url, json={'target': 'item', 'partition_id': str(first)})
    assert ok.status_code == 202
    assert ok.json()['revision'] == 1
    stale = await http.put(
        url,
        json={
            'target': 'item',
            'partition_id': str(second),
            'expected_serving_partition_id': str(second),
            'expected_revision': 1,
        },
    )
    assert stale.status_code == 409
    assert stale.json()['detail']['error'] == 'partition_conflict'
    listed = await http.get('/curation/projects/rt-act/clusters/partitions')
    assert listed.status_code == 200
    assert {p['id'] for p in listed.json()['partitions']} == {str(first), str(second)}
