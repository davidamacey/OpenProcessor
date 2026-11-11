"""The workflow facts read from PostgreSQL decide the Workspace default view."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg
import pytest
from _research_seed import make_partition, make_project
from curation.query_fakes import QueryFakeOpenSearch

from src.db.session import async_session_factory
from src.services.curation.class_counts import ClassCounts
from src.services.curation.workflow_steps import build_steps, gather_facts


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


async def _steps(project: uuid.UUID):
    async with async_session_factory()() as session:
        facts = await gather_facts(session, project, QueryFakeOpenSearch({}), [], ClassCounts())
    return build_steps(facts)


@pytest.mark.asyncio
async def test_workflow_steps_default_view_is_gallery_without_a_serving_partition(
    pg_env, conn
) -> None:
    project = make_project(conn, f'p{uuid.uuid4().hex[:12]}')
    make_partition(conn, project, status='ready')
    result = await _steps(project)
    assert result.workspace_default_view == 'gallery'
    assert {s.id: s.done for s in result.steps}['cluster'] is True
    assert {s.id: s.done for s in result.steps}['activate'] is False


@pytest.mark.asyncio
async def test_workflow_steps_default_view_is_clusters_with_a_serving_partition(
    pg_env, conn
) -> None:
    project = make_project(conn, f'p{uuid.uuid4().hex[:12]}')
    partition = make_partition(conn, project, status='ready')
    conn.execute(
        'INSERT INTO cluster_activations (project_id, target, level, embedding_field, '
        "serving_partition_id, updated_by) VALUES (%s, 'item', 'class', 'f', %s, 'test')",
        (project, partition),
    )
    assert (await _steps(project)).workspace_default_view == 'clusters'
