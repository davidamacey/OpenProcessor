"""Every answered VLM call is added to ``vlm_usage_hourly`` under the labeler's usage source."""

from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
import pytest_asyncio


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'db'))
from _research_seed import make_project

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import new_project_record
from src.db.engine import dispose_engines
from src.db.session import async_session_factory, project_tx
from src.domain.vlm_usage import crops_in_payload, usage_from_response
from src.services.labeling.vlm_usage import record_call, usage_sessions


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def world(pg_env: PgDatabase) -> AsyncIterator[dict[str, Any]]:
    with psycopg.connect(pg_env.conninfo('super'), autocommit=True) as conn:
        project = make_project(conn, f'usage-{uuid.uuid4().hex[:6]}')
        name = f'ep-{uuid.uuid4().hex[:6]}'
        doc = conn.execute(
            'INSERT INTO config_documents (project_id, kind, name, current_revision, updated_by) '
            "VALUES (NULL, 'vlm_endpoint', %s, 1, 'test') RETURNING id",
            (name,),
        ).fetchone()[0]
        conn.execute(
            'INSERT INTO config_revisions (config_id, revision, body, body_sha256, created_by) '
            "VALUES (%s, 1, '{}', 'x', 'test')",
            (doc,),
        )
        slug = conn.execute('SELECT slug FROM projects WHERE id = %s', (project,)).fetchone()[0]
    yield {'project': project, 'slug': slug, 'pg': pg_env, 'endpoint': name}
    await dispose_engines()


def _payload(crops: int) -> dict[str, Any]:
    parts = [{'type': 'text', 'text': 'q'}] + [{'type': 'image_url'}] * crops
    return {'messages': [{'role': 'user', 'content': parts}]}


def _rows(world: dict[str, Any]) -> list[tuple[Any, ...]]:
    with psycopg.connect(world['pg'].conninfo('super')) as conn:
        return conn.execute(
            'SELECT source, requests, crops, prompt_tokens, completion_tokens, usage_missing '
            'FROM vlm_usage_hourly WHERE project_id = %s ORDER BY source',
            (world['project'],),
        ).fetchall()


async def _record(world: dict[str, Any], source: str | None, crops: int, response: dict) -> None:
    record = replace(new_project_record(world['slug'], base_curation_config()), id=world['project'])

    @asynccontextmanager
    async def provider() -> AsyncIterator[Any]:
        async with async_session_factory()() as session, project_tx(session, world['project']):
            yield session

    with bind_project(record), usage_sessions(provider):
        await record_call(
            endpoint_name=world['endpoint'],
            source=source,
            payload=_payload(crops),
            response=response,
        )


@pytest.mark.asyncio
async def test_ten_crops_of_an_assist_run_add_ten_crops_under_that_source(world) -> None:
    response = {'usage': {'prompt_tokens': 120, 'completion_tokens': 30}}
    await _record(world, 'assist_run', 10, response)
    assert _rows(world) == [('assist_run', 1, 10, 120, 30, 0)]


@pytest.mark.asyncio
async def test_calls_add_up_within_the_hour_and_stay_apart_by_source(world) -> None:
    await _record(world, 'autolabel', 2, {'usage': {'prompt_tokens': 5, 'completion_tokens': 1}})
    await _record(world, 'autolabel', 3, {'usage': {'prompt_tokens': 7, 'completion_tokens': 2}})
    await _record(world, 'label_batch', 1, {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    assert _rows(world) == [('autolabel', 2, 5, 12, 3, 0), ('label_batch', 1, 1, 1, 1, 0)]


@pytest.mark.asyncio
async def test_a_response_without_usage_is_counted_as_missing(world) -> None:
    await _record(world, 'autolabel', 4, {'choices': []})
    assert _rows(world) == [('autolabel', 1, 4, 0, 0, 1)]


@pytest.mark.asyncio
async def test_a_probe_has_no_source_and_records_nothing(world) -> None:
    await _record(world, None, 1, {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    assert _rows(world) == []


def test_usage_and_crops_are_read_from_the_wire_shapes() -> None:
    assert usage_from_response({'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}).present
    assert not usage_from_response({'usage': {'prompt_tokens': 3}}).present
    assert not usage_from_response(
        {'usage': {'prompt_tokens': True, 'completion_tokens': 1}}
    ).present
    assert crops_in_payload(_payload(3)) == 3


@pytest.mark.asyncio
async def test_an_unknown_source_is_refused(world) -> None:
    with pytest.raises(ValueError, match='unknown vlm usage source'):
        await _record(world, 'nope', 1, {})
    assert _rows(world) == []
