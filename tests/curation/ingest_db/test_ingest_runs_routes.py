"""``/ingest/bulk*`` through the real routes over PostgreSQL: a start queues one
``bulk_ingest_run`` job per project, the run view is the job's, and the failures are the
source queue's rows. The worker side is covered by ``tests/tasks/test_families_c_ingest.py``."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_rows import project_session, sql

from curation.ingest_db.app_fixtures import project, url
from src.db.repos import ingest_sources as repo


if TYPE_CHECKING:
    import httpx

pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures(
        'bound_async_engine', 'api_async_engine', 'job_env', 'producer', 'wakes'
    ),
]


async def start(client: httpx.AsyncClient, slug: str, **body: Any) -> httpx.Response:
    return await client.post(url(slug, 'bulk'), json=body)


@pytest.mark.asyncio
async def test_config_serves_the_roots_and_run_bounds(app, pg, client, served_root) -> None:
    project(app, pg, 'rc-one')
    body = (await client.get(url('rc-one', 'bulk/config'))).json()
    assert body['source_roots'] == [str(served_root.resolve())]
    assert body['default_extensions'] == ['.jpg', '.jpeg', '.png']
    assert 1 <= body['default_batch_size'] <= body['max_batch_size']


@pytest.mark.asyncio
async def test_a_folder_start_queues_one_run_with_the_source_snapshot(
    app, pg, client, served_root, producer
) -> None:
    pid = project(app, pg, 'rs-one')
    (served_root / 'a').mkdir()

    reply = await start(client, 'rs-one', folder=str(served_root / 'a'), source='census')

    assert reply.status_code == 202, reply.text
    run = reply.json()
    assert (run['status'], run['resumed'], run['can_cancel']) == ('queued', False, True)
    assert run['options']['source_tag'] == 'census'
    assert run['source'] == {
        'kind': 'folder',
        'path': str(served_root / 'a'),
        'extensions': ['.jpg', '.jpeg', '.png'],
        'label': None,
    }
    assert producer.sent() == [run['run_id']]
    rows = sql(pg, 'SELECT kind, project_id FROM jobs WHERE id = %s', uuid.UUID(run['run_id']))
    assert rows == [('bulk_ingest_run', pid)]
    # the one-shot source of the folder is the run's queue
    sources = sql(pg, 'SELECT count(*) FROM ingest_sources WHERE project_id = %s', pid)
    assert sources == [(1,)]


@pytest.mark.asyncio
async def test_a_second_start_while_one_is_live_is_a_409_naming_the_run(
    app, pg, client, served_root
) -> None:
    project(app, pg, 'rb-one')
    (served_root / 'a').mkdir()
    (served_root / 'b').mkdir()
    live = (await start(client, 'rb-one', folder=str(served_root / 'a'))).json()

    for folder in ('a', 'b'):  # the same source and another one: one run per project
        reply = await start(client, 'rb-one', folder=str(served_root / folder))
        assert reply.status_code == 409
        detail = reply.json()['detail']
        assert (detail['error'], detail['run_id']) == ('bulk_ingest_busy', live['run_id'])


@pytest.mark.asyncio
async def test_cancel_stops_a_live_run_and_a_cancelled_one_is_resumed_by_a_new_start(
    app, pg, client, served_root
) -> None:
    project(app, pg, 'rx-one')
    (served_root / 'a').mkdir()
    first = (await start(client, 'rx-one', folder=str(served_root / 'a'))).json()

    cancelled = await client.post(url('rx-one', f'bulk/{first["run_id"]}/cancel'))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()['status'] == 'cancelled'
    assert cancelled.json()['can_resume'] is True
    again = await client.post(url('rx-one', f'bulk/{first["run_id"]}/cancel'))
    assert again.json()['status'] == 'cancelled'  # a finished run keeps its status

    second = (await start(client, 'rx-one', folder=str(served_root / 'a'))).json()
    assert second['run_id'] != first['run_id']
    assert second['resumed'] is True


@pytest.mark.asyncio
async def test_unknown_runs_and_other_projects_runs_are_404(app, pg, client, served_root) -> None:
    project(app, pg, 'rn-one')
    project(app, pg, 'rn-two')
    (served_root / 'a').mkdir()
    mine = (await start(client, 'rn-one', folder=str(served_root / 'a'))).json()

    for run_id in (mine['run_id'], str(uuid.uuid4()), 'bulk_20260101T000000_deadbeef'):
        for tail in ('', '/failures'):
            reply = await client.get(url('rn-two', f'bulk/{run_id}{tail}'))
            assert reply.status_code == 404, (run_id, tail)
            assert reply.json()['detail']['error'] == 'bulk_ingest_not_found'
    assert (await client.post(url('rn-two', f'bulk/{mine["run_id"]}/cancel'))).status_code == 404
    assert (await client.get(url('rn-two', 'bulk'))).json()['total'] == 0


@pytest.mark.asyncio
async def test_the_run_list_is_newest_first(app, pg, client, served_root) -> None:
    project(app, pg, 'rl-one')
    ids = []
    for name in ('a', 'b'):
        (served_root / name).mkdir()
        run = (await start(client, 'rl-one', folder=str(served_root / name))).json()
        await client.post(url('rl-one', f'bulk/{run["run_id"]}/cancel'))
        ids.append(run['run_id'])
    listed = (await client.get(url('rl-one', 'bulk'))).json()
    assert listed['total'] == 2
    assert [r['run_id'] for r in listed['items']] == ids[::-1]


@pytest.mark.asyncio
async def test_failures_are_the_queue_rows_the_run_left_failed(
    app, pg, client, served_root
) -> None:
    pid = project(app, pg, 'rf-one')
    (served_root / 'a').mkdir()
    run = (await start(client, 'rf-one', folder=str(served_root / 'a'))).json()
    job = uuid.UUID(run['run_id'])
    (source_id,) = sql(pg, 'SELECT id FROM ingest_sources WHERE project_id = %s', pid)[0]
    with project_session(pg.engine('app'), pid) as session:
        repo.enqueue(session, source_id, ['/x/ok.jpg', '/x/gone.jpg', '/x/bad.jpg'])
        repo.claim(session, source_id, job, 10)
        repo.mark_done(session, source_id, job, [('/x/ok.jpg', 'img-1')])
        repo.mark_failed(
            session,
            source_id,
            job,
            [
                ('/x/gone.jpg', repo.error_text('missing', 'decode_failed'), True),
                ('/x/bad.jpg', repo.error_text('503', 'batch_error'), False),
            ],
        )

    page = (await client.get(url('rf-one', f'bulk/{run["run_id"]}/failures'))).json()

    assert page['total'] == 2
    assert {r['path']: r['error_kind'] for r in page['items']} == {
        '/x/gone.jpg': 'decode_failed',
        '/x/bad.jpg': 'batch_error',
    }
    paged = (
        await client.get(
            url('rf-one', f'bulk/{run["run_id"]}/failures'), params={'offset': 1, 'limit': 1}
        )
    ).json()
    assert (paged['total'], len(paged['items'])) == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'body',
    [
        {},
        {'folder': '/x', 'manifest': '/y'},
        {'folder': '/x', 'batch_size': 0},
        {'folder': '/x', 'concurrency': 999},
        {'folder': '/x', 'surprise': 1},
    ],
)
async def test_malformed_requests_are_rejected(app, pg, client, body) -> None:
    slug = f'rm-{uuid.uuid4().hex[:6]}'
    project(app, pg, slug)
    assert (await client.post(url(slug, 'bulk'), json=body)).status_code == 422


@pytest.mark.asyncio
async def test_a_source_outside_the_roots_or_missing_is_refused_with_a_code(
    app, pg, client, served_root, tmp_path
) -> None:
    project(app, pg, 'rr-one')
    for body in (
        {'folder': str(tmp_path)},
        {'folder': str(served_root / 'nope')},
        {'manifest': str(served_root)},
    ):
        reply = await client.post(url('rr-one', 'bulk'), json=body)
        assert reply.status_code == 422, reply.text
        assert reply.json()['detail']['error'] == 'bulk_ingest_invalid_source'
    assert (await client.get(url('rr-one', 'bulk'))).json()['total'] == 0
