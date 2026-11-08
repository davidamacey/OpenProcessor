"""Registered ingest sources through the real routes over PostgreSQL: definitions, the
fail-closed S3 allow-list, run-now as a queued job, the one-shot bucket start, retry of dead
files and the webhook that queues files. Scans and drains are the worker's
(``tests/tasks/test_families_c_ingest.py``)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from fixtures.job_rows import insert_row, project_session, sql

from curation.ingest_db.app_fixtures import BUCKET_BODY, project, url
from src.db.repos import ingest_sources as repo


if TYPE_CHECKING:
    import httpx

pytestmark = [
    pytest.mark.db,
    pytest.mark.usefixtures(
        'bound_async_engine', 'api_async_engine', 'job_env', 'producer', 'wakes'
    ),
]


async def create(client: httpx.AsyncClient, slug: str, **over: Any) -> dict[str, Any]:
    body = {'name': 'inbox', 'backend': 's3', 's3': BUCKET_BODY, **over}
    reply = await client.post(url(slug, 'sources'), json=body)
    assert reply.status_code == 201, reply.text
    return reply.json()


@pytest.mark.asyncio
async def test_definitions_are_validated_against_the_deployment_not_trusted(
    app, pg, client, store, monkeypatch
) -> None:
    project(app, pg, 'sd-one')
    base = url('sd-one', 'sources')
    outside = await client.post(
        base, json={'name': 'x', 'backend': 's3', 's3': {'bucket': 'imgs', 'prefix': 'private/'}}
    )
    assert (outside.status_code, outside.json()['detail']['error']) == (422, 's3_not_allowed')

    bad_cron = await client.post(
        base,
        json={
            'name': 'x',
            'backend': 's3',
            's3': BUCKET_BODY,
            'triggers': {'schedule': {'enabled': True, 'cron': '61 * * * *'}},
        },
    )
    assert (bad_cron.status_code, bad_cron.json()['detail']['error']) == (
        422,
        'ingest_source_invalid',
    )
    fast_poll = await client.post(
        base,
        json={
            'name': 'x',
            'backend': 's3',
            's3': BUCKET_BODY,
            'triggers': {'watch': {'enabled': True, 'interval_s': 2}},
        },
    )
    assert fast_poll.status_code == 422

    created = await create(client, 'sd-one', triggers={'webhook_enabled': True})
    assert created['webhook']['path'].endswith(f'/ingest/sources/{created["source_id"]}/events')
    assert created['can_run'] is True
    duplicate = await client.post(base, json={'name': 'inbox', 'backend': 's3', 's3': BUCKET_BODY})
    assert duplicate.status_code == 409

    monkeypatch.delenv('OP_S3_ALLOWED')
    listing = (await client.get(base)).json()
    assert listing['s3_configured'] is False
    assert listing['items'][0]['blocked_reason'] == 's3_not_configured'
    assert listing['items'][0]['can_run'] is False
    refused = await client.post(url('sd-one', f'sources/{created["source_id"]}/run'))
    assert (refused.status_code, refused.json()['detail']['error']) == (503, 's3_not_configured')


@pytest.mark.asyncio
async def test_run_now_queues_a_run_that_scans_first_and_a_second_while_it_lives_is_409(
    app, pg, client, store, producer
) -> None:
    pid = project(app, pg, 'sr-one')
    source = await create(client, 'sr-one', source='bucket_test')

    started = await client.post(url('sr-one', f'sources/{source["source_id"]}/run'))

    assert started.status_code == 202, started.text
    run = started.json()
    assert run['source']['kind'] == 'queue'
    assert run['options']['source_tag'] == 'bucket_test'
    ((params,),) = sql(pg, 'SELECT params FROM jobs WHERE id = %s', uuid.UUID(run['run_id']))
    assert (params['source_id'], params['scan']) == (source['source_id'], True)
    assert producer.sent() == [run['run_id']]
    again = await client.post(url('sr-one', f'sources/{source["source_id"]}/run'))
    assert (again.status_code, again.json()['detail']['run_id']) == (409, run['run_id'])
    status = (await client.get(url('sr-one', f'sources/{source["source_id"]}'))).json()
    assert status['last_run']['run_id'] == run['run_id']
    assert status['can_run'] is False
    assert pid


@pytest.mark.asyncio
async def test_failed_files_are_dead_or_retryable_and_retry_dead_requeues_them(
    app, pg, client, store
) -> None:
    pid = project(app, pg, 'sf-one')
    source = await create(client, 'sf-one')
    source_id = uuid.UUID(source['source_id'])
    job = insert_row(pg, pid, 'running', kind='bulk_ingest_run', owner='w:1')
    with project_session(pg.engine('app'), pid) as session:
        repo.enqueue(session, source_id, ['ok', 'flaky', 'gone'])
        repo.claim(session, source_id, job, 10)
        repo.mark_done(session, source_id, job, [('ok', 'img-1')])
        repo.mark_failed(
            session,
            source_id,
            job,
            [('flaky', repo.error_text('503', 'batch_error'), False), ('gone', 'gone', True)],
        )

    status = (await client.get(url('sf-one', f'sources/{source["source_id"]}'))).json()
    assert status['counts'] == {'pending': 0, 'done': 1, 'failed': 1, 'dead': 1}
    by_key = {e['key']: e for e in status['errors']}
    assert by_key['gone']['state'] == 'dead'
    assert by_key['flaky']['state'] == 'failed'
    assert status['can_retry_dead'] is True

    revived = (await client.post(url('sf-one', f'sources/{source["source_id"]}/retry-dead'))).json()
    assert revived['retried'] == 1
    final = (await client.get(url('sf-one', f'sources/{source["source_id"]}'))).json()
    assert final['counts'] == {'pending': 1, 'done': 1, 'failed': 1, 'dead': 0}


@pytest.mark.asyncio
async def test_the_one_shot_bucket_start_is_one_source_and_stays_inside_the_allow_list(
    app, pg, client, store
) -> None:
    pid = project(app, pg, 'so-one')
    started = await client.post(
        url('so-one', 'bulk'), json={'s3': BUCKET_BODY, 'source': 'oneshot'}
    )
    assert started.status_code == 202, started.text
    run = started.json()
    assert run['source']['kind'] == 'queue'
    await client.post(url('so-one', f'bulk/{run["run_id"]}/cancel'))

    again = await client.post(url('so-one', 'bulk'), json={'s3': BUCKET_BODY})
    assert again.status_code == 202
    assert again.json()['resumed'] is True
    # a one-shot source is not listed and not addressable as a registered one
    assert (await client.get(url('so-one', 'sources'))).json()['items'] == []
    rows = sql(pg, 'SELECT count(*) FROM ingest_sources WHERE project_id = %s', pid)
    assert rows == [(1,)]

    denied = await client.post(
        url('so-one', 'bulk'), json={'s3': {'bucket': 'imgs', 'prefix': 'private/'}}
    )
    assert (denied.status_code, denied.json()['detail']['error']) == (422, 's3_not_allowed')
    both = await client.post(url('so-one', 'bulk'), json={'s3': BUCKET_BODY, 'folder': '/x'})
    assert both.status_code == 422


def event(key: str, etag: str, *, bucket: str = 'imgs', name: str = 'ObjectCreated:Put') -> dict:
    return {
        'eventName': name,
        's3': {'bucket': {'name': bucket}, 'object': {'key': key, 'eTag': etag, 'size': 10}},
    }


def counts(reply: Any) -> dict[str, int]:
    return {k: reply.json()[k] for k in ('received', 'queued', 'duplicates', 'ignored')}


@pytest.mark.asyncio
async def test_the_webhook_queues_matching_records_once(app, pg, client, store) -> None:
    project(app, pg, 'sw-one')
    source = await create(client, 'sw-one', triggers={'webhook_enabled': True})
    events_url = url('sw-one', f'sources/{source["source_id"]}/events')
    ref = store.ref('imgs', 'in/a0.jpg')
    body = {
        'Records': [
            event('in/a0.jpg', ref.etag),
            event('in/a0.jpg', ref.etag),  # a redelivery inside the same notification
            event('in/notes.txt', 'e1'),  # not an image
            event('elsewhere/b.jpg', 'e2'),  # outside the source prefix
            event('in/a1.jpg', 'e3', bucket='other'),  # another bucket
            event('in/a2.jpg', 'e4', name='ObjectRemoved:Delete'),
        ]
    }
    first = await client.post(events_url, json=body)
    assert first.status_code == 202, first.text
    assert counts(first) == {'received': 6, 'queued': 1, 'duplicates': 1, 'ignored': 4}
    replay = await client.post(events_url, json=body)
    assert counts(replay) == {'received': 6, 'queued': 0, 'duplicates': 2, 'ignored': 4}

    status = (await client.get(url('sw-one', f'sources/{source["source_id"]}'))).json()
    assert status['counts']['pending'] == 1
    assert (status['webhook']['received'], status['webhook']['queued']) == (12, 1)
    assert status['webhook']['last_event_at'] is not None


@pytest.mark.asyncio
async def test_the_webhook_refuses_bad_bodies_and_sources_that_did_not_opt_in(
    app, pg, client, store
) -> None:
    project(app, pg, 'sx-one')
    closed = await create(client, 'sx-one')
    refused = await client.post(
        url('sx-one', f'sources/{closed["source_id"]}/events'), json={'Records': []}
    )
    assert (refused.status_code, refused.json()['detail']['error']) == (409, 'webhook_disabled')

    opened = await create(client, 'sx-one', name='open', triggers={'webhook_enabled': True})
    events_url = url('sx-one', f'sources/{opened["source_id"]}/events')
    bad = await client.post(
        events_url, json={'Records': [{'eventName': 'ObjectCreated:Put', 's3': {}}]}
    )
    assert (bad.status_code, bad.json()['detail']['error']) == (422, 'invalid_s3_event')
    assert 'Records[0].s3.bucket' in bad.json()['detail']['message']
    not_json = await client.post(
        events_url, content=b'<xml/>', headers={'content-type': 'application/json'}
    )
    assert not_json.status_code == 422
    missing = await client.post(
        url('sx-one', f'sources/{uuid.uuid4()}/events'), json={'Records': []}
    )
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_a_source_of_another_project_is_not_found(app, pg, client, store) -> None:
    project(app, pg, 'sy-one')
    project(app, pg, 'sy-two')
    source = await create(client, 'sy-one')
    for method, tail in (('get', ''), ('delete', ''), ('post', '/run'), ('post', '/retry-dead')):
        reply = await getattr(client, method)(url('sy-two', f'sources/{source["source_id"]}{tail}'))
        assert reply.status_code == 404, (method, tail)
