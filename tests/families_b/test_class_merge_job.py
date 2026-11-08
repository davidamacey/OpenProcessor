"""``POST /classes/merge`` queues a ``class_merge`` job (4B-5): one row, one message, 409 while busy."""

from __future__ import annotations

import uuid

import pytest
from families_b.fake_items import FakeItems
from families_b.support import project
from fixtures.job_rows import sql

from src.services.classes.registry import ClassRegistry


pytestmark = pytest.mark.db


async def _two_classes(slug: str) -> tuple[int, int]:
    reg = ClassRegistry(slug)
    return await reg.add_class('sedan'), await reg.add_class('coupe')


@pytest.mark.asyncio
async def test_the_merge_route_queues_one_job_and_sends_one_message(
    app, client, pg, producer
) -> None:
    slug, pid = project(app, pg)
    app.state.opensearch = FakeItems([{'class_id': 0, 'class_name': 'sedan'}])
    a, b = await _two_classes(slug)

    reply = await client.post(
        f'/curation/projects/{slug}/classes/merge', json={'source_id': a, 'target_id': b}
    )
    assert reply.status_code == 202, reply.text
    job_id = reply.json()['id']
    assert reply.json()['status'] == 'queued'
    assert reply.json()['kind'] == 'class_merge'
    assert producer.sent() == [job_id]
    rows = sql(pg, 'SELECT kind, status, params FROM jobs WHERE project_id = %s', pid)
    assert rows == [('class_merge', 'queued', {'source_id': a, 'target_id': b})]
    merged = (await ClassRegistry(slug).snapshot()).get(a)
    assert merged.merged_into is None  # the registry changes in the job, not the request


@pytest.mark.asyncio
async def test_a_second_merge_while_one_is_active_is_409_job_busy(
    app, client, pg, producer
) -> None:
    slug, _pid = project(app, pg)
    app.state.opensearch = FakeItems([])
    a, b = await _two_classes(slug)
    url = f'/curation/projects/{slug}/classes/merge'
    first = await client.post(url, json={'source_id': a, 'target_id': b})
    again = await client.post(url, json={'source_id': b, 'target_id': a})
    assert first.status_code == 202
    assert again.status_code == 409
    assert again.json()['detail']['error'] == 'job_busy'
    assert producer.sent() == [first.json()['id']]


@pytest.mark.asyncio
async def test_broker_down_is_503_queue_unavailable_and_the_row_fails(
    app, client, pg, producer
) -> None:
    slug, pid = project(app, pg)
    app.state.opensearch = FakeItems([])
    a, b = await _two_classes(slug)
    producer.fail_send = True
    reply = await client.post(
        f'/curation/projects/{slug}/classes/merge', json={'source_id': a, 'target_id': b}
    )
    assert reply.status_code == 503
    assert reply.json()['detail']['error'] == 'queue_unavailable'
    assert sql(pg, 'SELECT status FROM jobs WHERE project_id = %s', pid) == [('failed',)]


@pytest.mark.asyncio
async def test_a_frozen_holdout_crop_refuses_the_merge_before_any_job(
    app, client, pg, producer
) -> None:
    slug, pid = project(app, pg)
    a, b = await _two_classes(slug)
    app.state.opensearch = FakeItems([{'class_id': a, 'class_name': 'sedan', 'test_holdout': True}])
    reply = await client.post(
        f'/curation/projects/{slug}/classes/merge', json={'source_id': a, 'target_id': b}
    )
    assert reply.status_code == 409
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', pid) == [(0,)]


@pytest.mark.asyncio
async def test_cancel_while_queued_cancels_the_merge_job(app, client, pg, producer) -> None:
    from src.db.session import async_session_factory
    from src.services.jobs.service import cancel_job

    slug, pid = project(app, pg)
    app.state.opensearch = FakeItems([])
    a, b = await _two_classes(slug)
    reply = await client.post(
        f'/curation/projects/{slug}/classes/merge', json={'source_id': a, 'target_id': b}
    )
    import uuid

    async with async_session_factory()() as session:
        await cancel_job(session, uuid.UUID(reply.json()['id']), pid, requested_by='test')
    assert sql(pg, 'SELECT status FROM jobs WHERE project_id = %s', pid) == [('cancelled',)]


@pytest.mark.asyncio
async def test_the_registered_task_merges_and_relabels_once_through_the_wrapper(
    app, client, pg, producer, worker_app, monkeypatch
) -> None:
    import asyncio

    from families_b.support import bind_worker_to

    from src.services.classes import projection_sync

    slug, _pid = project(app, pg)
    items = FakeItems([])
    app.state.opensearch = items
    a, b = await _two_classes(slug)
    reply = await client.post(
        f'/curation/projects/{slug}/classes/merge', json={'source_id': a, 'target_id': b}
    )
    job_id = reply.json()['id']
    calls = []

    async def relabel(_os, _index, source, target, name, *, cancelled):
        calls.append((source, target, name))
        return 2, 0

    monkeypatch.setattr(projection_sync, 'relabel_merged_items', relabel)
    bind_worker_to(monkeypatch, app, items)

    outcome = await asyncio.to_thread(
        worker_app.tasks['op.families_b.class_merge'].apply, args=[job_id], task_id=job_id
    )

    assert outcome.successful(), outcome.traceback
    assert calls == [(a, b, 'coupe')]
    [(status, result)] = sql(pg, 'SELECT status, result FROM jobs WHERE id = %s', uuid.UUID(job_id))
    assert status == 'completed'
    assert result['relabelled'] == 2
    assert (await ClassRegistry(slug).snapshot()).get(a).merged_into == b
