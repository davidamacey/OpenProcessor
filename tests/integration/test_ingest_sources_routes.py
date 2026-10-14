"""Registered ingest sources through the real app: definitions, the fail-closed S3
allow-list, run-now (scan, then the bulk runner draining the queue), per-object
failure bookkeeping and the one-shot bucket start of ``/ingest/bulk``. The OpenSearch,
Triton and object-store boundaries are faked; the routes, queue, runner and pipeline
are production code."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
from _fake_object_store import FakeObjectStore

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakePEEncoder,
    FakeTritonPool,
    curation_app,
    jpeg_bytes,
    route_bulk_ingest_through_the_app,
)
from src.routers.curation import ingest_objects
from src.services.curation.bulk_ingest import source_scan


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration
BASE = '/curation/projects/default/ingest'
S3_BODY = {'bucket': 'imgs', 'prefix': 'in/'}


@pytest.fixture
def store() -> FakeObjectStore:
    fake = FakeObjectStore()
    for i in range(3):
        fake.put('imgs', f'in/a{i}.jpg', jpeg_bytes(200 + i))
    return fake


@pytest.fixture
def opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def client(
    opensearch: FakeOpenSearch,
    store: FakeObjectStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    monkeypatch.setenv('OP_S3_ALLOWED', 'imgs/in/')
    monkeypatch.setattr(ingest_objects, 'get_object_store', lambda: store)
    monkeypatch.setattr(source_scan, 'get_object_store', lambda: store)
    with curation_app(opensearch, FakeTritonPool(), monkeypatch, pe_encoder=FakePEEncoder()) as c:
        route_bulk_ingest_through_the_app(monkeypatch)
        yield c


def _wait(client: TestClient, run_id: str, *, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    run: dict[str, Any] = {}
    while time.time() < deadline:
        run = client.get(f'{BASE}/bulk/{run_id}').json()
        if run['status'] not in {'queued', 'running'}:
            return run
        time.sleep(0.05)
    msg = f'run stuck: {run}'
    raise AssertionError(msg)


def _create(client: TestClient, **over: Any) -> dict[str, Any]:
    body = {'name': 'inbox', 'backend': 's3', 's3': S3_BODY, **over}
    resp = client.post(f'{BASE}/sources', json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_definitions_are_validated_against_the_deployment_not_trusted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = client.post(
        f'{BASE}/sources',
        json={'name': 'x', 'backend': 's3', 's3': {'bucket': 'imgs', 'prefix': 'private/'}},
    )
    assert (outside.status_code, outside.json()['detail']['error']) == (422, 's3_not_allowed')

    bad_cron = client.post(
        f'{BASE}/sources',
        json={
            'name': 'x',
            'backend': 's3',
            's3': S3_BODY,
            'triggers': {'schedule': {'enabled': True, 'cron': '61 * * * *'}},
        },
    )
    assert (bad_cron.status_code, bad_cron.json()['detail']['error']) == (
        422,
        'ingest_source_invalid',
    )
    fast_poll = client.post(
        f'{BASE}/sources',
        json={
            'name': 'x',
            'backend': 's3',
            's3': S3_BODY,
            'triggers': {'watch': {'enabled': True, 'interval_s': 2}},
        },
    )
    assert fast_poll.status_code == 422

    created = _create(client, triggers={'webhook_enabled': True})
    assert created['webhook']['path'].endswith(f'/ingest/sources/{created["source_id"]}/events')
    assert created['can_run'] is True

    monkeypatch.delenv('OP_S3_ALLOWED')
    listing = client.get(f'{BASE}/sources').json()
    assert listing['s3_configured'] is False
    assert listing['items'][0]['blocked_reason'] == 's3_not_configured'
    assert listing['items'][0]['can_run'] is False
    refused = client.post(f'{BASE}/sources/{created["source_id"]}/run')
    assert (refused.status_code, refused.json()['detail']['error']) == (503, 's3_not_configured')


def test_run_now_ingests_the_bucket_and_a_second_run_fetches_only_what_is_new(
    client: TestClient, store: FakeObjectStore, opensearch: FakeOpenSearch
) -> None:
    source = _create(client, source='bucket_test')
    started = client.post(f'{BASE}/sources/{source["source_id"]}/run')
    assert started.status_code == 202, started.text
    run = _wait(client, started.json()['run_id'])
    assert run['status'] == 'completed', run
    assert run['progress']['ingested'] == 3
    assert run['source']['kind'] == 'queue'
    assert len(opensearch.images) == 3
    assert {d['source'] for d in opensearch.images.values()} == {'bucket_test'}

    status = client.get(f'{BASE}/sources/{source["source_id"]}').json()
    assert status['counts'] == {'pending': 0, 'done': 3, 'failed': 0, 'dead': 0}
    assert status['last_run']['status'] == 'completed'
    assert status['scan']['last']['queued'] == 3

    fetched_before = len(store.get_calls)
    store.put('imgs', 'in/new.jpg', jpeg_bytes(299), modified=time.time())
    second = _wait(
        client, client.post(f'{BASE}/sources/{source["source_id"]}/run').json()['run_id']
    )
    assert second['progress']['ingested'] == 1
    assert store.get_calls[fetched_before:] == ['in/new.jpg']
    assert len(opensearch.images) == 4


def test_failed_objects_are_kept_dead_or_retryable_and_retry_dead_requeues_them(
    client: TestClient, store: FakeObjectStore
) -> None:
    store.put('imgs', 'in/gone.jpg', jpeg_bytes(7))
    store.put('imgs', 'in/flaky.jpg', jpeg_bytes(8))
    store.fail('imgs', 'in/gone.jpg', 'gone')
    store.fail('imgs', 'in/flaky.jpg', 'transient')
    source = _create(client)
    run = _wait(client, client.post(f'{BASE}/sources/{source["source_id"]}/run').json()['run_id'])
    assert run['status'] == 'completed_with_errors'
    assert run['progress']['ingested'] == 3
    assert run['progress']['failed'] == 2

    status = client.get(f'{BASE}/sources/{source["source_id"]}').json()
    assert status['counts'] == {'pending': 0, 'done': 3, 'failed': 1, 'dead': 1}
    by_key = {e['key']: e for e in status['errors']}
    assert by_key['in/gone.jpg']['state'] == 'dead'
    assert by_key['in/flaky.jpg']['state'] == 'failed'
    assert status['can_retry_dead'] is True

    store.errors.clear()
    revived = client.post(f'{BASE}/sources/{source["source_id"]}/retry-dead').json()
    assert revived['retried'] == 1
    again = _wait(client, client.post(f'{BASE}/sources/{source["source_id"]}/run').json()['run_id'])
    assert again['progress']['failed'] == 0
    final = client.get(f'{BASE}/sources/{source["source_id"]}').json()
    assert final['counts'] == {'pending': 0, 'done': 5, 'failed': 0, 'dead': 0}


def test_the_one_shot_bucket_start_uses_the_same_source_and_stays_inside_the_allow_list(
    client: TestClient, store: FakeObjectStore, opensearch: FakeOpenSearch
) -> None:
    started = client.post(f'{BASE}/bulk', json={'s3': S3_BODY, 'source': 'oneshot'})
    assert started.status_code == 202, started.text
    run = _wait(client, started.json()['run_id'])
    assert run['progress']['ingested'] == 3
    assert run['source']['kind'] == 'queue'
    assert len(client.get(f'{BASE}/sources').json()['items']) == 1

    again = _wait(client, client.post(f'{BASE}/bulk', json={'s3': S3_BODY}).json()['run_id'])
    assert again['progress']['ingested'] == 0
    assert len(client.get(f'{BASE}/sources').json()['items']) == 1  # the same implicit source
    assert len(opensearch.images) == 3

    denied = client.post(f'{BASE}/bulk', json={'s3': {'bucket': 'imgs', 'prefix': ''}})
    assert (denied.status_code, denied.json()['detail']['error']) == (422, 's3_not_allowed')
    both = client.post(f'{BASE}/bulk', json={'s3': S3_BODY, 'folder': '/x'})
    assert both.status_code == 422


def _counts(resp: Any) -> dict[str, int]:
    return {k: resp.json()[k] for k in ('received', 'queued', 'duplicates', 'ignored')}


def _event(key: str, etag: str, *, bucket: str = 'imgs', name: str = 'ObjectCreated:Put') -> dict:
    return {
        'eventName': name,
        's3': {'bucket': {'name': bucket}, 'object': {'key': key, 'eTag': etag, 'size': 10}},
    }


def test_the_webhook_queues_matching_records_once(
    client: TestClient, store: FakeObjectStore
) -> None:
    source = _create(client, triggers={'webhook_enabled': True})
    events_url = f'{BASE}/sources/{source["source_id"]}/events'
    ref = store.ref('imgs', 'in/a0.jpg')
    body = {
        'Records': [
            _event('in/a0.jpg', ref.etag),
            _event('in/a0.jpg', ref.etag),  # a redelivery inside the same notification
            _event('in/notes.txt', 'e1'),  # not an image
            _event('elsewhere/b.jpg', 'e2'),  # outside the source prefix
            _event('in/a1.jpg', 'e3', bucket='other'),  # another bucket
            _event('in/a2.jpg', 'e4', name='ObjectRemoved:Delete'),
        ]
    }
    first = client.post(events_url, json=body)
    assert first.status_code == 202, first.text
    assert _counts(first) == {'received': 6, 'queued': 1, 'duplicates': 1, 'ignored': 4}
    replay = client.post(events_url, json=body)
    assert _counts(replay) == {'received': 6, 'queued': 0, 'duplicates': 2, 'ignored': 4}

    status = client.get(f'{BASE}/sources/{source["source_id"]}').json()
    assert status['counts']['pending'] == 1
    assert (status['webhook']['received'], status['webhook']['queued']) == (12, 1)
    assert status['webhook']['last_event_at'] is not None


def test_the_webhook_refuses_bad_bodies_and_sources_that_did_not_opt_in(
    client: TestClient,
) -> None:
    closed = _create(client)
    refused = client.post(f'{BASE}/sources/{closed["source_id"]}/events', json={'Records': []})
    assert (refused.status_code, refused.json()['detail']['error']) == (409, 'webhook_disabled')

    open_source = _create(client, triggers={'webhook_enabled': True})
    url = f'{BASE}/sources/{open_source["source_id"]}/events'
    bad = client.post(url, json={'Records': [{'eventName': 'ObjectCreated:Put', 's3': {}}]})
    assert bad.status_code == 422
    assert bad.json()['detail']['error'] == 'invalid_s3_event'
    assert 'Records[0].s3.bucket' in bad.json()['detail']['message']
    not_json = client.post(url, content=b'<xml/>', headers={'content-type': 'application/json'})
    assert not_json.status_code == 422
    missing = client.post(f'{BASE}/sources/src_nosuchsource/events', json={'Records': []})
    assert missing.status_code == 404


def test_a_dispatcher_tick_drains_what_a_webhook_queued_and_a_quiet_watcher_makes_no_run(
    client: TestClient, store: FakeObjectStore, opensearch: FakeOpenSearch
) -> None:
    from src.services.curation.bulk_ingest import dispatcher

    hook = _create(client, name='hook', triggers={'webhook_enabled': True})
    watcher = _create(
        client,
        name='watch',
        s3={'bucket': 'imgs', 'prefix': 'in/empty/'},
        triggers={'watch': {'enabled': True, 'interval_s': 10}},
    )
    refs = [store.ref('imgs', f'in/a{i}.jpg') for i in range(2)]
    events = {'Records': [_event(r.key, r.etag) for r in refs]}
    assert client.post(f'{BASE}/sources/{hook["source_id"]}/events', json=events).status_code == 202

    async def tick() -> list[Any]:
        return dispatcher.run_project_tick()

    taken = client.portal.call(tick)
    kinds = {(a.source_id, a.kind) for a in taken}
    assert (hook['source_id'], 'drain') in kinds
    assert (watcher['source_id'], 'poll') in kinds

    runs = client.get(f'{BASE}/bulk').json()['items']
    assert len(runs) == 1, 'the drain started a run; the quiet watcher started none'
    run = _wait(client, runs[0]['run_id'])
    assert run['status'] == 'completed', run
    assert run['progress']['ingested'] == 2
    assert len(opensearch.images) == 2
    # The webhook source listed nothing: its work came from the events alone.
    time.sleep(0.3)  # let the watcher's scan finish: it found nothing, so it makes no run
    assert len(client.get(f'{BASE}/bulk').json()['items']) == 1
    assert client.get(f'{BASE}/sources/{hook["source_id"]}').json()['scan']['last'] is None
    assert client.get(f'{BASE}/sources/{hook["source_id"]}').json()['counts']['done'] == 2
