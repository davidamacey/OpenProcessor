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
    assert run['progress']['ingested'] == 3 and run['progress']['failed'] == 2

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
    assert run['progress']['ingested'] == 3 and run['source']['kind'] == 'queue'
    assert len(client.get(f'{BASE}/sources').json()['items']) == 1

    again = _wait(client, client.post(f'{BASE}/bulk', json={'s3': S3_BODY}).json()['run_id'])
    assert again['progress']['ingested'] == 0
    assert len(client.get(f'{BASE}/sources').json()['items']) == 1  # the same implicit source
    assert len(opensearch.images) == 3

    denied = client.post(f'{BASE}/bulk', json={'s3': {'bucket': 'imgs', 'prefix': ''}})
    assert (denied.status_code, denied.json()['detail']['error']) == (422, 's3_not_allowed')
    both = client.post(f'{BASE}/bulk', json={'s3': S3_BODY, 'folder': '/x'})
    assert both.status_code == 422
