"""``POST /ingest/objects`` against the real app with the OpenSearch, Triton and
object-store boundaries faked: the fail-closed allow-list, the stored object
reference, the no-download skip of an already stored version, and per-object failure
classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _fake_object_store import FakeObjectStore

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakePEEncoder,
    FakeTritonPool,
    curation_app,
    jpeg_bytes,
)
from src.routers.curation import ingest_objects


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration
URL = '/curation/projects/default/ingest/objects'


@pytest.fixture
def store() -> FakeObjectStore:
    fake = FakeObjectStore()
    for i in range(3):
        fake.put('imgs', f'in/a{i}.jpg', jpeg_bytes(100 + i))
    return fake


@pytest.fixture
def opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def client(
    opensearch: FakeOpenSearch, store: FakeObjectStore, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    monkeypatch.setenv('OP_S3_ALLOWED', 'imgs/in/')
    monkeypatch.setattr(ingest_objects, 'get_object_store', lambda: store)
    with curation_app(opensearch, FakeTritonPool(), monkeypatch, pe_encoder=FakePEEncoder()) as c:
        yield c


def _items(store: FakeObjectStore, *keys: str, bucket: str = 'imgs') -> list[dict[str, object]]:
    return [{'bucket': bucket, 'key': k, 'etag': store.ref(bucket, k).etag} for k in keys]


def test_s3_is_off_until_configured(
    opensearch: FakeOpenSearch, store: FakeObjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_S3_ALLOWED', raising=False)
    with curation_app(opensearch, FakeTritonPool(), monkeypatch, pe_encoder=FakePEEncoder()) as c:
        resp = c.post(URL, json={'items': _items(store, 'in/a0.jpg')})
    assert resp.status_code == 503
    assert resp.json()['detail']['error'] == 's3_not_configured'
    assert store.get_calls == []


def test_outside_the_allow_list_is_refused_without_a_fetch(
    client: TestClient, store: FakeObjectStore
) -> None:
    store.put('imgs', 'private/x.jpg', jpeg_bytes(1))
    store.put('other', 'in/y.jpg', jpeg_bytes(2))
    for bucket, key in (('imgs', 'private/x.jpg'), ('other', 'in/y.jpg')):
        resp = client.post(URL, json={'items': [{'bucket': bucket, 'key': key}]})
        assert resp.status_code == 422, resp.text
        assert resp.json()['detail']['error'] == 's3_not_allowed'
    assert store.get_calls == []


def test_ingest_stores_the_reference_and_a_restored_version_is_not_downloaded_again(
    client: TestClient, store: FakeObjectStore, opensearch: FakeOpenSearch
) -> None:
    items = _items(store, 'in/a0.jpg', 'in/a1.jpg')
    first = client.post(URL, json={'items': items, 'source': 's3_test'})
    assert first.status_code == 200, first.text
    assert first.json()['summary']['successful'] == 2
    expected = {store.ref('imgs', k).to_uri() for k in ('in/a0.jpg', 'in/a1.jpg')}
    assert {d['source_identifier'] for d in opensearch.images.values()} == expected
    assert {r['source_identifier'] for r in first.json()['results']} == expected
    assert {d['source'] for d in opensearch.images.values()} == {'s3_test'}

    gets = len(store.get_calls)
    again = client.post(URL, json={'items': items})
    assert again.json()['summary']['duplicates'] == 2
    assert [r['status'] for r in again.json()['results']] == ['duplicate', 'duplicate']
    assert len(store.get_calls) == gets  # the exact versions are stored: no GET


def test_each_failure_is_classified_and_the_rest_of_the_batch_succeeds(
    client: TestClient, store: FakeObjectStore
) -> None:
    store.put('imgs', 'in/gone.jpg', jpeg_bytes(7))
    store.put('imgs', 'in/denied.jpg', jpeg_bytes(8))
    store.put('imgs', 'in/flaky.jpg', jpeg_bytes(9))
    store.put('imgs', 'in/notes.txt', b'hello')
    store.fail('imgs', 'in/gone.jpg', 'gone')
    store.fail('imgs', 'in/denied.jpg', 'forbidden')
    store.fail('imgs', 'in/flaky.jpg', 'transient')
    keys = ['in/a2.jpg', 'in/gone.jpg', 'in/denied.jpg', 'in/flaky.jpg', 'in/notes.txt']
    resp = client.post(URL, json={'items': _items(store, *keys)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['status'] == 'partial'
    kinds = {
        r['source_identifier'].split('/', 3)[3].split('#')[0]: r['error_kind']
        for r in body['results']
    }
    assert kinds == {
        'in/a2.jpg': None,
        'in/gone.jpg': 'gone',
        'in/denied.jpg': 'forbidden',
        'in/flaky.jpg': 'fetch_error',
        'in/notes.txt': 'unsupported_type',
    }


def test_an_object_over_the_size_cap_fails_alone(
    client: TestClient, store: FakeObjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_S3_MAX_OBJECT_MB', '1')
    store.put('imgs', 'in/huge.jpg', b'\xff\xd8' + b'0' * (2 * 1024 * 1024))
    resp = client.post(URL, json={'items': _items(store, 'in/a0.jpg', 'in/huge.jpg')})
    by_status = {r['status']: r for r in resp.json()['results']}
    assert by_status['failed']['error_kind'] == 'too_large'
    assert by_status['success']
    assert 'in/huge.jpg' not in store.get_calls  # rejected from its size, never downloaded


def test_presign_gives_a_url_only_inside_the_allow_list(
    client: TestClient, store: FakeObjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_objects, 'get_object_store', lambda: store)
    ok = client.post(
        '/curation/projects/default/ingest/objects/presign',
        json={'source_identifier': store.ref('imgs', 'in/a0.jpg').to_uri(), 'expires_in': 120},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()['url'] == 'https://fake.example/imgs/in/a0.jpg?expires=120'
    assert ok.json()['expires_in'] == 120
    store.put('imgs', 'private/x.jpg', jpeg_bytes(3))
    denied = client.post(
        '/curation/projects/default/ingest/objects/presign',
        json={'source_identifier': store.ref('imgs', 'private/x.jpg').to_uri()},
    )
    assert denied.status_code == 422
