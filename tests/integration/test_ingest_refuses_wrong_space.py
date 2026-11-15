"""A project whose item space is of another encoder refuses every ingest route over HTTP.

The encoder guard raises ``EmbeddingSpaceError`` before anything is encoded or written; the app
must answer its own status (409) with the stable code and the remedy text, never a 500, and store
nothing. The writers' services are covered one by one in
``tests/curation/test_vector_writers_refuse_wrong_space.py``; this is the HTTP surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _embedding_registry_fake import new_space
from _fake_object_store import FakeObjectStore

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakePEEncoder,
    FakeTritonPool,
    curation_app,
    file_engine,
    jpeg_bytes,
)
from src.domain.embedding_spaces import Target
from src.routers.curation import ingest_objects


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

PREFIX = '/curation/projects/default'
PHOTO = Path('/tmp/refuse_wrong_space.jpg')


@pytest.fixture
def opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def store() -> FakeObjectStore:
    fake = FakeObjectStore()
    fake.put('imgs', 'in/a0.jpg', jpeg_bytes(100))
    return fake


@pytest.fixture
def client(
    opensearch: FakeOpenSearch,
    store: FakeObjectStore,
    embedding_registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    embedding_registry.add(new_space(Target.ITEM, 'older_encoder'), primary=True)
    PHOTO.write_bytes(jpeg_bytes(1))
    monkeypatch.setenv('OP_S3_ALLOWED', 'imgs/in/')
    monkeypatch.setattr(ingest_objects, 'get_object_store', lambda: store)
    with curation_app(
        opensearch, FakeTritonPool(), monkeypatch, pe_encoder=FakePEEncoder(), engine=file_engine()
    ) as c:
        yield c
    PHOTO.unlink(missing_ok=True)


def _batch(c: TestClient) -> Any:
    return c.post(f'{PREFIX}/ingest/batch', json={'items': [{'path': str(PHOTO), 'source': 's'}]})


def _image(c: TestClient) -> Any:
    return c.post(f'{PREFIX}/ingest/image', json={'path': str(PHOTO), 'source': 's'})


def _upload(c: TestClient) -> Any:
    return c.post(
        f'{PREFIX}/ingest/upload',
        files=[('images', ('a.jpg', jpeg_bytes(2), 'image/jpeg'))],
        data={'image_paths': json.dumps(['remote://a.jpg'])},
    )


def _objects(c: TestClient) -> Any:
    ref = {'bucket': 'imgs', 'key': 'in/a0.jpg'}
    return c.post(f'{PREFIX}/ingest/objects', json={'items': [ref]})


ROUTES: list[tuple[str, Callable[[TestClient], Any]]] = [
    ('ingest/batch', _batch),
    ('ingest/image', _image),
    ('ingest/upload', _upload),
    ('ingest/objects', _objects),
]


@pytest.mark.parametrize(('name', 'call'), ROUTES, ids=[n for n, _ in ROUTES])
def test_an_ingest_route_answers_409_with_the_remedy_and_stores_nothing(
    client: TestClient, opensearch: FakeOpenSearch, name: str, call: Callable[[Any], Any]
) -> None:
    resp = call(client)
    assert resp.status_code == 409, f'{name}: {resp.status_code} {resp.text}'
    detail = resp.json()['detail']
    assert detail['error'] == 'embedding_model_unavailable', resp.text
    for part in ('default', 'older_encoder_image', 'pe_crop_encoder', 'new project'):
        assert part in detail['message'], (part, resp.text)
    assert opensearch.images == {}, f'{name} stored an image document'
    assert opensearch.items == {}, f'{name} stored an item document'
