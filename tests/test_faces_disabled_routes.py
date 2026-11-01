"""With OP_FACES_ENABLED=false every face and person route refuses before touching a backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config.env import EnvError


REFUSED = [
    ('POST', '/faces/detect'),
    ('POST', '/faces/recognize'),
    ('POST', '/faces/verify'),
    ('POST', '/faces/search'),
    ('POST', '/faces/identify'),
    ('POST', '/v1/faces/detect'),
    ('POST', '/persons/cluster'),
    ('GET', '/persons'),
    ('GET', '/persons/p1'),
    ('PUT', '/persons/p1/name'),
    ('POST', '/persons/merge'),
    ('DELETE', '/persons/p1'),
    ('POST', '/search/face'),
    ('POST', '/clusters/train/faces'),
    ('GET', '/clusters/stats/faces'),
    ('GET', '/clusters/faces/1'),
]


def _boom(*_a: object, **_k: object) -> None:
    raise AssertionError('a face backend was reached while faces are off')


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.main import app

    monkeypatch.setattr('src.clients.fast_face_client._cached_face_client', _boom, raising=False)
    monkeypatch.setattr('src.services.face_identity.get_face_identity_service', _boom)
    monkeypatch.setattr('src.routers.persons.get_face_identity_service', _boom)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(('method', 'path'), REFUSED)
def test_face_routes_refuse_when_off(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    resp = client.request(method, path)
    assert resp.status_code == 400
    body = resp.json()
    assert body['error'] == 'faces_disabled'
    assert 'OP_FACES_ENABLED' in body['detail']
    assert resp.headers['X-Request-ID']


def test_faces_on_still_validates_the_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_FACES_ENABLED', raising=False)
    assert client.post('/faces/detect').status_code == 422


def test_non_face_cluster_index_is_not_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    resp = client.get('/clusters/stats/global')
    assert (resp.json() if resp.headers['content-type'].startswith('application/json') else {}).get(
        'error'
    ) != 'faces_disabled'


def test_face_client_factory_refuses_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.clients.fast_face_client import get_fast_face_client
    from src.config.faces import FacesDisabledError

    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    with pytest.raises(FacesDisabledError):
        get_fast_face_client()


def test_malformed_switch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config.faces import faces_enabled

    monkeypatch.setenv('OP_FACES_ENABLED', 'maybe')
    with pytest.raises(EnvError):
        faces_enabled()
