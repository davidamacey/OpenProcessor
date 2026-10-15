"""MAX_FILE_SIZE_MB: 413 with the served cap, counted per file (#239)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.core.upload_limits as limits


MIB = 1024 * 1024


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.app.factory import create_app

    monkeypatch.setattr(
        limits,
        'get_settings',
        lambda: SimpleNamespace(max_file_size_mb=1, max_file_size_bytes=MIB),
    )
    monkeypatch.setattr(
        limits,
        'get_curation_config',
        lambda: SimpleNamespace(upload_max_bytes_per_request=4 * MIB),
    )
    return TestClient(create_app(), raise_server_exceptions=False)


def _jpeg(size: int) -> tuple[str, bytes, str]:
    return ('a.jpg', b'\xff' * size, 'image/jpeg')


def test_one_oversize_file_answers_413_with_the_cap(client: TestClient) -> None:
    resp = client.post('/detect', files={'image': _jpeg(MIB + 1)})
    assert resp.status_code == 413
    body = resp.json()
    assert body['detail'] == 'File too large. Maximum: 1MB'
    assert body['max_file_size_mb'] == 1
    assert body['request_id'] == resp.headers['X-Request-ID']


def test_an_oversize_body_that_is_not_a_form_answers_413_not_500(client: TestClient) -> None:
    resp = client.post('/detect', content=b'x' * (MIB + 1), headers={'content-type': 'text/plain'})
    assert resp.status_code == 413
    assert resp.json()['max_file_size_mb'] == 1


def test_the_cap_counts_each_file_not_the_whole_request(client: TestClient) -> None:
    files = [('images', _jpeg(MIB // 2 + 100)), ('images', _jpeg(MIB // 2 + 100))]
    resp = client.post('/detect/batch', files=files)  # 1.00 MiB + in total, 0.5 MiB each
    assert resp.status_code != 413


def test_a_later_file_in_a_multi_file_request_is_checked_too(client: TestClient) -> None:
    files = [('images', _jpeg(100)), ('images', _jpeg(MIB + 1))]
    resp = client.post('/detect/batch', files=files)
    assert resp.status_code == 413
    assert resp.json()['detail'] == 'File too large. Maximum: 1MB'


def test_a_multipart_body_past_the_largest_accepted_request_is_refused_early(
    client: TestClient,
) -> None:
    resp = client.post('/detect', files={'image': _jpeg(4 * MIB + 2 * MIB)})
    assert resp.status_code == 413
