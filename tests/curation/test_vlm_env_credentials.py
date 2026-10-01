"""A credential in ``OP_VLM_URL`` (W9 review M5): the env built-in drops it at
the source, so no route, log line or labeler carries it."""

from __future__ import annotations

import pytest

from curation.conftest import GLOBAL_VLM, SCOPED
from src.services.labeling.vlm_endpoints import env_endpoint
from src.services.labeling.vlm_url_policy import strip_userinfo


_SECRET_URL = 'http://user:SECRETPW@vlm:8000/v1'


def test_the_env_endpoint_never_holds_the_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', _SECRET_URL)
    monkeypatch.setenv('OP_VLM_MODEL', 'served')
    endpoint = env_endpoint()
    assert endpoint is not None
    assert endpoint.body.base_url == 'http://vlm:8000/v1'
    assert 'SECRETPW' not in repr(endpoint)
    assert 'SECRETPW' not in endpoint.ref + endpoint.etag


@pytest.mark.parametrize(
    ('raw', 'clean'),
    [
        ('http://u:p@host:8000/v1', 'http://host:8000/v1'),
        ('http://u@[::1]:8000/v1?key=1#x', 'http://[::1]:8000/v1'),
        ('https://host/v1', 'https://host/v1'),
    ],
)
def test_strip_userinfo(raw: str, clean: str) -> None:
    assert strip_userinfo(raw) == clean


def test_no_route_that_serves_a_base_url_leaks_it(vlm_api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_VLM_URL', _SECRET_URL)
    monkeypatch.setenv('OP_VLM_MODEL', 'served')
    leaks = []
    for url in (
        GLOBAL_VLM,
        f'{GLOBAL_VLM}/env',
        f'{SCOPED}/config/vocabulary',
        f'{SCOPED}/methods',
        f'{SCOPED}/vlm/endpoints/active',
        f'{SCOPED}/models/status',
        '/curation/vlm/catalog',
    ):
        response = vlm_api.lenient.get(url)
        if 'SECRETPW' in response.text:
            leaks.append(url)
    assert not leaks


def test_a_labeler_built_from_the_env_endpoint_logs_no_credential(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from src.services.labeling.vlm_factory import build_uncached_labeler
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    monkeypatch.setenv('OP_VLM_URL', 'http://user:SECRETPW@127.0.0.1:1/v1')
    monkeypatch.setenv('OP_VLM_MODEL', 'served')
    endpoint = env_endpoint()
    assert endpoint is not None
    labeler = build_uncached_labeler(endpoint, GENERIC_ITEM_PACK)
    health = asyncio.run(labeler.health())
    assert health.reachable is False
    captured = capsys.readouterr()
    assert 'SECRETPW' not in captured.out + captured.err
    assert 'SECRETPW' not in (health.last_error or '')
