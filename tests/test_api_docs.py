"""The API docs are self-hosted: no external asset URL in the served HTML and
every referenced asset answers 200 from the app itself."""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.api_docs import install_api_docs


@pytest.fixture
def client() -> TestClient:
    app = FastAPI(title='T', docs_url=None, redoc_url=None)
    install_api_docs(app)
    return TestClient(app)


_ASSET_ATTR = re.compile(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']|url:\s*["\']([^"\']+)["\']')


@pytest.mark.parametrize('page', ['/docs', '/redoc'])
def test_page_has_no_external_asset_and_assets_load(client: TestClient, page: str) -> None:
    r = client.get(page)
    assert r.status_code == 200
    refs = [a or b for a, b in _ASSET_ATTR.findall(r.text)]
    assert refs
    assert not [u for u in refs if re.match(r'(?:https?:)?//', u)], refs
    assert 'googleapis' not in r.text
    assert 'jsdelivr' not in r.text
    for u in refs:
        if u.startswith('/docs-assets/'):
            assert client.get(u).status_code == 200, u


def test_openapi_json_and_oauth_redirect_at_same_paths(client: TestClient) -> None:
    assert client.get('/openapi.json').status_code == 200
    assert client.get('/docs/oauth2-redirect').status_code == 200
    assert 'openapi.json' in client.get('/redoc').text
