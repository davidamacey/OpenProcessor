"""``GET /serving_profiles/active``: nothing active is an answer (200 ``{active: null}``), not a 404.

Cropwright asks on every open of the Serving screen; a deployment that never activated a profile
is a normal state, and a 404 there is a failed request in every log and monitor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.curation import research_configs
from src.routers.curation.projects import global_router


if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

URL = '/curation/serving_profiles/active'


def _client(monkeypatch: pytest.MonkeyPatch, found: Callable[[], Any]) -> TestClient:
    async def active() -> Any:
        return found()

    monkeypatch.setattr(research_configs, 'active_serving_profile', active)
    app = FastAPI()
    app.include_router(global_router, prefix='/curation')
    return TestClient(app)


def test_nothing_active_is_200_with_a_null(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _client(monkeypatch, lambda: None).get(URL)
    assert (response.status_code, response.json()) == (200, {'active': None})


def test_an_active_profile_is_served_under_active(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = SimpleNamespace(name='standard', revision=2, body={'load_list': {}})
    response = _client(monkeypatch, lambda: (doc, 5)).get(URL)
    assert response.status_code == 200, response.text
    assert response.json() == {
        'active': {
            'name': 'standard',
            'revision': 2,
            'setting_revision': 5,
            'body': {'load_list': {}},
        }
    }
