"""A service's refusal reaches the caller exactly as the ``HTTPException`` it replaced.

Services no longer import FastAPI; they raise ``RequestRejectedError`` and the
curation app maps it to the same status and ``{"detail": ...}`` body.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.core.exceptions import RequestRejectedError
from src.routers.curation._api_error import api_error
from src.routers.curation._project_deps import install_project_exception_handlers
from src.services.rejections import rejection


_DETAILS: list[Any] = [
    'path traversal not allowed',
    {'error': 'slug_taken', 'message': 'already exists', 'project': 'alpha'},
]


def _app(raiser: Any) -> TestClient:
    router = APIRouter()

    @router.get('/boom')
    async def boom() -> None:
        raise raiser

    app = FastAPI()
    app.include_router(router)
    install_project_exception_handlers(app)
    return TestClient(app)


@pytest.mark.parametrize('detail', _DETAILS)
@pytest.mark.parametrize('status', [400, 404, 409, 422, 503])
def test_the_response_matches_an_http_exception(status: int, detail: Any) -> None:
    refused = _app(RequestRejectedError(status, detail)).get('/boom')
    raised = _app(HTTPException(status_code=status, detail=detail)).get('/boom')

    assert refused.status_code == raised.status_code == status
    assert refused.json() == raised.json() == {'detail': detail}


def test_a_service_rejection_has_the_body_the_router_helper_builds() -> None:
    fields: dict[str, Any] = {'project': 'alpha', 'project_status': 'archived'}
    refused = _app(rejection(409, 'invalid_transition', 'not allowed', **fields)).get('/boom')
    raised = _app(api_error(409, 'invalid_transition', 'not allowed', **fields)).get('/boom')

    assert refused.status_code == raised.status_code == 409
    assert refused.json() == raised.json()
    assert refused.json()['detail']['error'] == 'invalid_transition'
