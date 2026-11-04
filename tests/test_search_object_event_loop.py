"""``POST /search/object`` keeps the event loop free during detection and refuses a class that
has no per-object index instead of querying the whole-image index with a box vector."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from src.core.dependencies import get_visual_search_service
from src.core.exceptions import RequestRejectedError
from src.routers.search import router
from src.services.visual_search.search import SearchMixin


CAR, CHAIR = 2, 56
SLEEP_S = 0.3


class _Opensearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search_vehicles(self, **_kw: Any) -> list[dict[str, Any]]:
        self.calls.append('vehicles')
        return [{'image_id': 'img-1', 'score': 0.9, 'box': [0, 0, 1, 1], 'class_id': CAR}]

    async def search_people(self, **_kw: Any) -> list[dict[str, Any]]:
        self.calls.append('people')
        return []

    async def search_global(self, **_kw: Any) -> list[dict[str, Any]]:
        self.calls.append('global')
        return []


class _SlowTriton:
    def __init__(self, class_id: int) -> None:
        self.class_id = class_id

    def infer_yolo_clip_cpu(self, _image_bytes: bytes) -> dict[str, Any]:
        time.sleep(SLEEP_S)  # a blocking sync Triton call
        return {
            'num_dets': 1,
            'classes': [self.class_id],
            'scores': [0.9],
            'normalized_boxes': [np.array([0.1, 0.1, 0.5, 0.5])],
            'box_embeddings': [np.zeros(512, dtype=np.float32)],
        }


class _Service(SearchMixin):
    def __init__(self) -> None:
        self.opensearch = _Opensearch()  # type: ignore[assignment]


def _app(service: _Service, monkeypatch: pytest.MonkeyPatch, class_id: int) -> FastAPI:
    monkeypatch.setattr(
        'src.clients.triton_client.get_sync_triton_client', lambda _url: _SlowTriton(class_id)
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_visual_search_service] = lambda: service

    @app.exception_handler(RequestRejectedError)
    async def _rejected(_request, exc: RequestRejectedError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})

    return app


def _post(client: httpx.AsyncClient):
    return client.post('/search/object', files={'image': ('q.jpg', b'x', 'image/jpeg')})


@pytest.mark.asyncio
async def test_two_concurrent_requests_overlap_their_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _Service()
    app = _app(service, monkeypatch, CAR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://t') as client:
        started = time.perf_counter()
        first, second = await asyncio.gather(_post(client), _post(client))
        elapsed = time.perf_counter() - started
    assert first.status_code == second.status_code == 200
    assert elapsed < 0.45, f'the two {SLEEP_S}s detections ran back to back ({elapsed:.2f}s)'


@pytest.mark.asyncio
async def test_a_vehicle_query_returns_vehicle_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _Service()
    app = _app(service, monkeypatch, CAR)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
        resp = await _post(c)
    assert resp.status_code == 200
    assert [r['image_id'] for r in resp.json()['results']] == ['img-1']
    assert service.opensearch.calls == ['vehicles']  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_class_with_no_object_index_is_422_and_never_queries_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _Service()
    app = _app(service, monkeypatch, CHAIR)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://t') as c:
        resp = await _post(c)
    assert resp.status_code == 422
    assert resp.json()['detail']['error'] == 'object_class_not_indexed'
    assert service.opensearch.calls == []  # type: ignore[attr-defined]
