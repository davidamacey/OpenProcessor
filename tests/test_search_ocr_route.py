"""Found live: ``POST /search/ocr`` answered 500 for any real match -- the
result model capped ``score`` at 1.0 but text relevance is an unbounded
BM25 score."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.dependencies import get_visual_search_service
from src.routers.search import router


class _Opensearch:
    async def search_ocr(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                'image_id': 'img-1',
                'image_path': 'a.jpg',
                'score': 5.06,
                'text': 'CAUTION',
                'box_normalized': [0.1, 0.2, 0.3, 0.4],
            }
        ]


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_visual_search_service] = lambda: SimpleNamespace(
        opensearch=_Opensearch()
    )
    return TestClient(app)


def test_a_bm25_score_above_one_is_served(client: TestClient) -> None:
    resp = client.post('/search/ocr', params={'text': 'caution', 'min_score': 2.0})

    assert resp.status_code == 200, resp.text
    (hit,) = resp.json()['results']
    assert hit['score'] == pytest.approx(5.06)
    assert hit['matched_text'] == 'CAUTION'
    assert hit['text_box'] == [0.1, 0.2, 0.3, 0.4]
