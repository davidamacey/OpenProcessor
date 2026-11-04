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
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search_ocr_page(
        self, query_text: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], int]:
        self.calls.append((query_text, kwargs))
        return (
            [
                {
                    'image_id': 'img-1',
                    'image_path': 'a.jpg',
                    'score': 5.06,
                    'text': 'CAUTION',
                    'box_normalized': [0.1, 0.2, 0.3, 0.4],
                }
            ],
            41,
        )


@pytest.fixture
def opensearch() -> _Opensearch:
    return _Opensearch()


@pytest.fixture
def client(opensearch: _Opensearch) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_visual_search_service] = lambda: SimpleNamespace(
        opensearch=opensearch
    )
    return TestClient(app)


def test_a_bm25_score_above_one_is_served(client: TestClient) -> None:
    resp = client.post('/search/ocr', params={'text': 'caution', 'min_score': 2.0})

    assert resp.status_code == 200, resp.text
    (hit,) = resp.json()['results']
    assert hit['score'] == pytest.approx(5.06)
    assert hit['matched_text'] == 'CAUTION'
    assert hit['text_box'] == [0.1, 0.2, 0.3, 0.4]


def test_paging_and_exact_match_reach_the_index_and_total_counts_all_images(
    client: TestClient, opensearch: _Opensearch
) -> None:
    resp = client.post(
        '/search/ocr', params={'text': 'stop', 'top_k': 20, 'page': 2, 'fuzzy': 'false'}
    )

    assert resp.status_code == 200, resp.text
    ((text, kwargs),) = opensearch.calls
    assert text == 'stop'
    assert (kwargs['offset'], kwargs['size'], kwargs['exact']) == (40, 20, True)
    body = resp.json()
    assert (body['total_results'], len(body['results'])) == (41, 1)


def test_the_defaults_are_the_first_fuzzy_page(client: TestClient, opensearch: _Opensearch) -> None:
    client.post('/search/ocr', params={'text': 'stop'})

    ((_, kwargs),) = opensearch.calls
    assert (kwargs['offset'], kwargs['size'], kwargs['exact']) == (0, 10, False)


def test_exactly_one_ocr_text_search_route_is_mounted() -> None:
    from src.main import app

    paths = [
        p
        for p, ops in app.openapi()['paths'].items()
        if 'post' in ops and not p.startswith('/v1') and 'ocr' in p and 'search' in p
    ]

    assert paths == ['/search/ocr']
