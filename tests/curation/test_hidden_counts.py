"""List pages say how many items they hide (test holdout, human-ignored)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.curation.hidden_counts import hidden_counts


def _os(holdout: int, excluded: int) -> AsyncMock:
    fake = AsyncMock()
    fake.search = AsyncMock(
        return_value={
            'aggregations': {
                'hidden': {
                    'buckets': {
                        'test_holdout': {'doc_count': holdout},
                        'excluded': {'doc_count': excluded},
                    }
                }
            }
        }
    )
    return fake


@pytest.mark.asyncio
async def test_counts_what_the_default_views_hide() -> None:
    fake = _os(25, 4)
    assert await hidden_counts(fake, [], include_test=False, include_excluded=False) == {
        'test_holdout': 25,
        'excluded': 4,
    }
    body: dict[str, Any] = fake.search.await_args.kwargs['body']
    assert body['size'] == 0


@pytest.mark.asyncio
async def test_counts_follow_the_requests_filter() -> None:
    fake = _os(3, 1)
    clause = {'term': {'class_name': 'car'}}
    await hidden_counts(fake, [clause], include_test=False, include_excluded=False)
    body: dict[str, Any] = fake.search.await_args.kwargs['body']
    assert body['query'] == {'bool': {'filter': [clause]}}


@pytest.mark.asyncio
async def test_what_the_page_includes_is_not_hidden() -> None:
    fake = _os(25, 4)
    assert await hidden_counts(fake, [], include_test=True, include_excluded=True) == {
        'test_holdout': 0,
        'excluded': 0,
    }
    fake.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_count_fails_the_page_instead_of_reporting_zero() -> None:
    fake = AsyncMock()
    fake.search = AsyncMock(side_effect=RuntimeError('down'))
    with pytest.raises(RuntimeError):
        await hidden_counts(fake, [], include_test=False, include_excluded=False)


def test_crops_and_review_pages_serve_what_they_hide(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.routers.curation import _raw_opensearch_dep, router as curation_router
    from src.services.curation import index_bootstrap

    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    fake = AsyncMock()

    async def search(*, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        if body.get('size') == 0 and 'hidden' in (body.get('aggs') or {}):
            return {
                'aggregations': {
                    'hidden': {
                        'buckets': {
                            'test_holdout': {'doc_count': 25},
                            'excluded': {'doc_count': 4},
                        }
                    }
                }
            }
        return {'hits': {'total': {'value': 0}, 'hits': []}}

    fake.search = search
    fake.count = AsyncMock(return_value={'count': 0})
    from _curation_app import mount_curation_routers

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    client = TestClient(app)
    crops = client.get('/curation/projects/default/crops').json()
    assert crops['hidden'] == {'test_holdout': 25, 'excluded': 4}
    shown = client.get('/curation/projects/default/crops', params={'include_test': True}).json()
    assert shown['hidden'] == {'test_holdout': 0, 'excluded': 4}
    review = client.get('/curation/projects/default/review/all').json()
    assert review['hidden'] == {'test_holdout': 25, 'excluded': 4}
