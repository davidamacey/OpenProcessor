"""DQ-M6: ``max_rank`` (subject size) is honoured on every review tab that
serves it, and ``GET /review/tabs`` says which filters each tab applies.

Before the fix only ``primary_low_conf`` / ``classifier_blind_spots`` applied
``crop_rank_in_image <= max_rank``; every other tab silently ignored it,
so a client's "largest subject only" control changed nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields
from src.services.curation.ingest_class_sources import unlabeled_proposal_class_sources
from src.services.curation.review_queries import KNOWN_TABS, review_tab_catalog, tab_filters


if TYPE_CHECKING:
    from collections.abc import Iterator


ITEMS = get_curation_config().items_index
F = get_region_fields()

# One doc body per tab that qualifies for that tab's queue.
_QUALIFYING: dict[str, dict[str, Any]] = {
    'all': {'class_source': 'vlm_unmatched'},
    'mismatches': {'class_source': 'vlm_unmatched'},
    'vlm_low_conf': {'class_source': 'vlm', 'vlm_confidence': 'low', 'confidence': 0.3},
    'outliers': {'cluster_distance': 0.6},
    'uncertainty': {'probe_pred_entropy': 1.5},
    'regions': {F.bbox_norm: [0.1, 0.1, 0.2, 0.2]},
    'new_class_proposals': {'needs_new_class': True},
    'primary_low_conf': {'class_source': sorted(unlabeled_proposal_class_sources())[0]},
    'classifier_blind_spots': {'class_source': sorted(unlabeled_proposal_class_sources())[0]},
}


@pytest.fixture(autouse=True)
def _fresh_sort_coverage() -> Iterator[None]:
    # These fakes lack most sort fields; never leak a 0%-coverage cache
    # (and so a fallback sort) into other tests, or inherit one.
    from src.services.curation.strategy_registry import _reset_field_coverage_cache

    _reset_field_coverage_cache()
    yield
    _reset_field_coverage_cache()


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _docs(tab: str) -> dict[str, dict[str, Any]]:
    body = _QUALIFYING[tab]
    return {
        f'rank{rank}': {'crop_id': f'rank{rank}', 'crop_rank_in_image': rank, **body}
        for rank in (1, 2, 3, 5)
    }


@pytest.mark.parametrize('tab', sorted(_QUALIFYING))
def test_max_rank_limits_every_tab(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get(f'/curation/review/{tab}', params={'max_rank': 1, 'page_size': 50})
    assert r.status_code == 200, r.text
    assert {i['crop_id'] for i in r.json()['items']} == {'rank1'}

    r3 = client.get(f'/curation/review/{tab}', params={'max_rank': 3, 'page_size': 50})
    assert {i['crop_id'] for i in r3.json()['items']} == {'rank1', 'rank2', 'rank3'}


@pytest.mark.parametrize(
    'tab', sorted(set(_QUALIFYING) - {'primary_low_conf', 'classifier_blind_spots'})
)
def test_no_max_rank_serves_every_rank(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get(f'/curation/review/{tab}', params={'page_size': 50})
    assert r.json()['total'] == 4


@pytest.mark.parametrize('tab', ['primary_low_conf', 'classifier_blind_spots'])
def test_primary_tabs_keep_their_served_default(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get(f'/curation/review/{tab}', params={'page_size': 50})
    assert {i['crop_id'] for i in r.json()['items']} == {'rank1', 'rank2'}
    served = {t['id']: t for t in review_tab_catalog()}[tab]
    assert served['filter_defaults'] == {'max_rank': 2}


def test_locate_uses_the_same_max_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs('all')}), monkeypatch)
    body = client.get(
        '/curation/review/all/locate', params={'crop_id': 'rank3', 'max_rank': 2}
    ).json()
    assert body['in_queue'] is False
    assert body['reason'] == 'filtered_out'
    assert body['total'] == 2


@pytest.mark.usefixtures('reference_region_profile')
def test_tabs_catalog_serves_filters_per_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: {}}), monkeypatch)
    tabs = {t['id']: t for t in client.get('/curation/review/tabs').json()['tabs']}
    assert set(tabs) == set(KNOWN_TABS)
    for tab_id, tab in tabs.items():
        assert tab['filters'] == list(tab_filters(tab_id))
        assert 'max_rank' in tab['filters']
        assert isinstance(tab['filter_defaults'], dict)
    # text search is a regions-only filter: no other tab may advertise it.
    assert [t for t, spec in tabs.items() if 'text' in spec['filters']] == ['regions']


def test_text_is_ignored_off_the_regions_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs('all')}), monkeypatch)
    r = client.get('/curation/review/all', params={'text': 'nothing-matches', 'page_size': 50})
    assert r.json()['total'] == 4
