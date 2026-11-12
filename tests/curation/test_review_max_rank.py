"""DQ-M6: ``max_rank`` (subject size) is honoured on every review tab that
serves it, and ``GET /review/tabs`` says which filters each tab applies.

Before the fix only ``primary_low_conf`` / ``classifier_blind_spots`` applied
``crop_rank_in_image <= max_rank``; every other tab silently ignored it,
so a client's "largest subject only" control changed nothing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from _msearch_fake import install_msearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation_accessors import base_curation_config
from src.domain.ingest_class_sources import unlabeled_proposal_class_sources
from src.services.curation.review_tabs import QUEUES, tab_filters


# The regions queue exists only while a region profile is active (GH #51).
pytestmark = pytest.mark.usefixtures('reference_region_profile')


ITEMS = base_curation_config().items_index
F = get_region_fields()

# One doc body per queue that qualifies for that queue.
_QUALIFYING: dict[str, dict[str, Any]] = {
    'to_label': {},
    'check_machine': {'class_id': 1, 'class_source': 'vlm', 'confidence': 0.3},
    'disagreements': {'class_source': 'vlm_unmatched'},
    'audit': {'audit_sample': True},
    'regions': {
        F.boxes: [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}]
    },
    'new_class_proposals': {'needs_new_class': True},
}
# `review_tab` filter values that are rank-limited by definition.
_PRIMARY_FILTER_VALUES: dict[str, dict[str, Any]] = {
    'primary_low_conf': {'class_source': sorted(unlabeled_proposal_class_sources())[0]},
    'classifier_blind_spots': {'class_source': sorted(unlabeled_proposal_class_sources())[0]},
}


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    install_msearch(fake)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        'src.services.curation.strategy_registry.resolve_effective_default',
        AsyncMock(return_value=None),
    )
    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _docs(tab: str) -> dict[str, dict[str, Any]]:
    body = {**_QUALIFYING, **_PRIMARY_FILTER_VALUES}[tab]
    return {
        f'rank{rank}': {'crop_id': f'rank{rank}', 'crop_rank_in_image': rank, **body}
        for rank in (1, 2, 3, 5)
    }


@pytest.mark.parametrize('tab', sorted(_QUALIFYING))
def test_max_rank_limits_every_tab(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get(
        f'/curation/projects/default/review/{tab}', params={'max_rank': 1, 'page_size': 50}
    )
    assert r.status_code == 200, r.text
    assert {i['crop_id'] for i in r.json()['items']} == {'rank1'}

    r3 = client.get(
        f'/curation/projects/default/review/{tab}', params={'max_rank': 3, 'page_size': 50}
    )
    assert {i['crop_id'] for i in r3.json()['items']} == {'rank1', 'rank2', 'rank3'}


@pytest.mark.parametrize('tab', sorted(_QUALIFYING))
def test_no_max_rank_serves_every_rank(tab: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get(f'/curation/projects/default/review/{tab}', params={'page_size': 50})
    assert r.json()['total'] == 4


@pytest.mark.parametrize('tab', sorted(_PRIMARY_FILTER_VALUES))
def test_primary_filter_values_keep_their_rank_default(
    tab: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs(tab)}), monkeypatch)
    r = client.get('/curation/projects/default/crops', params={'review_tab': tab})
    assert {c['crop_id'] for c in r.json()['crops']} == {'rank1', 'rank2'}


def test_locate_uses_the_same_max_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs('to_label')}), monkeypatch)
    body = client.get(
        '/curation/projects/default/review/to_label/locate',
        params={'crop_id': 'rank3', 'max_rank': 2},
    ).json()
    assert body['in_queue'] is False
    assert body['reason'] == 'filtered_out'
    assert body['total'] == 2


@pytest.mark.usefixtures('reference_region_profile')
def test_tabs_catalog_serves_filters_per_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: {}}), monkeypatch)
    tabs = {t['id']: t for t in client.get('/curation/projects/default/review/tabs').json()['tabs']}
    # The audit queue exists only while a drawn sample is pending.
    assert set(tabs) == set(QUEUES) - {'audit'}
    for tab_id, tab in tabs.items():
        assert tab['filters'] == list(tab_filters(tab_id))
        assert 'max_rank' in tab['filters']
        assert isinstance(tab['filter_defaults'], dict)
    # text search is a regions-only filter: no other tab may advertise it.
    assert [t for t, spec in tabs.items() if 'text' in spec['filters']] == ['regions']


def test_text_is_ignored_off_the_regions_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(QueryFakeOpenSearch({ITEMS: _docs('to_label')}), monkeypatch)
    r = client.get(
        '/curation/projects/default/review/to_label',
        params={'text': 'nothing-matches', 'page_size': 50},
    )
    assert r.json()['total'] == 4
