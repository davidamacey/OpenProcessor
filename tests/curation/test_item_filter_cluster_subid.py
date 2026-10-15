"""Sub-clusters are addressable on the server: ``ItemFilter.cluster_subid`` and
``GET /crops?cluster_subid=`` filter by the stored sub-id."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default/crops'


def _doc(crop_id: str, cluster_id: int, subid: str | None) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'bbox_norm': [0.1, 0.2, 0.3, 0.5],
        'class_source': 'unlabeled_proposal',
        'cluster_id': cluster_id,
        'cluster_subid': subid,
        'updated_at': '2026-10-01T00:00:00+00:00',
    }


def test_the_filter_makes_a_term_clause() -> None:
    clauses = item_filter_clauses(ItemFilter(cluster_subid='47b'))
    assert {'term': {'cluster_subid': '47b'}} in clauses


def test_unset_adds_no_clause() -> None:
    assert item_filter_clauses(ItemFilter()) == []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.services.curation.index_bootstrap.ensure_indexes', _noop)
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _doc('a1', 47, '47a'),
                'b1': _doc('b1', 47, '47b'),
                'b2': _doc('b2', 47, '47b'),
                'c1': _doc('c1', 48, '48b'),
                'none': _doc('none', 47, None),
            }
        }
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(URL, params=params)
    assert r.status_code == 200, r.text
    return {c['crop_id'] for c in r.json()['crops']}


def test_crops_filter_by_subid_with_the_cluster(client: TestClient) -> None:
    assert _ids(client, cluster_id=47, cluster_subid='47b') == {'b1', 'b2'}


def test_subid_alone_is_accepted_because_it_embeds_the_cluster(client: TestClient) -> None:
    assert _ids(client, cluster_subid='47b') == {'b1', 'b2'}
    assert _ids(client, cluster_subid='48b') == {'c1'}
