"""Derived and pass-through item keys every endpoint serves.

The frontend used to fabricate these per page (``proposed_class_*``
outside ``/review``), recompute them (cluster kind from the id, similarity
from ``cluster_distance``) or read legacy storage names (``source``).
They now come from the one item serializer, so every item endpoint
(``/crops``, ``/crops/{id}``, ``/review``, ``/regions``, search, undo,
discard) carries them.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.cluster_ids import (
    CORE_SIMILARITY_MIN,
    RESIDUAL_CLUSTER_ID_OFFSET,
    cluster_kind,
)
from src.services.curation.wire import ITEM_WIRE_KEYS, REVIEW_EXTRA_KEYS, serialize_item


def _item(**doc: Any) -> dict[str, Any]:
    return serialize_item(doc, 'c1', api_prefix='')


# ------------------------------------------------------------- proposed class


def test_proposed_class_is_the_vlm_suggestion_when_there_is_one() -> None:
    item = _item(class_id=3, class_name='thing', class_source='vlm', class_validated=False)
    assert (item['proposed_class_id'], item['proposed_class_name']) == (3, 'thing')


def test_proposed_class_for_a_new_class_proposal_has_no_id() -> None:
    item = _item(
        class_id=7,
        class_name='old',
        class_source='vlm_new_class_pending',
        vlm_proposed_class='widget_xl',
    )
    assert (item['proposed_class_id'], item['proposed_class_name']) == (None, 'widget_xl')


def test_proposed_class_falls_back_to_current_class_or_raw_answer() -> None:
    item = _item(class_id=5, class_name='gadget', class_source='det_model', class_validated=True)
    assert (item['proposed_class_id'], item['proposed_class_name']) == (5, 'gadget')
    item = _item(class_source='vlm_unmatched', vlm_raw_class='mystery')
    assert (item['proposed_class_id'], item['proposed_class_name']) == (None, 'mystery')
    item = _item()
    assert (item['proposed_class_id'], item['proposed_class_name']) == (None, '')


def test_proposed_class_is_no_longer_a_review_only_extra() -> None:
    assert {'proposed_class_id', 'proposed_class_name'} <= ITEM_WIRE_KEYS
    assert frozenset({'reason'}) == REVIEW_EXTRA_KEYS


# --------------------------------------------------------------- cluster facts


@pytest.mark.parametrize(
    ('cid', 'kind'),
    [
        (None, None),
        (-2, 'unassigned'),
        (0, 'class'),
        (RESIDUAL_CLUSTER_ID_OFFSET - 1, 'class'),
        (RESIDUAL_CLUSTER_ID_OFFSET, 'candidate'),
    ],
)
def test_cluster_kind(cid: int | None, kind: str | None) -> None:
    assert cluster_kind(cid) == kind
    assert _item(cluster_id=cid)['cluster_kind'] == kind


def test_cluster_similarity_and_core_flag() -> None:
    near = _item(cluster_id=3, cluster_distance=0.2)
    assert near['cluster_similarity'] == pytest.approx(0.8)
    assert near['cluster_is_core'] is (CORE_SIMILARITY_MIN <= 0.8)
    far = _item(cluster_id=3, cluster_distance=1.5)
    assert far['cluster_similarity'] == 0.0
    assert far['cluster_is_core'] is False
    none = _item(cluster_id=3)
    assert none['cluster_similarity'] is None
    assert none['cluster_is_core'] is None


# ------------------------------------------------------------ pass-through keys


def test_new_class_exclusion_probe_and_source_keys() -> None:
    item = _item(
        needs_new_class=True,
        needs_new_class_note='looks like a trailer',
        class_excluded=True,
        excluded_reason='blurry',
        excluded_at='2026-09-24T00:00:00+00:00',
        probe_pred_class='gadget',
        probe_pred_class_id=5,
        source='disk_a',
    )
    assert item['needs_new_class'] is True
    assert item['needs_new_class_note'] == 'looks like a trailer'
    assert item['class_excluded'] is True
    assert item['excluded_reason'] == 'blurry'
    assert item['excluded_at'] == '2026-09-24T00:00:00+00:00'
    assert item['probe_pred_class_id'] == 5
    assert item['source'] == 'disk_a'


def test_pass_through_defaults() -> None:
    item = _item()
    assert item['needs_new_class'] is False
    assert item['class_excluded'] is False
    assert item['source'] == ''
    assert item['probe_pred_class_id'] is None


# ------------------------------------------------------------- GET /crops?ids=


def test_crops_by_ids_returns_items_in_request_order() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from curation.query_fakes import QueryFakeOpenSearch
    from src.config import get_curation_config
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    items = get_curation_config().items_index
    fake = QueryFakeOpenSearch(
        {items: {k: {'crop_id': k, 'class_id': 1, 'class_source': 'vlm'} for k in 'abc'}}
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        r = client.get('/curation/crops', params={'ids': 'c,gone,a'})
        assert r.status_code == 200, r.text
        body = r.json()
        assert [c['crop_id'] for c in body['crops']] == ['c', 'a']
        assert body['total'] == 2
        assert body['crops'][0]['proposed_class_id'] == 1
        too_many = ','.join(f'x{i}' for i in range(501))
        assert client.get('/curation/crops', params={'ids': too_many}).status_code == 400


def test_crops_needs_new_class_and_source_filters() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from curation.query_fakes import QueryFakeOpenSearch
    from src.config import get_curation_config
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    items = get_curation_config().items_index
    fake = QueryFakeOpenSearch(
        {
            items: {
                'f': {'crop_id': 'f', 'needs_new_class': True, 'source': 'disk_a'},
                'n': {'crop_id': 'n', 'source': 'disk_b'},
            }
        }
    )
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        r = client.get('/curation/crops', params={'needs_new_class': 'true'})
        assert [c['crop_id'] for c in r.json()['crops']] == ['f']
        r = client.get('/curation/crops', params={'source': 'disk_b'})
        assert [c['crop_id'] for c in r.json()['crops']] == ['n']
