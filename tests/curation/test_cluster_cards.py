"""``GET /clusters``: typed cards, server-side sort and filter, truncation and review counts."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from curation import query_fakes
from curation._cluster_cards_helpers import bucket, cards_by_id, list_cards, os_returning
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.schemas.curation.clusters import ClusterCard, ClusterListResponse
from src.services.curation import cluster_cards as cc, review_tab_clauses
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.item_filter import ItemFilter
from src.services.curation.review_request import ReviewFilters, build_review_request


CAND = RESIDUAL_CLUSTER_ID_OFFSET


def _card(cid: int, **over: Any) -> ClusterCard:
    base: dict[str, Any] = {
        'cluster_id': cid,
        'cluster_kind': 'class',
        'size': 10,
        'validated_count': 0,
        'labelled_count': 10,
        'dominant_count': 10,
        'purity_n': 0,
        'purity_basis': 'nearest_centroid',
        'promotable': False,
        'is_unlabeled': False,
        'n_subclusters': 0,
        'subcluster_ids': [],
        'representatives': [],
        'validation_state': 'none',
        'dominant_class_id': None,
        'dominant_class_name': None,
        'purity': None,
        'purity_tier': None,
        'label_agreement': None,
        'labelled_share': None,
        'updated_at': None,
        'detector_top_class': None,
        'detector_top_share': None,
        'detector_conf_mean': None,
        'review_counts': {},
        'needs_review_count': 0,
        'flags': [],
    }
    base.update(over)
    return base  # type: ignore[return-value]


def _fixture() -> list[ClusterCard]:
    return [
        _card(
            1,
            size=100,
            validated_count=100,
            validation_state='complete',
            label_agreement=0.95,
            purity=0.9,
            detector_conf_mean=0.9,
            updated_at='2026-01-01T00:00:00.000Z',
            dominant_class_name='car',
            detector_top_class='car',
            detector_top_share=0.99,
        ),
        _card(
            2,
            size=50,
            validated_count=10,
            validation_state='partial',
            label_agreement=0.5,
            purity=0.4,
            detector_conf_mean=0.3,
            needs_review_count=20,
            updated_at='2026-03-01T00:00:00.000Z',
            dominant_class_name='bus',
            detector_top_class='truck',
            detector_top_share=0.4,
            flags=['label_disagreement'],
        ),
        _card(
            3,
            cluster_kind='candidate',
            size=50,
            needs_review_count=20,
            dominant_class_name=None,
            flags=['needs_review', 'unvalidated'],
        ),
        _card(
            4,
            size=10,
            validated_count=5,
            validation_state='partial',
            label_agreement=0.7,
            purity=0.7,
            detector_conf_mean=0.6,
            needs_review_count=5,
            updated_at='2026-02-01T00:00:00.000Z',
            dominant_class_name='ant',
            detector_top_class='ant',
            detector_top_share=1.0,
            flags=['needs_review'],
        ),
    ]


def _ids(cards: list[ClusterCard]) -> list[int]:
    return [c['cluster_id'] for c in cards]


@pytest.mark.parametrize(
    ('sort', 'expected'),
    [
        ('needs_review_desc', [2, 3, 4, 1]),  # 2 and 3 tie: by cluster_id
        ('size_desc', [1, 2, 3, 4]),
        ('size_asc', [4, 2, 3, 1]),
        ('label_agreement_asc', [2, 4, 1, 3]),  # unmeasured last
        ('purity_asc', [2, 4, 1, 3]),
        ('purity_desc', [1, 4, 2, 3]),
        ('detector_conf_asc', [2, 4, 1, 3]),
        ('unvalidated_desc', [3, 2, 4, 1]),  # size minus validated: 50, 40, 5, 0
        ('updated_desc', [2, 4, 1, 3]),
        ('dominant_class', [4, 2, 1, 3]),  # ant, bus, car, unnamed last
    ],
)
def test_each_sort_orders_the_fixture(sort: str, expected: list[int]) -> None:
    cards = _fixture()
    assert _ids(cc.sort_cards(cards, sort)) == expected  # type: ignore[arg-type]
    # Input order never changes the result: ties break on cluster_id.
    assert _ids(cc.sort_cards(list(reversed(cards)), sort)) == expected  # type: ignore[arg-type]


def test_every_served_sort_option_is_implemented() -> None:
    ids = [o[0] for o in cc.SORT_OPTIONS]
    assert len(set(ids)) == len(ids)
    for sort_id in ids:
        assert sorted(_ids(cc.sort_cards(_fixture(), sort_id))) == [1, 2, 3, 4]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ('over', 'expected'),
    [
        ({'dominant_class': ('car', 'ant')}, [1, 4]),
        ({'dominant_class': ('Bus',)}, [2]),  # names normalise like the item filter
        ({'detector_class': ('truck',)}, [2]),
        ({'min_size': 50}, [1, 2, 3]),
        ({'max_size': 50}, [2, 3, 4]),
        ({'min_size': 20, 'max_size': 60}, [2, 3]),
        ({'validation_state': 'partial'}, [2, 4]),
        ({'validation_state': 'complete'}, [1]),
        ({'validation_state': 'none'}, [3]),
        ({'flags': ('needs_review',)}, [3, 4]),
        ({'flags': ('needs_review', 'label_disagreement')}, [2, 3, 4]),  # any of
        ({'needs_review': True}, [2, 3, 4]),
        ({'needs_review': False}, [1]),
    ],
)
def test_each_filter_removes_exactly_the_expected_cards(
    over: dict[str, Any], expected: list[int]
) -> None:
    got = cc.filter_cards(_fixture(), cc.ClusterListParams(**over))
    assert sorted(_ids(got)) == expected


@pytest.mark.asyncio
async def test_filters_and_sort_apply_before_representatives_paging() -> None:
    client = os_returning(
        bucket(1, 10, validated=10), bucket(2, 30, validated=0), bucket(3, 20, validated=0)
    )
    client.msearch = _msearch_recorder()
    resp = await list_cards(
        client, per_cluster=2, validation_state='none', sort='size_asc', offset=1, limit=1
    )
    assert [c['cluster_id'] for c in resp['items']] == [3, 2]
    assert resp['total'] == 2
    assert resp['sort'] == 'size_asc'
    # The page [1, 2) of the SERVED order is cluster 2 alone.
    lines = client.msearch.await_args.kwargs['body']
    assert len(lines) == 2
    assert [c['cluster_id'] for c in resp['items'] if c['representatives']] == [2]


def _msearch_recorder() -> Any:
    from unittest.mock import AsyncMock

    return AsyncMock(
        return_value={
            'responses': [{'hits': {'hits': [{'_id': 'c1', '_source': {'crop_id': 'c1'}}]}}]
        }
    )


@pytest.mark.asyncio
async def test_truncated_when_the_aggregation_hits_max_clusters() -> None:
    full = [bucket(i, 5) for i in range(1, 4)]
    at_cap = await list_cards(os_returning(*full), max_clusters=3)
    assert at_cap['truncated'] is True
    under = await list_cards(os_returning(*full), max_clusters=4)
    assert under['truncated'] is False


@pytest.mark.asyncio
async def test_truncation_reads_the_aggregation_not_the_filtered_list() -> None:
    full = [bucket(i, 5) for i in range(1, 4)]
    resp = await list_cards(os_returning(*full), max_clusters=3, min_size=999)
    assert resp['items'] == []
    assert resp['truncated'] is True


@pytest.mark.asyncio
async def test_response_validates_and_serves_vocabularies() -> None:
    resp = await cc.list_cluster_cards(
        os_returning(bucket(7, 5)), cc.ClusterListParams(per_cluster=0)
    )
    # Strict validation against the documented shape: a missing or mistyped field fails.
    TypeAdapter(ClusterListResponse).validate_python(resp, strict=True)
    assert resp['sort'] == 'size_desc'
    assert [o['id'] for o in resp['sort_options']] == [o[0] for o in cc.SORT_OPTIONS]
    assert {f['id'] for f in resp['flag_vocabulary']} >= {
        'promotable',
        'needs_review',
        'unvalidated',
    }


@pytest.mark.asyncio
async def test_new_fields_come_from_the_aggregation() -> None:
    cards = await cards_by_id(
        bucket(
            3,
            40,
            [('car', 40)],
            validated=10,
            detector=[('car', 30), ('van', 10)],
            detector_conf=0.75,
            review={'all': 7, 'outliers': 2},
        )
    )
    card = cards[3]
    assert card['detector_top_class'] == 'car'
    assert card['detector_top_share'] == pytest.approx(0.75)
    assert card['detector_conf_mean'] == 0.75
    assert card['validation_state'] == 'partial'
    assert card['review_counts']['all'] == 7
    assert card['review_counts']['outliers'] == 2
    assert set(card['review_counts']) == set(cc.CLUSTER_REVIEW_TABS)
    assert card['needs_review_count'] == 7
    assert 'needs_review' in card['flags']
    assert 'label_purity' not in card


@pytest.mark.asyncio
async def test_a_cluster_without_detector_classes_has_none_not_zero() -> None:
    card = (await cards_by_id(bucket(3, 40)))[3]
    assert card['detector_top_class'] is None
    assert card['detector_top_share'] is None
    assert card['detector_conf_mean'] is None


def test_review_tabs_are_real_simple_tabs() -> None:
    assert set(cc.CLUSTER_REVIEW_TABS) <= set(review_tab_clauses.SIMPLE_TABS)
    assert cc.CLUSTER_REVIEW_TABS[0] == 'all'


@pytest.mark.asyncio
async def test_the_aggregation_counts_each_review_tab_and_hides_the_holdout() -> None:
    client = os_returning()
    await list_cards(client)
    body = client.search.await_args.kwargs['body']
    sub = body['aggs']['clusters']['aggs']
    assert set(sub['review']['filters']['filters']) == set(cc.CLUSTER_REVIEW_TABS)
    assert {'test_holdout': True} in [
        m['term'] for m in body['query']['bool']['must_not'] if 'term' in m
    ]


# --- review counts equal what the review route serves (query-capable fake) -----------------

ITEMS = base_curation_config().items_index


def _disagrees(doc: dict[str, Any]) -> bool:
    detector, current = doc.get('detector_class_name'), doc.get('class_name')
    return bool(detector) and bool(current) and detector != current


def _item(crop_id: str, cluster_id: int, **kw: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'cluster_id': cluster_id,
        'class_id': 1,
        'class_name': 'widget',
        'class_source': 'vlm',
        'class_validated': False,
        'detector_class_name': 'widget',
        'detector_confidence': 0.8,
        'vlm_confidence': 'high',
        **kw,
    }


def _docs() -> dict[str, dict[str, Any]]:
    docs = [
        _item('a1', 1, vlm_confidence='low'),
        _item('a2', 1, detector_class_name='gadget'),
        _item('a3', 1, cluster_distance=0.5),
        _item('a4', 1, class_source='vlm_unmatched', class_name=None, class_id=None),
        _item('a5', 1),
        _item('a6', 1, class_validated=True, class_source='human'),
        _item('a7', 1, test_holdout=True, vlm_confidence='low'),
        _item('a8', 1, class_excluded=True, vlm_confidence='low'),
        _item('a9', 1, review_dismissed_at='2026-10-01T00:00:00+00:00', vlm_confidence='low'),
        _item('b1', 2, vlm_confidence='low'),
        _item('b2', 2, detector_class_name='gadget'),
    ]
    return {d['crop_id']: d for d in docs}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> QueryFakeOpenSearch:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, _disagrees
    )
    return QueryFakeOpenSearch({ITEMS: _docs()})


@pytest.fixture
def client(fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from unittest.mock import AsyncMock

    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


URL = '/curation/projects/default'


@pytest.mark.asyncio
async def test_review_counts_equal_what_the_review_query_matches(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    body = client.get(f'{URL}/clusters', params={'per_cluster': 0}).json()
    cards = {c['cluster_id']: c for c in body['items']}
    assert set(cards) == {1, 2}
    for cluster_id, card in cards.items():
        for tab in cc.CLUSTER_REVIEW_TABS:
            request = await build_review_request(
                tab, ReviewFilters(item=ItemFilter(cluster_id=cluster_id)), None, fake
            )
            matched = (await fake.count(index=ITEMS, body={'query': request.query}))['count']
            assert card['review_counts'][tab] == matched, (cluster_id, tab)
    # The route itself, across both clusters.
    for tab in cc.CLUSTER_REVIEW_TABS:
        served = client.get(f'{URL}/review/{tab}', params={'page_size': 100}).json()
        assert served['total'] == sum(c['review_counts'][tab] for c in cards.values()), tab
    # a1 (low VLM confidence), a3 (outlier) and a4 (unmatched) are in the queue; a2 only
    # disagrees with the detector (its own tab) and a5 is clean.
    assert cards[1]['review_counts']['all'] == 3
    assert cards[1]['needs_review_count'] == 3
    assert cards[1]['review_counts']['detector_disagreements'] == 1
    assert cards[2]['review_counts']['all'] == 1


def test_holdout_and_excluded_members_are_never_counted(client: TestClient) -> None:
    cards = {
        c['cluster_id']: c
        for c in client.get(f'{URL}/clusters', params={'per_cluster': 0}).json()['items']
    }
    # a1..a9: holdout (a7) and excluded (a8) are out of the size too.
    assert cards[1]['size'] == 7


def test_the_route_serves_filtered_sorted_cards(client: TestClient) -> None:
    body = client.get(
        f'{URL}/clusters', params={'per_cluster': 0, 'sort': 'size_asc', 'min_size': 1}
    ).json()
    assert [c['cluster_id'] for c in body['items']] == [2, 1]
    only = client.get(f'{URL}/clusters', params={'per_cluster': 0, 'needs_review': 'false'}).json()
    assert only['items'] == []
    flagged = client.get(
        f'{URL}/clusters', params={'per_cluster': 0, 'flag': ['needs_review', 'promotable']}
    ).json()
    assert [c['cluster_id'] for c in flagged['items']] == [1, 2]
    bad = client.get(f'{URL}/clusters', params={'sort': 'nonsense'})
    assert bad.status_code == 422
    bad_state = client.get(f'{URL}/clusters', params={'validation_state': 'most'})
    assert bad_state.status_code == 422
