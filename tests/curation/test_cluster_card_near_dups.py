"""A cluster card counts the near-duplicate groups of its members (served, not recomputed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

from curation._cluster_cards_helpers import bucket, cards_by_id, curation_client, uid
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.schemas.curation.clusters import ClusterListResponse
from src.services.curation import cluster_cards as cc
from src.services.embedding_spaces.readers import WriteFields


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
URL = '/curation/projects/default'


@pytest.mark.asyncio
async def test_near_dup_group_count_comes_from_the_aggregation() -> None:
    cards = await cards_by_id(bucket(4, 10, near_dup_groups=3), bucket(5, 10))
    assert cards[uid(4)]['near_dup_group_count'] == 3
    assert cards[uid(5)]['near_dup_group_count'] == 0


def test_the_aggregation_asks_for_the_groups() -> None:
    fields = WriteFields(vector_field='pe_embedding', state_field='embedding_state')
    sub = cc.aggregation_body(cc.ClusterListParams(), fields)['aggs']['clusters']['aggs']
    assert sub['near_dup_groups'] == {'cardinality': {'field': 'near_dup_group_id'}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from curation import query_fakes
    from src.services.curation import review_tab_clauses

    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS,
        review_tab_clauses.DETECTOR_DISAGREES_SCRIPT,
        lambda _d: False,
    )
    docs: list[dict[str, Any]] = [
        {'crop_id': 'a1', 'cluster_uid': uid(1), 'near_dup_group_id': 'g1'},
        {'crop_id': 'a2', 'cluster_uid': uid(1), 'near_dup_group_id': 'g1'},
        {'crop_id': 'a3', 'cluster_uid': uid(1), 'near_dup_group_id': 'g2'},
        {'crop_id': 'a4', 'cluster_uid': uid(1)},
        {'crop_id': 'b1', 'cluster_uid': uid(2)},
    ]
    fake = QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}})
    return curation_client(fake, monkeypatch)


def test_the_route_serves_the_group_count(client: TestClient) -> None:
    body = client.get(f'{URL}/clusters', params={'per_cluster': 0}).json()
    TypeAdapter(ClusterListResponse).validate_python(body)
    counts = {c['cluster_uid']: c['near_dup_group_count'] for c in body['items']}
    assert counts == {uid(1): 2, uid(2): 0}
