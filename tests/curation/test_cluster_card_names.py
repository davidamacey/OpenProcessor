"""A cluster card names itself the way an item does.

The clusters are built from the residual pool (no confident class), so the dominant class is
often null; the card then falls back to the detector's own label, with the same
`display_name` / `display_name_kind` an item serves (`wire.display_name_of`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

from curation import query_fakes
from curation._cluster_cards_helpers import bucket, cards_by_id, curation_client, uid
from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config
from src.schemas.curation.clusters import ClusterListResponse
from src.services.curation import review_tab_clauses


if TYPE_CHECKING:
    from fastapi.testclient import TestClient

ITEMS = base_curation_config().items_index
URL = '/curation/projects/default'


@pytest.mark.asyncio
async def test_a_class_names_the_card() -> None:
    card = (await cards_by_id(bucket(1, 10, [('van', 8), ('bus', 2)], detector=[('car', 10)])))[
        uid(1)
    ]
    assert (card['display_name'], card['display_name_kind']) == ('van', 'class')


@pytest.mark.asyncio
async def test_the_detector_class_names_a_card_with_no_class() -> None:
    card = (await cards_by_id(bucket(2, 10, [], detector=[('dog', 7), ('cat', 3)])))[uid(2)]
    assert card['dominant_class_name'] is None
    assert (card['display_name'], card['display_name_kind']) == ('dog', 'detector')


@pytest.mark.asyncio
async def test_a_card_nothing_names_says_none() -> None:
    card = (await cards_by_id(bucket(3, 10, [])))[uid(3)]
    assert (card['display_name'], card['display_name_kind']) == ('', 'none')


def _item(crop_id: str, cluster: int, **extra: Any) -> dict[str, Any]:
    return {'crop_id': crop_id, 'cluster_uid': uid(cluster), **extra}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setitem(
        query_fakes.SCRIPT_EVALUATORS,
        review_tab_clauses.DETECTOR_DISAGREES_SCRIPT,
        lambda _d: False,
    )
    docs = [
        _item('a1', 1, detector_class_name='dog'),
        _item('a2', 1, detector_class_name='dog'),
        _item('a3', 1, detector_class_name='cat'),
        _item('a4', 1),
        _item('b1', 2, class_id=3, class_name='van', class_source='human'),
    ]
    fake = QueryFakeOpenSearch({ITEMS: {d['crop_id']: d for d in docs}})
    return curation_client(fake, monkeypatch)


def test_the_route_serves_the_name(client: TestClient) -> None:
    body = client.get(f'{URL}/clusters', params={'per_cluster': 0}).json()
    TypeAdapter(ClusterListResponse).validate_python(body)
    cards = {c['cluster_uid']: c for c in body['items']}
    one, two = cards[uid(1)], cards[uid(2)]
    assert (one['display_name'], one['display_name_kind']) == ('dog', 'detector')
    assert (two['display_name'], two['display_name_kind']) == ('van', 'class')
