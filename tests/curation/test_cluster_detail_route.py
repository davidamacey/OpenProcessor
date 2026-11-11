"""``GET {P}/clusters/{cluster_uid}``: the one served card of a cluster, 404 with the reason
when the serving partition has no such cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from curation._cluster_cards_helpers import bucket, curation_client, os_returning, uid


if TYPE_CHECKING:
    import pytest


URL = '/curation/projects/default'


def _client(monkeypatch: pytest.MonkeyPatch, *buckets: dict[str, Any]) -> tuple[Any, AsyncMock]:
    fake = os_returning(*buckets)
    fake.msearch = AsyncMock(
        return_value={
            'responses': [
                {
                    'hits': {
                        'hits': [{'_id': 'c-1', '_source': {'crop_id': 'c-1', 'class_name': 'van'}}]
                    }
                }
            ]
        }
    )
    return curation_client(fake, monkeypatch), fake


def test_serves_the_card_with_its_representatives(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = _client(monkeypatch, bucket(7, 12, [('van', 12)]))
    resp = client.get(f'{URL}/clusters/{uid(7)}')
    assert resp.status_code == 200, resp.text
    card = resp.json()
    assert card['cluster_uid'] == uid(7)
    assert card['size'] == 12
    assert [r['crop_id'] for r in card['representatives']] == ['c-1']
    body = fake.search.await_args.kwargs['body']
    assert {'term': {'cluster_uid': uid(7)}} in body['query']['bool']['filter']


def test_a_cluster_the_partition_does_not_serve_is_a_404_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _client(monkeypatch)
    resp = client.get(f'{URL}/clusters/{uid(9)}')
    assert resp.status_code == 404
    detail = resp.json()['detail']
    assert detail['error'] == 'cluster_not_found'
    assert uid(9) in detail['message']


def test_the_literal_routes_still_win_over_the_uid_route(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch)
    assert client.get(f'{URL}/clusters/representatives').status_code == 200


def test_the_literal_partition_routes_are_registered_before_the_uid_route() -> None:
    from fastapi.routing import APIRoute
    from starlette.routing import Match

    from src.routers.curation import router

    def first(path: str) -> str:
        scope = {'type': 'http', 'path': path, 'method': 'GET'}
        for route in router.routes:
            if isinstance(route, APIRoute) and route.matches(scope)[0] == Match.FULL:
                return route.path
        raise AssertionError(path)

    assert first('/clusters/partitions') == '/clusters/partitions'
    assert first('/clusters/representatives') == '/clusters/representatives'
    assert first(f'/clusters/{uid(1)}') == '/clusters/{cluster_uid}'
