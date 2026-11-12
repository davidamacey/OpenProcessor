"""``GET /stats/classes?sort=``: the server orders the class rows and serves the orders it
accepts (``sort_options``), plus the held-out test count the Export table sorts by."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.class_counts import parse_class_counts


pytest_plugins = ['_records_world']

BASE = '/curation/projects/default'
BUCKETS = [
    # key, total, validated, holdout, excluded
    (1, 10, 8, 1, 0),
    (2, 900, 800, 5, 0),
    (3, 20, 4, 0, 1),
]
NAMES = {1: 'bus', 2: 'Apple', 3: 'cat'}
TEST_COUNTS = {1: 4, 2: 1, 3: 9}


def _aggs() -> dict[str, Any]:
    return {
        'by_class': {
            'buckets': [
                {
                    'key': k,
                    'doc_count': n,
                    'validated': {'doc_count': v},
                    'validated_test_holdout': {'doc_count': h},
                    'validated_excluded': {'doc_count': x},
                }
                for k, n, v, h, x in BUCKETS
            ]
        }
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    aggs = _aggs()
    monkeypatch.setattr(
        'src.services.curation.stats_classes.fetch_class_counts',
        AsyncMock(return_value=(parse_class_counts(aggs), aggs)),
    )
    monkeypatch.setattr(
        'src.services.curation.stats_classes.holdout_stats',
        AsyncMock(
            return_value={
                'by_class': [
                    {'key': k, 'doc_count': n, 'deficient': n < 5} for k, n in TEST_COUNTS.items()
                ]
            }
        ),
    )
    registry = SimpleNamespace(
        snapshot=AsyncMock(
            return_value=SimpleNamespace(
                classes=[
                    SimpleNamespace(class_id=k, class_name=name, deprecated=False)
                    for k, name in NAMES.items()
                ]
            )
        )
    )
    monkeypatch.setattr('src.routers.curation.stats.get_class_registry', lambda: registry)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()
    return TestClient(app)


def _ids(client: TestClient, sort: str | None) -> list[int]:
    r = client.get(BASE + '/stats/classes', params={'sort': sort} if sort else {})
    assert r.status_code == 200, r.text
    return [c['class_id'] for c in r.json()['classes']]


@pytest.mark.parametrize(
    ('sort', 'expected'),
    [
        (None, [1, 2, 3]),
        ('class_name:asc', [2, 1, 3]),
        ('class_id:desc', [3, 2, 1]),
        ('count:desc', [2, 3, 1]),
        ('validated_count:asc', [3, 1, 2]),
        ('trainable:desc', [2, 1, 3]),
        ('aug_target:asc', [1, 3, 2]),
        ('trainable_gap:desc', [3, 1, 2]),
        ('test_count:desc', [3, 1, 2]),
    ],
)
def test_the_rows_come_back_in_the_asked_order(
    client: TestClient, sort: str | None, expected: list[int]
) -> None:
    assert _ids(client, sort) == expected


def test_it_serves_the_orders_it_accepts_and_the_one_applied(client: TestClient) -> None:
    body = client.get(BASE + '/stats/classes', params={'sort': 'test_count:asc'}).json()
    assert body['sort_applied'] == 'test_count:asc'
    served = {o['id'] for o in body['sort_options']}
    assert {'class_name:asc', 'test_count:desc', 'trainable_gap:desc'} <= served
    for option in body['sort_options']:
        assert _ids(client, option['id'])


def test_a_row_carries_its_held_out_test_count(client: TestClient) -> None:
    rows = {c['class_id']: c for c in client.get(BASE + '/stats/classes').json()['classes']}
    assert (rows[3]['test_count'], rows[3]['test_deficient']) == (9, False)
    assert (rows[2]['test_count'], rows[2]['test_deficient']) == (1, True)


@pytest.mark.parametrize('sort', ['nope:asc', 'count:sideways'])
def test_an_unknown_sort_is_a_400(client: TestClient, sort: str) -> None:
    r = client.get(BASE + '/stats/classes', params={'sort': sort})
    assert r.status_code == 400, r.text
    assert 'sort' in r.json()['detail']
