"""``GET /training_cohorts/summary``: one matrix of classes x cohorts, counted without a per-class
fan-out."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.curation.cohort_summary import UNDECIDED_ID, agg_body, build_summary
from src.services.curation.training_cohorts import cohort_catalog


@dataclass
class _Class:
    class_id: int
    class_name: str
    deprecated: bool = False
    merged_into: int | None = None


def _answer(total: int, buckets: dict[Any, int]) -> dict[str, Any]:
    return {
        'hits': {'total': {'value': total}},
        'aggregations': {
            'groups': {'buckets': [{'key': k, 'doc_count': n} for k, n in buckets.items()]}
        },
    }


CLASSES: Any = [
    _Class(1, 'car'),
    _Class(2, 'dog'),
    _Class(3, 'bus', deprecated=True),
    _Class(4, 'old', merged_into=1),
]


def _responses(**by_cohort: dict[Any, int]) -> dict[str, dict[str, Any]]:
    return {
        c['id']: _answer(sum(by_cohort.get(c['id'], {}).values()), by_cohort.get(c['id'], {}))
        for c in cohort_catalog()
    }


def test_each_cohort_count_lands_in_its_class_row() -> None:
    summary = build_summary(
        cohort_catalog(),
        _responses(validated={1: 5, 2: 2}, needs_labeling={1: 7}),
        _answer(0, {}),
        CLASSES,
    )
    rows = {r['class_name']: r for r in summary['rows']}
    assert set(rows) == {'car', 'dog'}  # a deprecated or merged class has no row
    assert rows['car']['counts']['validated'] == 5
    assert rows['car']['counts']['needs_labeling'] == 7
    assert rows['dog']['counts']['validated'] == 2
    assert rows['dog']['counts']['needs_labeling'] == 0
    totals = {c['id']: c['total'] for c in summary['cohorts']}
    assert totals['validated'] == 7


def test_every_row_carries_a_count_for_every_cohort() -> None:
    summary = build_summary(cohort_catalog(), _responses(), _answer(0, {}), CLASSES)
    ids = [c['id'] for c in summary['cohorts']]
    assert all(list(r['counts']) == ids for r in summary['rows'])


def test_undecided_proposals_count_under_the_class_of_that_name_or_a_row_of_their_own() -> None:
    summary = build_summary(
        cohort_catalog(),
        _responses(),
        _answer(10, {'car': 4, 'traffic light': 5, 'dog': 1}),
        CLASSES,
    )
    assert summary['cohorts'][-1]['id'] == UNDECIDED_ID
    assert summary['cohorts'][-1]['total'] == 10
    rows = {r['class_name']: r for r in summary['rows']}
    assert rows['car']['counts'][UNDECIDED_ID] == 4
    assert rows['dog']['counts'][UNDECIDED_ID] == 1
    # The registry has no such class: the detector's label is a row, after the real classes.
    light = rows['traffic_light']
    assert (light['class_id'], light['kind'], light['counts'][UNDECIDED_ID]) == (
        None,
        'detector',
        5,
    )
    assert [r['kind'] for r in summary['rows']] == ['class', 'class', 'detector']


def test_the_undecided_cohort_is_left_out_when_every_item_has_a_class() -> None:
    summary = build_summary(cohort_catalog(), _responses(), _answer(0, {}), CLASSES)
    assert UNDECIDED_ID not in [c['id'] for c in summary['cohorts']]
    assert all(UNDECIDED_ID not in r['counts'] for r in summary['rows'])


def test_the_search_groups_by_the_field_and_counts_nothing_else() -> None:
    body = agg_body({'match_all': {}}, 'class_id')
    assert body['size'] == 0
    assert body['track_total_hits'] is True
    assert body['aggs']['groups']['terms']['field'] == 'class_id'


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.routers.curation import (
        _common,
        _raw_opensearch_dep,
        cohorts,
        router as curation_router,
    )

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: object()

    class _Registry:
        async def snapshot(self) -> Any:
            return SimpleNamespace(classes=CLASSES)

    app.dependency_overrides[_common._registry_dep] = lambda: _Registry()
    searched: list[list[dict[str, Any]]] = []

    async def msearch(_opensearch: Any, _index: str, bodies: list[dict[str, Any]]) -> Any:
        searched.append(bodies)
        return [_answer(0, {}) for _ in bodies]

    monkeypatch.setattr(cohorts, 'msearch_checked', msearch)
    with TestClient(app) as c:
        c.searched = searched  # type: ignore[attr-defined]
        yield c


def test_the_route_runs_one_search_per_cohort_in_one_request(client: TestClient) -> None:
    r = client.get('/curation/projects/default/training_cohorts/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {'cohorts', 'rows'}
    (bodies,) = client.searched  # type: ignore[attr-defined]
    # The catalog's cohorts plus the undecided one, whatever the number of classes.
    assert len(bodies) == len(cohort_catalog()) + 1


def test_the_searches_are_the_cohorts_own_queries_grouped_by_class(client: TestClient) -> None:
    client.get('/curation/projects/default/training_cohorts/summary')
    (bodies,) = client.searched  # type: ignore[attr-defined]
    by_field = [b['aggs']['groups']['terms']['field'] for b in bodies]
    assert by_field == ['class_id'] * (len(bodies) - 1) + ['detector_class_name']
    queries = {c['id']: b['query'] for c, b in zip(cohort_catalog(), bodies, strict=False)}
    # The validated cohort is the listing's validated filter and the needs-labeling one its
    # negation, neither narrowed to a class; the undecided one is the to_label queue.
    assert "'class_validated': True" in str(queries['validated'])
    assert "'class_validated': False" in str(queries['needs_labeling'])
    assert all("'class_id': 4" not in str(q) for q in queries.values())
    assert "'exists': {'field': 'class_id'}" in str(bodies[-1]['query'])


@pytest.mark.asyncio
async def test_a_region_cohort_counts_the_items_with_a_matching_box() -> None:
    from src.routers.curation.cohorts import cohort_search_query

    cohort = {
        'id': 'low_conf_correct',
        'params': {'mode': 'low_conf_correct'},
        'row_kind': 'region',
    }
    query = await cohort_search_query(object(), cohort)
    nested = [c for c in query['bool']['filter'] if 'nested' in c]
    assert len(nested) == 1
    assert 'inner_hits' not in nested[0]['nested']  # counted, never paged
    assert {'term': {'test_holdout': True}} in query['bool']['must_not']
