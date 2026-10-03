"""Bulk item writes on a selection: exclude / unexclude / label by filter, with a
dry run that counts exactly what the write then changes."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config
from src.routers.curation import _common


P = f'{_common.config.api_prefix}/projects/default'


def _items() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i in range(6):
        out[f'p{i}'] = {
            'crop_id': f'p{i}',
            'proposal_name': 'person',
            'confidence': 0.3 + i / 10,
            'class_source': 'item_proposal',
            'cluster_id': 10_001,
        }
    out['c0'] = {'crop_id': 'c0', 'proposal_name': 'car', 'confidence': 0.9, 'cluster_id': 10_002}
    out['held'] = {
        'crop_id': 'held',
        'proposal_name': 'person',
        'confidence': 0.9,
        'test_holdout': True,
    }
    return out


class _Registry:
    class _Entry:
        class_name = 'widget'

    def validate_id(self, class_id: int) -> bool:
        return class_id == 1

    def get(self, class_id: int) -> Any:
        return self._Entry() if class_id == 1 else None


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({get_curation_config().items_index: _items()})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake: QueryFakeOpenSearch) -> Any:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', {'default'})
    monkeypatch.setattr('src.routers.curation.crops.get_class_registry', lambda: _Registry())
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _excluded(fake: QueryFakeOpenSearch) -> set[str]:
    return {
        k
        for k, d in fake.docs(get_curation_config().items_index).items()
        if d.get('class_excluded')
    }


PEOPLE = {'filter': {'class_names': ['person'], 'conf_max': 0.55}}


def test_dry_run_counts_the_selection_and_writes_nothing(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    r = client.post(f'{P}/crops/batch_exclude', json={'selection': PEOPLE, 'dry_run': True})
    assert r.json() == {'dry_run': True, 'selected': 3}  # p0..p2; the holdout item is not browsed
    assert _excluded(fake) == set()
    assert fake.write_calls == 0


def test_exclude_by_filter_matches_the_dry_run_and_is_reversible(
    client: TestClient, fake: QueryFakeOpenSearch
) -> None:
    dry = client.post(f'{P}/crops/batch_exclude', json={'selection': PEOPLE, 'dry_run': True})
    r = client.post(f'{P}/crops/batch_exclude', json={'selection': PEOPLE, 'reason': 'blurry'})
    assert r.status_code == 200, r.text
    assert r.json() == {'excluded': dry.json()['selected'], 'errors': 0}
    assert _excluded(fake) == {'p0', 'p1', 'p2'}

    undo = {'selection': {'filter': {'review_status': ['excluded']}}}
    assert client.post(f'{P}/crops/batch_unexclude', json={**undo, 'dry_run': True}).json() == {
        'dry_run': True,
        'selected': 3,
    }
    r = client.post(f'{P}/crops/batch_unexclude', json=undo)
    assert r.json() == {'unexcluded': 3, 'errors': 0}
    assert _excluded(fake) == set()


def test_a_cap_limits_how_many_are_changed(client: TestClient, fake: QueryFakeOpenSearch) -> None:
    body = {
        'selection': {
            'filter': {'class_names': ['person']},
            'limit': 2,
            'sample': 'largest',
        }
    }
    assert client.post(f'{P}/crops/batch_exclude', json=body).json()['excluded'] == 2


def test_label_by_filter(client: TestClient, fake: QueryFakeOpenSearch) -> None:
    body = {'selection': {'filter': {'class_names': ['car']}}, 'class_id': 1}
    r = client.put(f'{P}/crops/batch_label', json=body)
    assert r.status_code == 200, r.text
    assert r.json()['updated_ids'] == ['c0']
    doc = fake.docs(get_curation_config().items_index)['c0']
    assert doc['class_id'] == 1
    assert doc['class_source'] == 'human'


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'crop_ids': ['p0'], 'selection': PEOPLE},
        {'selection': {'filter': {}}},
        {'selection': {'crop_ids': ['p0'], 'limit': 2}},
    ],
)
def test_exactly_one_target_form_and_no_unbounded_selection(
    client: TestClient, body: dict[str, Any]
) -> None:
    assert client.post(f'{P}/crops/batch_exclude', json=body).status_code == 422


def test_a_selection_over_the_bulk_limit_is_refused_not_truncated(
    client: TestClient, fake: QueryFakeOpenSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('src.routers.curation._selection.MAX_BULK_SELECTION', 2)
    r = client.post(f'{P}/crops/batch_exclude', json={'selection': PEOPLE})
    assert r.status_code == 422
    assert 'limit' in r.text
    assert _excluded(fake) == set()


def test_explicit_ids_still_work_unchanged(client: TestClient, fake: QueryFakeOpenSearch) -> None:
    r = client.post(f'{P}/crops/batch_exclude', json={'crop_ids': ['p0', 'c0']})
    assert r.json() == {'excluded': 2, 'errors': 0}
    assert _excluded(fake) == {'p0', 'c0'}
