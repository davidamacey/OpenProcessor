"""The ``imported`` review tab: validated labels a dataset import wrote.

Every other tab excludes validated items; this one is exactly them (spot
checks). Run through the real route over the query-evaluating item fake.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation import review_queries


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default/review'


def _item(crop_id: str, **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 3,
        'class_name': 'car',
        'class_source': 'external_label',
        'class_validated': True,
        'import_ids': ['imp1'],
        'dataset_split': 'train',
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    doc.update(over)
    return doc


SEEDED = {
    'a': _item('a'),
    'b': _item('b', import_ids=['imp2'], dataset_split='val', class_id=4, class_name='truck'),
    # Not an imported, validated label:
    'human': _item('human', class_source='human'),
    'suggestion': _item('suggestion', class_validated=False),
    'dismissed': _item('dismissed', review_dismissed_at='2026-10-01T00:00:00+00:00'),
    'excluded': _item('excluded', class_excluded=True),
    # An unvalidated item the `all` tab does list (to tell an ignored filter from a honoured one):
    'queued': _item(
        'queued', class_source='vlm_unmatched', class_validated=False, import_ids=['other']
    ),
    # A frozen test-split label stays out unless asked for:
    'holdout': _item('holdout', test_holdout=True, dataset_split='test'),
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', _noop)
    fake = QueryFakeOpenSearch({ITEMS: dict(SEEDED)})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        c.fake = fake  # type: ignore[attr-defined]
        yield c


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(f'{BASE}/imported', params=params)
    assert r.status_code == 200, r.text
    return {i['crop_id'] for i in r.json()['items']}


def test_the_tab_lists_only_validated_imported_labels(client: TestClient) -> None:
    assert _ids(client) == {'a', 'b'}  # not human, suggestion, dismissed, excluded, queued


def test_include_test_adds_the_frozen_split(client: TestClient) -> None:
    assert _ids(client, include_test=True) == {'a', 'b', 'holdout'}


def test_filters_by_import_split_and_class(client: TestClient) -> None:
    assert _ids(client, import_id='imp1') == {'a'}
    assert _ids(client, import_id='imp2') == {'b'}
    assert _ids(client, dataset_split='val') == {'b'}
    assert _ids(client, class_name='car') == {'a'}
    assert _ids(client, import_id='imp1', dataset_split='val') == set()


def test_the_import_filters_belong_to_this_tab_only(client: TestClient) -> None:
    """Another tab ignores ``import_id`` (accepted, not honoured), as every
    tab-specific filter is."""
    assert client.get(f'{BASE}/all').json()['total'] == 1
    assert client.get(f'{BASE}/all', params={'import_id': 'imp1'}).json()['total'] == 1


def test_the_catalog_serves_the_tab_with_its_filters_and_split_options(client: TestClient) -> None:
    tabs = {t['id']: t for t in client.get(f'{BASE}/tabs').json()['tabs']}
    tab = tabs['imported']
    assert {'import_id', 'dataset_split', 'class_name'} <= set(tab['filters'])
    assert 'class_id' not in tab['filters']
    assert tab['label'] == 'Imported labels'
    split_spec = next(s for s in tab['filter_specs'] if s['param'] == 'dataset_split')
    assert [o['value'] for o in split_spec['options']] == ['', 'train', 'val', 'test']
    assert 'import_id' not in tabs['all']['filters']
    assert review_queries.tab_filters('imported') == tuple(tab['filters'])


def test_empty_reason_tells_the_operator_what_to_do(client: TestClient) -> None:
    mismatch = client.get(f'{BASE}/imported', params={'import_id': 'nope'}).json()
    assert mismatch['total'] == 0
    assert mismatch['empty_reason'] == 'no imported labels match these filters'
    client.fake.store[ITEMS].clear()
    bare = client.get(f'{BASE}/imported').json()
    assert bare['empty_reason'] == 'no imported labels: import a dataset first'
    assert client.get(f'{BASE}/tabs').json()['empty_state']['has_imported_labels'] is False


def test_the_empty_state_flag_turns_on_with_an_imported_label(client: TestClient) -> None:
    assert client.get(f'{BASE}/tabs').json()['empty_state']['has_imported_labels'] is True
