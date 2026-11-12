"""The ``imported`` review_tab filter value: validated labels a dataset import wrote.

It is not a review queue (every queue excludes validated items), so it filters ``GET /crops`` as
``review_tab=imported`` for spot checks. Run through the real route over the query-evaluating
item fake.
"""

from __future__ import annotations

from typing import Any

import pytest
from _msearch_fake import install_msearch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import base_curation_config


class _Client(TestClient):
    """The test client with the fakes its fixture hangs on it."""

    fake: Any


ITEMS = base_curation_config().items_index
BASE = '/curation/projects/default'


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
    # An unvalidated item a queue lists (to tell an ignored filter from a honoured one):
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

    monkeypatch.setattr('src.services.curation.index_bootstrap.ensure_indexes', _noop)
    fake = install_msearch(QueryFakeOpenSearch({ITEMS: dict(SEEDED)}))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with _Client(app) as c:
        c.fake = fake
        yield c


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


def _ids(client: _Client, **params: Any) -> set[str]:
    r = client.get(f'{BASE}/crops', params={'review_tab': 'imported', **params})
    assert r.status_code == 200, r.text
    return {c['crop_id'] for c in r.json()['crops']}


def test_the_filter_lists_only_validated_imported_labels(client: _Client) -> None:
    assert _ids(client) == {'a', 'b'}  # not human, suggestion, dismissed, excluded, queued


def test_the_frozen_split_stays_out(client: _Client) -> None:
    assert 'holdout' not in _ids(client)


def test_narrows_by_import_split_and_class(client: _Client) -> None:
    assert _ids(client, import_id='imp1') == {'a'}
    assert _ids(client, import_id='imp2') == {'b'}
    assert _ids(client, dataset_split='val') == {'b'}
    assert _ids(client, class_name='car') == {'a'}
    assert _ids(client, import_id='imp1', dataset_split='val') == set()


def test_it_is_not_served_as_a_queue(client: _Client) -> None:
    ids = {t['id'] for t in client.get(f'{BASE}/review/tabs').json()['tabs']}
    assert 'imported' not in ids
    assert client.get(f'{BASE}/review/imported').status_code == 404
