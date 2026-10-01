"""``GET /crops`` import filters: ``import_id``, ``dataset_split``,
``label_source``, ``on_negative_frame``, ``proposed_by_import``."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default/crops'


def _item(crop_id: str, **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_source': 'external_label',
        'label_source': 'import',
        'import_ids': ['imp1'],
        'dataset_split': 'train',
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    doc.update(over)
    return doc


SEEDED = {
    'lab_train': _item('lab_train'),
    'lab_val': _item('lab_val', dataset_split='val', import_ids=['imp2']),
    'prop': _item(
        'prop',
        class_source='item_proposal',
        label_source='item_proposal',
        import_ids=[],
        dataset_split=None,
        proposed_by_import='imp1',
    ),
    'prop_neg': _item(
        'prop_neg',
        class_source='item_proposal',
        label_source='item_proposal',
        import_ids=[],
        dataset_split=None,
        proposed_by_import='imp2',
        on_negative_frame=True,
    ),
    'plain': _item(
        'plain', class_source='human', label_source='human', import_ids=[], dataset_split=None
    ),
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.routers.curation._ensure_indexes', _noop)
    fake = QueryFakeOpenSearch({ITEMS: dict(SEEDED)})
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(URL, params=params)
    assert r.status_code == 200, r.text
    return {c['crop_id'] for c in r.json()['crops']}


def test_import_id_matches_labels_and_proposals_of_that_import(client: TestClient) -> None:
    assert _ids(client, import_id='imp1') == {'lab_train', 'prop'}
    assert _ids(client, import_id='imp2') == {'lab_val', 'prop_neg'}
    assert _ids(client, import_id='nope') == set()


def test_dataset_split(client: TestClient) -> None:
    assert _ids(client, dataset_split='val') == {'lab_val'}
    assert _ids(client, dataset_split='train') == {'lab_train'}


def test_label_source(client: TestClient) -> None:
    assert _ids(client, label_source='import') == {'lab_train', 'lab_val'}


def test_on_negative_frame_both_directions(client: TestClient) -> None:
    assert _ids(client, on_negative_frame=True) == {'prop_neg'}
    assert 'prop_neg' not in _ids(client, on_negative_frame=False)
    assert len(_ids(client, on_negative_frame=False)) == 4


def test_proposed_by_import_both_directions(client: TestClient) -> None:
    assert _ids(client, proposed_by_import=True) == {'prop', 'prop_neg'}
    assert _ids(client, proposed_by_import=False) == {'lab_train', 'lab_val', 'plain'}


def test_filters_combine(client: TestClient) -> None:
    assert _ids(client, import_id='imp1', proposed_by_import=True) == {'prop'}
    assert _ids(client, import_id='imp2', on_negative_frame=True, proposed_by_import=True) == {
        'prop_neg'
    }


def test_the_wire_item_carries_the_import_fields(client: TestClient) -> None:
    crops = client.get(URL, params={'import_id': 'imp2', 'on_negative_frame': True}).json()['crops']
    (item,) = crops
    assert item['proposed_by_import'] == 'imp2'
    assert item['on_negative_frame'] is True
