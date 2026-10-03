"""Open-vocabulary provenance on the item wire and the ``GET /crops`` filters."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation import base_curation_config
from src.services.curation.wire import item_list_source_excludes, serialize_item


ITEMS = base_curation_config().items_index
URL = '/curation/projects/default/crops'
POLY = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5]]


def _doc(crop_id: str, **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': f'img-{crop_id}',
        'bbox_norm': [0.1, 0.2, 0.3, 0.5],
        'class_source': 'open_vocab_proposal',
        'class_name': 'cone',
        'class_detector': 'sam3',
        'source_prompt': 'traffic cone',
        'open_vocab_set': 'street',
        'open_vocab_revision': 3,
        'mask_polygon': POLY,
        'updated_at': '2026-10-01T00:00:00+00:00',
    }
    doc.update(over)
    return doc


def test_the_item_wire_carries_the_provenance_and_the_outline() -> None:
    wire = serialize_item(_doc('a'))
    assert wire['source_prompt'] == 'traffic cone'
    assert wire['open_vocab_set'] == 'street'
    assert wire['open_vocab_revision'] == 3
    assert wire['mask_polygon'] == POLY
    assert wire['class_detector'] == 'sam3'


def test_an_ordinary_item_has_the_keys_with_null_values() -> None:
    wire = serialize_item({'crop_id': 'plain', 'image_id': 'i'})
    assert [wire[k] for k in ('source_prompt', 'open_vocab_set', 'open_vocab_revision')] == [
        None,
        None,
        None,
    ]
    assert wire['mask_polygon'] is None


def test_list_endpoints_do_not_fetch_the_outline() -> None:
    assert 'mask_polygon' in item_list_source_excludes()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr('src.routers.curation._ensure_indexes', _noop)
    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'cone': _doc('cone'),
                'cup': _doc('cup', source_prompt='cup', class_name='cup'),
                'other_set': _doc('other_set', open_vocab_set='garage'),
                'plain': {
                    'crop_id': 'plain',
                    'image_id': 'i',
                    'bbox_norm': [0, 0, 1, 1],
                    'class_source': 'human',
                    'updated_at': '2026-10-01T00:00:00+00:00',
                },
            }
        }
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def _ids(client: TestClient, **params: Any) -> set[str]:
    r = client.get(URL, params=params)
    assert r.status_code == 200, r.text
    return {c['crop_id'] for c in r.json()['crops']}


def test_filters_by_set_and_by_prompt(client: TestClient) -> None:
    assert _ids(client, open_vocab_set='street') == {'cone', 'cup'}
    assert _ids(client, open_vocab_set='garage') == {'other_set'}
    assert _ids(client, source_prompt='traffic cone') == {'cone', 'other_set'}
    assert _ids(client, open_vocab_set='street', source_prompt='cup') == {'cup'}
    assert _ids(client, class_source='open_vocab_proposal') == {'cone', 'cup', 'other_set'}
