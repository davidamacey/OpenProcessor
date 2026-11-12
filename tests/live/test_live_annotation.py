"""Live: draw a box, move it, delete it, undo it, against a real OpenSearch.

The drawn item is created on a seeded image, so the seeded cohorts other tests rely on are never
edited. Every claim is checked in the index, not only in the response. Concurrent edits holding the
same revision: exactly one wins. The real COCO ingest checks (embed, re-detect) run in the opt-in
Cropwright write tier, because this harness has a fake Triton.
"""

from __future__ import annotations

import concurrent.futures
import uuid
from typing import Any

import pytest

from .conftest import CLASS_NAMES, INDEXES, get_doc, refresh


pytestmark = pytest.mark.live

BOX = [0.2, 0.2, 0.5, 0.5]
MOVED = [0.3, 0.3, 0.7, 0.8]


def _source(opensearch: Any, crop_id: str) -> dict[str, Any]:
    return get_doc(opensearch, INDEXES['items'], crop_id)['_source']


@pytest.fixture(scope='module')
def image_id(api_client: Any) -> str:
    resp = api_client.get('/images', params={'limit': 1})
    assert resp.status_code == 200, resp.text
    return str(resp.json()['items'][0]['image_id'])


@pytest.fixture(scope='module')
def drawn(api_client: Any, opensearch: Any, image_id: str) -> dict[str, Any]:
    resp = api_client.post(
        f'/images/{image_id}/items',
        json={'bbox_norm': BOX, 'class_name': CLASS_NAMES[0], 'request_id': str(uuid.uuid4())},
    )
    assert resp.status_code == 201, resp.text
    refresh(opensearch, INDEXES['items'])
    return resp.json()


def test_a_drawn_item_exists_with_a_human_class(opensearch: Any, drawn: dict[str, Any]) -> None:
    doc = _source(opensearch, drawn['item']['crop_id'])
    assert (doc['class_source'], doc['class_validated']) == ('human', True)
    assert (doc['geometry_source'], doc['geometry_revision'], doc['bbox_norm']) == ('human', 1, BOX)
    assert drawn['embedding_state'] in ('embedded', 'deferred')


def test_move_bumps_the_revision_and_changes_the_crop(
    api_client: Any, opensearch: Any, drawn: dict[str, Any]
) -> None:
    crop_id = drawn['item']['crop_id']
    before = api_client.get(f'/crops/{crop_id}/thumbnail').content
    resp = api_client.patch(
        f'/crops/{crop_id}/geometry', json={'bbox_norm': MOVED, 'expected_geometry_revision': 1}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['item']['thumbnail_url'].endswith('?v=2')
    doc = _source(opensearch, crop_id)
    assert (doc['geometry_revision'], doc['bbox_norm']) == (2, MOVED)
    assert api_client.get(f'/crops/{crop_id}/thumbnail').content != before


def test_two_edits_with_the_same_revision_one_wins(api_client: Any, drawn: dict[str, Any]) -> None:
    crop_id = drawn['item']['crop_id']
    current = api_client.get(f'/crops/{crop_id}').json()['geometry_revision']

    def edit(box: list[float]) -> int:
        return api_client.patch(
            f'/crops/{crop_id}/geometry',
            json={'bbox_norm': box, 'expected_geometry_revision': current},
        ).status_code

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = sorted(pool.map(edit, ([0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9])))
    assert codes == [200, 409]


def test_undo_restores_the_geometry(
    api_client: Any, opensearch: Any, drawn: dict[str, Any]
) -> None:
    crop_id = drawn['item']['crop_id']
    before = _source(opensearch, crop_id)
    resp = api_client.post(
        f'/crops/{crop_id}/geometry/undo',
        json={'expected_geometry_revision': before['geometry_revision']},
    )
    assert resp.status_code == 200, resp.text
    after = _source(opensearch, crop_id)
    assert after['bbox_norm'] != before['bbox_norm']
    assert after['geometry_revision'] == before['geometry_revision'] + 1


def test_delete_as_not_an_object_and_restore(
    api_client: Any, opensearch: Any, drawn: dict[str, Any]
) -> None:
    crop_id = drawn['item']['crop_id']
    gone = api_client.post(
        '/crops/batch_exclude', json={'crop_ids': [crop_id], 'reason': 'not_an_object'}
    )
    assert gone.status_code == 200, gone.text
    assert _source(opensearch, crop_id)['class_excluded'] is True
    back = api_client.post('/crops/batch_unexclude', json={'crop_ids': [crop_id]})
    assert back.status_code == 200, back.text
    assert _source(opensearch, crop_id)['class_excluded'] is False
