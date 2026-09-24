"""Discard is a recorded, undoable human write; human label writes decide
their own provenance.

- ``POST /crops/{id}/discard`` (+ ``/crops/discard_batch``) clears the
  item's class and/or dismisses it from review, and ``POST
  /crops/{id}/label/undo`` restores exactly what it replaced.
- ``label_source`` on the human label writes is restricted to the human
  sources; ``class_source`` is always the server's own ``human``.
- ``batch_label`` / ``move`` return ``updated_ids`` so a client's undo
  stack holds exactly the crops that were written (never a conflicted one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_crops_undo_exclude import (  # noqa: F401 - fixtures
    ITEMS,
    _class_state,
    _label,
    _proposal_doc,
    client_for,
    crops,
    ids,
    registry,
    registry_ids,
)


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def _state(doc: dict[str, Any]) -> dict[str, Any]:
    return {**_class_state(doc), 'review_dismissed_at': doc.get('review_dismissed_at')}


@pytest.mark.asyncio
async def test_discard_then_undo_restores_exactly(crops, registry, ids, client_for) -> None:  # noqa: F811
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    await _label(crops, fake, registry, 'c1', ids['widget'])
    labeled = _state(fake.docs(ITEMS)['c1'])
    client: TestClient = client_for(fake)

    r = client.post('/curation/crops/c1/discard', json={})
    assert r.status_code == 200, r.text
    after = fake.docs(ITEMS)['c1']
    assert after['class_id'] is None
    assert after['class_validated'] is False
    assert after['cluster_id'] is None
    assert r.json()['class_id'] is None

    r = client.post('/curation/crops/c1/label/undo')
    assert r.status_code == 200, r.text
    assert _state(fake.docs(ITEMS)['c1']) == labeled


@pytest.mark.asyncio
async def test_undo_after_discard_after_label_steps_back(crops, registry, ids, client_for) -> None:  # noqa: F811
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    original = _state(fake.docs(ITEMS)['c1'])
    await _label(crops, fake, registry, 'c1', ids['widget'])
    labeled = _state(fake.docs(ITEMS)['c1'])
    client = client_for(fake)
    assert client.post('/curation/crops/c1/discard', json={}).status_code == 200

    assert client.post('/curation/crops/c1/label/undo').status_code == 200
    assert _state(fake.docs(ITEMS)['c1']) == labeled
    assert client.post('/curation/crops/c1/label/undo').status_code == 200
    assert _state(fake.docs(ITEMS)['c1']) == original
    assert client.post('/curation/crops/c1/label/undo').status_code == 409


@pytest.mark.asyncio
async def test_review_dismiss_only_keeps_class_and_is_undoable(
    crops,  # noqa: F811
    registry,  # noqa: F811
    ids,  # noqa: F811
    client_for,  # noqa: F811
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    before = _state(fake.docs(ITEMS)['c1'])
    client = client_for(fake)
    r = client.post(
        '/curation/crops/c1/discard', json={'clear_class': False, 'dismiss_from_review': True}
    )
    assert r.status_code == 200, r.text
    after = fake.docs(ITEMS)['c1']
    assert after['review_dismissed_at']
    assert _class_state(after) == _class_state(fake.docs(ITEMS)['c1'])
    assert after['cluster_id'] == before['cluster_id']

    assert client.post('/curation/crops/c1/label/undo').status_code == 200
    assert _state(fake.docs(ITEMS)['c1']) == before


def test_discard_requires_an_effect(client_for) -> None:  # noqa: F811
    client = client_for(QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}}))
    r = client.post(
        '/curation/crops/c1/discard', json={'clear_class': False, 'dismiss_from_review': False}
    )
    assert r.status_code == 422


def test_discard_unknown_crop_is_404(client_for) -> None:  # noqa: F811
    client = client_for(QueryFakeOpenSearch({ITEMS: {}}))
    assert client.post('/curation/crops/nope/discard', json={}).status_code == 404


def test_discard_batch(client_for) -> None:  # noqa: F811
    fake = QueryFakeOpenSearch({ITEMS: {k: _proposal_doc(k) for k in ('a', 'b')}})
    before = {k: _state(v) for k, v in fake.docs(ITEMS).items()}
    client = client_for(fake)
    r = client.post('/curation/crops/discard_batch', json={'crop_ids': ['a', 'b', 'gone']})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['discarded'] == 2
    assert sorted(i['crop_id'] for i in body['items']) == ['a', 'b']
    assert body['not_found'] == ['gone']
    r = client.post('/curation/crops/label/undo_batch', json={'crop_ids': ['a', 'b']})
    assert r.status_code == 200, r.text
    assert {k: _state(v) for k, v in fake.docs(ITEMS).items()} == before


# ------------------------------------------------------------------ label_source


def test_label_source_must_be_a_human_source(client_for, ids) -> None:  # noqa: F811
    client = client_for(QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}}))
    r = client.put(
        '/curation/crops/c1/label', json={'class_id': ids['widget'], 'label_source': 'vlm'}
    )
    assert r.status_code == 422
    r = client.put(
        '/curation/crops/batch_label',
        json={'crop_ids': ['c1'], 'class_id': ids['widget'], 'label_source': 'cluster_majority'},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_class_source_is_always_human_on_a_human_write(crops, registry, ids) -> None:  # noqa: F811
    from src.routers.curation._common import CropLabelRequest

    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    await crops.label_crop(
        'c1',
        CropLabelRequest(class_id=ids['widget'], label_source='human_confirmed'),
        fake,
        registry,
    )
    doc = fake.docs(ITEMS)['c1']
    assert doc['class_source'] == 'human'
    assert doc['label_source'] == 'human_confirmed'


@pytest.mark.asyncio
async def test_batch_writes_return_updated_ids(crops, ids) -> None:  # noqa: F811
    from src.routers.curation._common import CropBatchLabelRequest, CropMoveRequest

    fake = QueryFakeOpenSearch({ITEMS: {k: _proposal_doc(k) for k in ('a', 'b')}})
    out = await crops.batch_label_crops(
        CropBatchLabelRequest(crop_ids=['a', 'gone'], class_id=ids['widget']), fake
    )
    assert out['updated_ids'] == ['a']
    assert [c['crop_id'] for c in out['conflicts']] == ['gone']
    out = await crops.move_crops(
        CropMoveRequest(crop_ids=['b', 'gone'], cluster_id=ids['gadget']), fake
    )
    assert out['updated_ids'] == ['b']
