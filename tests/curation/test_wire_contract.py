"""Item wire contract: every endpoint emits the same keys for the same doc.

One fixture document is served through ``GET /crops``, ``GET /crops/{id}``,
``GET /review/{tab}``, ``GET /regions``, ``GET /regions/training_candidates``,
semantic search and the ``crop.region_verified`` SSE event. Each must
produce exactly the shared item key set (plus its documented extras), and
an ``OP_REGION_FIELD_*`` storage override must not change a single wire key.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import region_fields as region_fields_mod
from src.config.region_fields import RegionFields
from src.routers.curation import _common
from src.routers.curation._common import ItemDoc
from src.services.curation import semantic_search, wire
from src.services.curation.wire import (
    ITEM_WIRE_KEYS,
    REGION_WIRE_ATTRS,
    REGION_WIRE_KEYS,
    REVIEW_EXTRA_KEYS,
    SEARCH_EXTRA_KEYS,
    TRAINING_CANDIDATE_EXTRA_KEYS,
)


# A deployment whose stored region fields use a different vocabulary.
_OVERRIDE_STORAGE = RegionFields(
    **{
        f.name: getattr(RegionFields(), f.name).replace('region', 'legacyroi', 1)
        for f in dataclass_fields(RegionFields)
    }
)


def _region_values() -> dict[str, Any]:
    """A distinct, JSON-safe value per region attribute."""
    values: dict[str, Any] = {}
    for attr in REGION_WIRE_ATTRS:
        if attr == 'bbox_norm':
            values[attr] = [0.1, 0.2, 0.3, 0.4]
        elif attr == 'detector_chain':
            values[attr] = ['det:miss', 'seg:hit', 'seg:vlm_verify_ok']
        else:
            values[attr] = f'v-{attr}'
    return values


def _stored_doc(storage: RegionFields) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': 'crop-1',
        'image_id': 'img-1',
        'image_path': '/data/img-1.jpg',
        'bbox_norm': [0.0, 0.0, 0.5, 0.5],
        'class_id': 3,
        'class_name': 'thing',
        'class_source': 'vlm',
        'confidence': 0.42,
        'label_source': 'vlm',
        'class_validated': False,
        'vlm_confidence': 'medium',
        'vlm_proposed_class': 'thing',
        'cluster_id': 7,
        'mistakenness_score': 0.9,
        'mistakenness_method': 'm',
        'updated_at': '2026-09-23T00:00:00+00:00',
        'pe_embedding': [0.0] * 4,
        storage.embedding: [0.0] * 4,
    }
    for attr, value in _region_values().items():
        doc[getattr(storage, attr)] = value
    return doc


class _FakeItemsOS:
    """Serves one document to every read path the item endpoints use."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc

    def _hit(self) -> dict[str, Any]:
        return {'_id': self.doc['crop_id'], '_source': dict(self.doc), '_score': 0.77}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'hits': {'total': {'value': 1}, 'hits': [self._hit()]}}

    async def get(self, *, index: str, id: str, **_kw: Any) -> dict[str, Any]:  # noqa: A002, ARG002
        if id != self.doc['crop_id']:
            raise KeyError(id)
        return {**self._hit(), 'found': True}

    async def mget(self, *, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'docs': [{**self._hit(), 'found': True}]}


def _client(monkeypatch: pytest.MonkeyPatch, storage: RegionFields) -> TestClient:
    monkeypatch.setattr(region_fields_mod, '_default_region_fields', storage)
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = _FakeItemsOS(_stored_doc(storage))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _endpoint_items(client: TestClient) -> dict[str, dict[str, Any]]:
    prefix = _common.config.api_prefix
    out: dict[str, dict[str, Any]] = {}
    r = client.get(f'{prefix}/crops')
    assert r.status_code == 200, r.text
    out['crops'] = r.json()['crops'][0]
    r = client.get(f'{prefix}/crops/crop-1')
    assert r.status_code == 200, r.text
    out['crop'] = r.json()
    r = client.get(f'{prefix}/review/all')
    assert r.status_code == 200, r.text
    out['review'] = r.json()['items'][0]
    r = client.get(f'{prefix}/regions')
    assert r.status_code == 200, r.text
    out['regions'] = r.json()['items'][0]
    r = client.get(f'{prefix}/regions/training_candidates', params={'mode': 'human_corrected'})
    assert r.status_code == 200, r.text
    out['training_candidates'] = r.json()['items'][0]
    return out


_EXTRAS = {
    'crops': frozenset(),
    'crop': frozenset(),
    'review': REVIEW_EXTRA_KEYS,
    'regions': frozenset(),
    'training_candidates': TRAINING_CANDIDATE_EXTRA_KEYS,
}


def test_item_doc_model_documents_exactly_the_serializer_keys() -> None:
    assert set(ItemDoc.model_fields) == ITEM_WIRE_KEYS


def test_region_wire_keys_are_the_stock_region_names() -> None:
    assert all(k.startswith('region_') for k in REGION_WIRE_KEYS)
    assert set(REGION_WIRE_KEYS) <= ITEM_WIRE_KEYS
    assert 'region_thumbnail_url' in ITEM_WIRE_KEYS
    assert not any('plate' in k or 'gemma' in k for k in ITEM_WIRE_KEYS)


def test_every_endpoint_emits_the_same_item_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, RegionFields()) as client:
        items = _endpoint_items(client)
    for name, item in items.items():
        assert set(item) == ITEM_WIRE_KEYS | _EXTRAS[name], name
    # Same data -> same values across endpoints on every shared key.
    reference = items['crop']
    for name, item in items.items():
        for key in ITEM_WIRE_KEYS:
            assert item[key] == reference[key], (name, key)


def test_semantic_search_item_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = RegionFields()
    hit = {'_id': 'crop-1', '_source': _stored_doc(storage), '_score': 0.77}
    item = semantic_search._hydrate_item(hit, storage, _common.config.api_prefix)
    assert set(item) == ITEM_WIRE_KEYS | SEARCH_EXTRA_KEYS
    assert item['semantic_score'] == 0.77
    with _client(monkeypatch, storage) as client:
        crop = client.get(f'{_common.config.api_prefix}/crops/crop-1').json()
    assert {k: item[k] for k in ITEM_WIRE_KEYS} == crop


def test_region_values_reach_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, RegionFields()) as client:
        crop = client.get(f'{_common.config.api_prefix}/crops/crop-1').json()
    for attr, value in _region_values().items():
        assert crop[wire.region_wire_key(attr)] == value, attr
    assert crop['vlm_confidence'] == 'medium'
    # F-6 / D-1: classifier_raw_confidence is retired -- never written in
    # production, so it must no longer appear on the wire at all.
    assert 'classifier_raw_confidence' not in crop


def test_vlm_suggestion_keys_on_every_item() -> None:
    assert {'vlm_proposed_class_id', 'vlm_proposed_class_name'} <= ITEM_WIRE_KEYS
    empty = wire.serialize_item({}, 'x', api_prefix='')
    assert empty['vlm_proposed_class_id'] is None
    assert empty['vlm_proposed_class_name'] is None


def test_vlm_suggestion_reaches_every_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, RegionFields()) as client:
        items = _endpoint_items(client)
    for name, item in items.items():
        assert item['vlm_proposed_class_id'] == 3, name
        assert item['vlm_proposed_class_name'] == 'thing', name
    # Review's proposed_class_* agree with the suggestion when there is one.
    assert items['review']['proposed_class_id'] == 3
    assert items['review']['proposed_class_name'] == 'thing'


def _review_item(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> dict[str, Any]:
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = _FakeItemsOS({**_stored_doc(RegionFields()), **overrides})
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        r = client.get(f'{_common.config.api_prefix}/review/all')
    assert r.status_code == 200, r.text
    return r.json()['items'][0]


def test_review_new_class_proposal_has_no_stale_class_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _review_item(
        monkeypatch, class_source='vlm_new_class_pending', vlm_proposed_class='gizmo'
    )
    assert (item['vlm_proposed_class_id'], item['vlm_proposed_class_name']) == (None, 'gizmo')
    assert (item['proposed_class_id'], item['proposed_class_name']) == (None, 'gizmo')


def test_review_falls_back_to_current_class_without_a_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _review_item(
        monkeypatch,
        class_source='vlm_unmatched',
        vlm_raw_class='mystery',
        vlm_proposed_class='stale',
    )
    assert (item['vlm_proposed_class_id'], item['vlm_proposed_class_name']) == (None, None)
    assert (item['proposed_class_id'], item['proposed_class_name']) == (3, 'mystery')


def test_storage_override_does_not_change_wire_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OP_REGION_FIELD_*`` picks where values are READ from; the wire
    keys and values are identical to a stock deployment's."""
    with _client(monkeypatch, RegionFields()) as client:
        stock = _endpoint_items(client)
    with _client(monkeypatch, _OVERRIDE_STORAGE) as client:
        overridden = _endpoint_items(client)
    assert stock == overridden
    for item in overridden.values():
        assert not any(k.startswith('legacyroi') for k in item)


def test_region_write_responses_use_wire_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PUT response echoes wire keys, not the overridden storage keys."""

    class _WritableOS(_FakeItemsOS):
        async def get(self, *, index: str, id: str, **kw: Any) -> dict[str, Any]:  # noqa: A002
            resp = await super().get(index=index, id=id, **kw)
            return {**resp, '_seq_no': 0, '_primary_term': 1}

        async def update(self, **kwargs: Any) -> dict[str, Any]:
            self.doc.update(kwargs['body']['doc'])
            return {'result': 'updated'}

    monkeypatch.setattr(region_fields_mod, '_default_region_fields', _OVERRIDE_STORAGE)
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake = _WritableOS(_stored_doc(_OVERRIDE_STORAGE))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        r = client.put(
            f'{_common.config.api_prefix}/crops/crop-1/region',
            json={'region_bbox_norm': [0.1, 0.1, 0.2, 0.2]},
        )
    assert r.status_code == 200, r.text
    assert set(r.json()) == {'crop_id', 'region_bbox_norm', 'region_status', 'item'}
    # The post-write item is the shared wire item under the storage override.
    assert set(r.json()['item']) == ITEM_WIRE_KEYS
    assert r.json()['item']['region_bbox_norm'] == [0.1, 0.1, 0.2, 0.2]
    assert fake.doc[_OVERRIDE_STORAGE.bbox_norm] == [0.1, 0.1, 0.2, 0.2]


@pytest.mark.parametrize(
    ('method', 'path', 'body'),
    [
        ('put', '/crops/crop-1/region', {'bbox_norm': [0.1, 0.1, 0.2, 0.2]}),
        ('patch', '/crops/crop-1/region_meta', {'plate_status': 'detected'}),
        ('post', '/regions/batch_status', {'crop_ids': ['crop-1'], 'plate_status': 'detected'}),
        ('put', '/crops/batch_region', {'crop_ids': ['crop-1'], 'bbox_norm': None}),
    ],
)
def test_region_request_bodies_reject_old_key_names(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, body: dict[str, Any]
) -> None:
    with _client(monkeypatch, RegionFields()) as client:
        r = getattr(client, method)(f'{_common.config.api_prefix}{path}', json=body)
    assert r.status_code == 422, r.text


def test_server_built_urls_follow_the_configured_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wire, '_api_prefix', lambda: '/custom-mount')
    with _client(monkeypatch, RegionFields()) as client:
        items = _endpoint_items(client)
    for name, item in items.items():
        assert item['thumbnail_url'] == '/custom-mount/crops/crop-1/thumbnail', name
        assert item['region_thumbnail_url'] == '/custom-mount/crops/crop-1/region_thumbnail', name


# ---------------------------------------------------------------------------
# crop.region_verified SSE event (audit S7)
# ---------------------------------------------------------------------------


class _RecordingHub:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _event_data_keys(event: dict[str, Any]) -> set[str]:
    return set(event) - {'type', 'topic', 'ts'}


def test_publish_endpoint_carries_region_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.routers.curation import events

    hub = _RecordingHub()
    monkeypatch.setattr(events, 'get_event_hub', lambda: hub)
    with _client(monkeypatch, _OVERRIDE_STORAGE) as client:
        r = client.post(
            f'{_common.config.api_prefix}/events/publish',
            json=wire.region_event_payload('crop-1', region_status='detected', region_text='AB'),
        )
    assert r.status_code == 200, r.text
    (event,) = hub.events
    assert event['type'] == 'crop.region_verified'
    assert event['region_status'] == 'detected'
    assert event['region_text'] == 'AB'
    assert _event_data_keys(event) <= ITEM_WIRE_KEYS


def test_publish_endpoint_rejects_unknown_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The S7 bug: an unknown status key was silently dropped (200, empty event)."""
    with _client(monkeypatch, RegionFields()) as client:
        r = client.post(
            f'{_common.config.api_prefix}/events/publish',
            json={'type': 'crop.region_verified', 'crop_id': 'c', 'plate_status': 'detected'},
        )
    assert r.status_code == 422


def test_in_process_region_event_uses_wire_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import event_hub

    hub = _RecordingHub()
    monkeypatch.setattr(event_hub, 'get_event_hub', lambda: hub)
    monkeypatch.setattr(region_fields_mod, '_default_region_fields', _OVERRIDE_STORAGE)
    event_hub.publish_region_verified('crop-1', region_status='detected')
    (event,) = hub.events
    assert event['region_status'] == 'detected'
    assert _event_data_keys(event) <= ITEM_WIRE_KEYS


@pytest.mark.asyncio
async def test_worker_region_events_reach_subscribers_with_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the SAM worker reads the status under the storage name,
    POSTs it, and the published event still carries ``region_status``."""
    from scripts.curation.worker import bulk_writer
    from scripts.curation.worker.state import _ItemTask
    from src.routers.curation import events

    hub = _RecordingHub()
    monkeypatch.setattr(events, 'get_event_hub', lambda: hub)
    monkeypatch.setattr(region_fields_mod, '_default_region_fields', _OVERRIDE_STORAGE)
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', True)
    from src.routers.curation import router as curation_router

    app = FastAPI()
    app.include_router(curation_router)
    api = TestClient(app)

    class _ForwardingClient:
        async def post(self, url: str, *, json: dict[str, Any], timeout: float) -> Any:  # noqa: ARG002
            return api.post(url.removeprefix('http://api'), json=json)

    monkeypatch.setattr(bulk_writer, '_EVENT_API_URL', 'http://api')
    monkeypatch.setattr(bulk_writer, '_EVENT_CLIENT', _ForwardingClient())
    task = _ItemTask.__new__(_ItemTask)
    task.crop_id = 'crop-1'
    task.update_doc = {_OVERRIDE_STORAGE.status: 'detected', _OVERRIDE_STORAGE.text: 'AB'}
    await bulk_writer._publish_region_events([task])

    (event,) = hub.events
    assert event['type'] == 'crop.region_verified'
    assert event['region_status'] == 'detected'
    assert event['region_text'] == 'AB'
