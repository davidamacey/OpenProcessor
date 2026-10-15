"""The item ``assist`` object: stored suggestions reach the wire through the one
serializer, boxes that no longer exist are dropped, and the index maps it."""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.assist_fields import ASSIST_ITEM_MAPPING
from src.clients.curation_opensearch.bodies_core import _items_body
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox, boxes_write_fields
from src.routers.curation._item_models import ItemDoc
from src.services.curation.wire import ITEM_WIRE_KEYS, serialize_item


F = get_region_fields()

STORED: dict[str, Any] = {
    'run_id': 'asr_1',
    'run_created_at': '2026-10-15T10:00:00+00:00',
    'vlm_endpoint': 'env@3',
    'vlm_model': 'm',
    'prompt_pack': 'cars',
    'prompt_pack_revision': 7,
    'class_name': 'sedan',
    'class_confidence': 'high',
    'attributes': [{'key': 'make', 'value': 'acme', 'confidence': 'medium'}],
    'boxes': [
        {'box_id': 'b1', 'attributes': [{'key': 'rim', 'value': 'alloy', 'confidence': 'high'}]},
        {'box_id': 'gone', 'attributes': [{'key': 'rim', 'value': 'steel', 'confidence': 'low'}]},
    ],
}


def _item(assist: dict[str, Any] | None) -> dict[str, Any]:
    box = RegionBox(box_id='b1', bbox_norm=(0.1, 0.1, 0.5, 0.5), state='accepted')
    src: dict[str, Any] = {
        'crop_id': 'c1',
        'image_id': 'i1',
        **boxes_write_fields([box], current_src={}),
    }
    if assist is not None:
        src['assist'] = assist
    return src


def test_an_item_without_suggestions_has_a_null_assist() -> None:
    assert serialize_item(_item(None))['assist'] is None


def test_the_stored_assist_object_reaches_the_wire() -> None:
    wire = serialize_item(_item(STORED))['assist']
    assert wire['run_id'] == 'asr_1'
    assert wire['class_name'] == 'sedan'
    assert wire['attributes'] == [{'key': 'make', 'value': 'acme', 'confidence': 'medium'}]
    assert wire['prompt_pack'] == 'cars'
    assert wire['prompt_pack_revision'] == 7


def test_boxes_that_no_longer_exist_are_dropped() -> None:
    boxes = serialize_item(_item(STORED))['assist']['boxes']
    assert [b['box_id'] for b in boxes] == ['b1']


def test_the_wire_model_documents_the_key() -> None:
    assert 'assist' in ITEM_WIRE_KEYS
    assert 'assist' in ItemDoc.model_fields
    ItemDoc.model_validate(serialize_item(_item(STORED)))


def test_the_items_index_maps_the_assist_object() -> None:
    props = _items_body()['mappings']['properties']
    assert props['assist']['properties']['attributes']['type'] == 'nested'
    assert props['assist']['properties']['run_created_at']['type'] == 'date'
    assert set(ASSIST_ITEM_MAPPING) == {'assist', 'region_run_id', 'region_run_verify'}
