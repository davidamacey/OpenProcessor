"""The VLM's raw class answer is on the item wire (``vlm_raw_class``).

A ``vlm_unmatched`` item is reviewed for exactly what the VLM said
("motorcycle", a misspelt class name, ...): the answer is stored and
mapped, but the wire used to drop it, so a client could only show the
registry class the item happened to carry.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.routers.curation._item_models import ItemDoc
from src.services.curation.wire import ITEM_WIRE_KEYS, serialize_item


REPO = Path(__file__).resolve().parents[2]


def test_unmatched_item_serves_the_vlm_answer() -> None:
    item = serialize_item(
        {'class_source': 'vlm_unmatched', 'class_name': 'class_b', 'vlm_raw_class': 'trike'},
        'x',
        api_prefix='',
    )
    assert item['vlm_raw_class'] == 'trike'
    assert item['class_name'] == 'class_b'


def test_missing_answer_is_null() -> None:
    assert serialize_item({}, 'x', api_prefix='')['vlm_raw_class'] is None


def test_answer_is_documented_in_the_model_and_the_contract() -> None:
    assert 'vlm_raw_class' in ITEM_WIRE_KEYS
    assert 'vlm_raw_class' in ItemDoc.model_fields
    contract = json.loads((REPO / 'contracts/json/item_wire.json').read_text())
    assert 'vlm_raw_class' in json.dumps(contract)
