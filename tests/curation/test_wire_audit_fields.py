"""The accuracy-audit state is on the item wire (#166), so a queue item can show
whether it was drawn into an audit, the machine label it carried and the verdict."""

from __future__ import annotations

from src.domain.audit_math import AUDIT_FIELDS
from src.routers.curation._item_models import ItemDoc
from src.services.curation.wire import ITEM_WIRE_KEYS, serialize_item


def test_drawn_item_serves_its_audit_state() -> None:
    src = {
        'audit_sample': True,
        'audit_batch_id': 'b1',
        'audit_sampled_at': '2026-10-01T00:00:00Z',
        'audit_label_name': 'cup',
        'audit_label_source': 'vlm',
        'audit_outcome': 'vlm_wrong',
        'audit_human_name': 'mug',
    }
    item = serialize_item(src, 'x', api_prefix='')
    for key, value in src.items():
        assert item[key] == value


def test_undrawn_item_serves_nulls() -> None:
    item = serialize_item({}, 'x', api_prefix='')
    assert all(item[k] is None for k in AUDIT_FIELDS)


def test_every_stored_audit_field_is_documented() -> None:
    assert set(AUDIT_FIELDS) <= ITEM_WIRE_KEYS
    assert set(AUDIT_FIELDS) <= set(ItemDoc.model_fields)
