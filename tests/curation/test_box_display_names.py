"""Every box an image view draws carries one served name (item display name or region text)."""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.region_boxes import RegionBox, region_box_display_name
from src.services.curation.image_boxes import item_box_views
from src.services.curation.wire import region_boxes_to_wire, serialize_item


BOX = [0.1, 0.1, 0.5, 0.5]


def _view(src: dict[str, Any]) -> Any:
    return item_box_views({'bbox_norm': BOX, **src}, 'c1')[0]


def test_item_box_label_is_the_item_display_name() -> None:
    src = {'bbox_norm': BOX, 'class_id': 3, 'class_name': 'car', 'crop_id': 'c1'}
    wire = serialize_item(src, state_fields={}, api_prefix='')
    view = _view(src)
    assert view.label == wire['display_name'] == 'car'
    assert view.label_kind == wire['display_name_kind'] == 'class'


def test_proposed_and_detector_names_are_served_with_their_kind() -> None:
    proposed = _view({'vlm_proposed_class_name': 'bus', 'vlm_raw_class': 'bus'})
    detector = _view({'detector_class_name': 'person'})
    nameless = _view({})
    assert (proposed.label, proposed.label_kind) == ('bus', 'proposed')
    assert (detector.label, detector.label_kind) == ('person', 'detector')
    assert (nameless.label, nameless.label_kind) == ('unlabeled', 'none')


def test_region_box_display_name_is_its_text_else_slot_label_and_order() -> None:
    read = RegionBox(box_id='b1', bbox_norm=(0, 0, 1, 1), state='accepted', text='ABC123')
    unread = RegionBox(box_id='b2', bbox_norm=(0, 0, 1, 1), state='proposed')
    assert region_box_display_name(read, slot_label='Plate', ordinal=1) == 'ABC123'
    assert region_box_display_name(unread, slot_label='Plate', ordinal=2) == 'Plate 2'


@pytest.mark.parametrize('kind', ['region'])
def test_region_boxes_are_served_with_display_name_on_the_item_wire_and_the_overlay(
    kind: str,
) -> None:
    from src.config import get_region_fields

    f = get_region_fields()
    src = {
        'bbox_norm': BOX,
        f.boxes: [
            {'box_id': 'b1', 'bbox_norm': [0.2, 0.2, 0.3, 0.3], 'state': 'accepted', 'text': 'XY'},
            {'box_id': 'b2', 'bbox_norm': [0.3, 0.3, 0.4, 0.4], 'state': 'proposed'},
        ],
    }
    wire = region_boxes_to_wire(src, crop_id='c1', prefix='')['region_boxes']
    views = [v for v in item_box_views(src, 'c1') if v.kind == kind]
    assert [b['display_name'] for b in wire] == ['XY', 'Region 2']
    assert [(v.label, v.label_kind) for v in views] == [('XY', 'region'), ('Region 2', 'region')]
