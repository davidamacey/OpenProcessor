"""The capture index fields and the documents built from CaptureMetadata (CM-2, CM-4)."""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.curation_opensearch.capture_fields import (
    CAPTURE_ITEM_MAPPING,
    capture_image_block,
    capture_item_fields,
)
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.domain.capture_metadata import CaptureLocation, CaptureMetadata, unknown_capture
from src.domain.location_policy import apply_location_policy


META = CaptureMetadata(
    exif_status='ok',
    taken_local='2024-03-05T14:22:11',
    taken_utc='2024-03-05T13:22:11',
    offset_minutes=60,
    time_source='exif_original',
    offset_source='exif_offset',
    time_precision='second',
    camera_make='Acme',
    location=CaptureLocation(47.123456, 8.987654, altitude_m=410.0),
    has_location=True,
)


def _properties(role_suffix: str) -> dict[str, Any]:
    for role, body in INDEX_BODIES.items():
        if role.value == role_suffix:
            return dict(body['mappings']['properties'])
    raise AssertionError(role_suffix)


def test_the_images_body_maps_capture_with_a_geo_point_and_the_items_body_has_no_location() -> None:
    capture = _properties('images')['capture']['properties']
    assert capture['location'] == {'type': 'geo_point'}
    assert capture['taken_utc']['type'] == 'date'
    items = _properties('items')
    assert items['capture_taken_local']['type'] == 'date'
    assert set(CAPTURE_ITEM_MAPPING) == {'capture_taken_local', 'capture_taken_utc'}
    assert not [name for name in items if 'location' in name or 'gps' in name.lower()]


def test_the_image_block_carries_the_location_only_when_the_policy_left_one() -> None:
    stored = capture_image_block(apply_location_policy(META, 'store'))
    assert stored['location'] == {'lat': 47.123456, 'lon': 8.987654}
    assert stored['location_precision'] == 'exact'
    coarse = capture_image_block(apply_location_policy(META, 'store_coarse'))
    assert coarse['location'] == {'lat': 47.12, 'lon': 8.99}
    assert 'altitude_m' not in coarse
    dropped = capture_image_block(apply_location_policy(META, 'drop'))
    assert 'location' not in dropped
    assert 'altitude_m' not in dropped
    assert dropped['has_location'] is True


def test_the_item_copies_are_the_two_capture_times_and_never_a_coordinate() -> None:
    assert capture_item_fields(META) == {
        'capture_taken_local': '2024-03-05T14:22:11',
        'capture_taken_utc': '2024-03-05T13:22:11',
    }
    assert capture_item_fields(CaptureMetadata(exif_status='absent')) == {}


@pytest.mark.parametrize('policy', ['drop', 'store_coarse', 'store'])
def test_no_item_field_ever_holds_a_coordinate(policy: str) -> None:
    fields = capture_item_fields(apply_location_policy(META, policy))
    assert '47.1' not in str(fields)
    assert '8.9' not in str(fields)


def test_an_outage_capture_has_no_parser_stamp_so_the_backfill_reads_it_again() -> None:
    block = capture_image_block(unknown_capture())
    assert block['exif_status'] == 'unknown'
    assert 'parser' not in block
