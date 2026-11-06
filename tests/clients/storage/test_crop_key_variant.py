"""Crop cache keys carry the size variant so a source-size crop is never served as a capped one."""

from __future__ import annotations

import pytest

from src.domain.storage_keys import ParsedKey, crop_key, parse_key


UNSAFE = 'unsafe storage|unknown storage root|max_side'


def test_capped_and_uncapped_crops_get_different_keys() -> None:
    assert crop_key('cats', 'item1', 'box1', 512) != crop_key('cats', 'item1', 'box1', 0)


def test_key_shape_and_round_trip() -> None:
    key = crop_key('cats', 'item1', 'box1', 512)
    assert key == 'projects/cats/crops/item1/box1/512.jpg'
    assert crop_key('cats', 'item1', 'box1', 0) == 'projects/cats/crops/item1/box1/0.jpg'
    assert parse_key(key) == ParsedKey('projects', ('cats', 'crops', 'item1', 'box1', '512.jpg'))


@pytest.mark.parametrize('max_side', [-1, 1.5, True])
def test_max_side_must_be_a_non_negative_int(max_side: object) -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        crop_key('cats', 'item1', 'box1', max_side)  # type: ignore[arg-type]


def test_unsafe_ids_are_refused() -> None:
    with pytest.raises(ValueError, match=UNSAFE):
        crop_key('cats', '../x', 'box1', 0)
