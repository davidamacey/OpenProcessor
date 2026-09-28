"""W8.6/W8.15: the numbered-overlay VLM contract.

``draw_region_overlay`` / ``overlay_description`` / ``render_region_block``
/ ``box_verdicts``. D-B: the pre-W8 flat reply shape is dropped -- a
reply lacking the list key raises ``MultiRegionKeysMissingError``.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.config.region_fields import RegionFields
from src.services.labeling.region_overlay import (
    MultiRegionKeysMissingError,
    box_verdicts,
    draw_region_overlay,
    overlay_description,
    render_region_block,
)


def _jpeg(w: int = 200, h: int = 200) -> bytes:
    img = Image.new('RGB', (w, h), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def test_numbered_overlay_draws_distinct_pixels_per_box() -> None:
    base = _jpeg()
    boxes = [(0.05, 0.05, 0.3, 0.3), (0.6, 0.6, 0.9, 0.9)]
    out = draw_region_overlay(base, boxes)
    assert out is not None
    assert out != base
    img = Image.open(io.BytesIO(out)).convert('RGB')
    w, h = img.size
    # A pixel inside each box's outline region should no longer match the
    # plain background fill -- the outline (or its number tag) painted it.
    px1 = img.getpixel((int(0.05 * w), int(0.05 * h)))
    px2 = img.getpixel((int(0.6 * w), int(0.6 * h)))
    assert px1 != (10, 20, 30) or px2 != (10, 20, 30)


def test_n1_overlay_draws_a_single_tagged_box() -> None:
    base = _jpeg()
    out = draw_region_overlay(base, [(0.1, 0.1, 0.4, 0.4)])
    assert out is not None
    assert out != base


def test_overlay_none_on_empty_boxes() -> None:
    assert draw_region_overlay(_jpeg(), []) is None


def test_overlay_edge_box_tag_is_clamped_inside_image() -> None:
    # A box touching the top-left corner would otherwise put its tag
    # partially off-canvas.
    base = _jpeg()
    out = draw_region_overlay(base, [(0.0, 0.0, 0.2, 0.2)])
    assert out is not None
    # No exception/crash is the behavioural assertion here; re-decodable:
    Image.open(io.BytesIO(out)).load()


def test_overlay_description_reads_naturally_for_any_n() -> None:
    assert overlay_description(1) == (
        'the candidate regions are marked by numbered red rectangles (1 to 1); '
        'the rectangles and numbers are not part of the photo'
    )
    assert '1 to 3' in overlay_description(3)


def test_render_region_block_no_boxes() -> None:
    assert render_region_block([], overlay_drawn=False) == 'No region-bbox candidate was provided. '


def test_render_region_block_overlay_drawn() -> None:
    block = render_region_block([(0.1, 0.1, 0.2, 0.2)], overlay_drawn=True)
    assert 'numbered red rectangles (1 to 1)' in block
    assert 'Answer for each numbered box' in block


def test_render_region_block_multi_box_overlay_drawn() -> None:
    block = render_region_block([(0.0, 0.0, 0.1, 0.1)] * 3, overlay_drawn=True)
    assert '1 to 3' in block


def test_render_region_block_overlay_failed_falls_back_to_coords() -> None:
    boxes = [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)]
    block = render_region_block(boxes, overlay_drawn=False)
    assert '1=[0.100, 0.200, 0.300, 0.400]' in block
    assert '2=[0.500, 0.600, 0.700, 0.800]' in block


def test_box_verdicts_list_shape() -> None:
    F = RegionFields()
    entry = {
        F.boxes: [
            {'box': 1, F.bbox_correct: True, F.confidence: 'high', F.text: 'ABC'},
            {'box': 2, F.bbox_correct: False, F.confidence: 'low'},
        ]
    }
    verdicts = box_verdicts(entry, 3, F)
    assert len(verdicts) == 3
    assert verdicts[0].box == 1
    assert verdicts[0].bbox_correct is True
    assert verdicts[0].confidence == 'high'
    assert verdicts[0].text_reply == 'ABC'
    assert verdicts[1].bbox_correct is False
    # Box 3 got no element at all -> no verdict.
    assert verdicts[2].bbox_correct is None


def test_box_verdicts_out_of_range_and_duplicate_ignored() -> None:
    F = RegionFields()
    entry = {
        F.boxes: [
            {'box': 1, F.bbox_correct: True},
            {'box': 1, F.bbox_correct: False},  # duplicate -- first wins
            {'box': 99, F.bbox_correct: True},  # out of range -- ignored
        ]
    }
    verdicts = box_verdicts(entry, 1, F)
    assert len(verdicts) == 1
    assert verdicts[0].bbox_correct is True


def test_box_verdicts_string_box_number_accepted() -> None:
    F = RegionFields()
    entry = {F.boxes: [{'box': 'box 2', F.bbox_correct: True}]}
    verdicts = box_verdicts(entry, 2, F)
    assert verdicts[1].bbox_correct is True


def test_box_verdicts_missing_list_key_raises_pack_multi_region_keys_missing() -> None:
    F = RegionFields()
    with pytest.raises(MultiRegionKeysMissingError) as exc_info:
        box_verdicts({F.bbox_correct: True}, 1, F)
    assert exc_info.value.code == 'pack_multi_region_keys_missing'


def test_box_verdicts_non_list_value_raises() -> None:
    F = RegionFields()
    with pytest.raises(MultiRegionKeysMissingError):
        box_verdicts({F.boxes: 'not a list'}, 1, F)
