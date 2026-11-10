"""The letterbox pad is the integer offset the frame is pasted at, so un-mapping is exact."""

from __future__ import annotations

import pytest
from PIL import Image

from src.domain.image_geometry import (
    crop_rect,
    letterbox_geometry,
    undo_letterbox,
    whole_square_geometry,
)
from src.services.detection.geometry import letterbox_to_square


def _paste_offset(size: tuple[int, int], target: int) -> tuple[int, int]:
    """Where the frame actually lands in the canvas: the first non-fill row and column of an
    all-white frame (white never equals the grey fill)."""
    chw, _scale, _pad = letterbox_to_square(Image.new('RGB', size, (255, 255, 255)), target)
    white = chw[0, 0] == 255
    return int(white.any(axis=0).argmax()), int(white.any(axis=1).argmax())


@pytest.mark.parametrize('size', [(101, 61), (333, 500), (640, 480), (5184, 3456), (97, 100)])
def test_the_returned_pad_is_the_paste_offset(size: tuple[int, int]) -> None:
    left, top = _paste_offset(size, 640)
    _chw, _scale, pad = letterbox_to_square(Image.new('RGB', size), 640)
    assert pad == (float(left), float(top))
    lb = letterbox_geometry(*size, 640)
    assert (lb.pad_w, lb.pad_h) == (float(left), float(top))


def test_an_odd_margin_round_trips_under_a_hundredth_of_a_pixel() -> None:
    lb = letterbox_geometry(101, 61, 640)
    assert (640 - lb.new_h) % 2 == 1  # the case that was half a pixel off
    for src_y in (0.0, 17.0, 60.0):
        net_y = src_y * lb.scale + lb.top
        _x1, y1, _x2, _y2 = undo_letterbox(
            (0.0, net_y, 1.0, net_y + 1), lb.scale, (lb.pad_w, lb.pad_h)
        )
        assert abs(y1 - src_y) < 0.01


def test_an_even_margin_is_unchanged() -> None:
    lb = letterbox_geometry(640, 480, 640)
    assert (lb.left, lb.top, lb.pad_w, lb.pad_h) == (0, 80, 0.0, 80.0)


def test_a_degenerate_frame_is_refused() -> None:
    with pytest.raises(ValueError, match='degenerate'):
        letterbox_geometry(0, 10, 640)


def test_whole_square_and_crop_rect_keep_their_arithmetic() -> None:
    assert whole_square_geometry(400, 200, 100) == (200, 100, 50, 0)
    assert crop_rect((10.4, 20.6, 50.5, 70.49), 120, 90) == (10, 21, 50, 70)
