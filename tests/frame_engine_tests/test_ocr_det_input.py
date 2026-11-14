"""``frame_engine/ocr/det_input.py`` against PaddleX release/3.3 ``DetResizeForTest`` (processors.py
sha256 7b714e1a...a5de) with ``max_side_limit`` 1920: the table is what the reference returned."""

from __future__ import annotations

import pytest

from frame_engine.ocr.det_input import det_size


# (height, width, limit_type, limit_side_len, expected height, expected width)
REFERENCE = [
    (480, 640, 'min', 736, 736, 992),
    (375, 500, 'min', 736, 736, 992),
    (1080, 1920, 'min', 736, 1088, 1920),
    (3648, 5472, 'min', 736, 1280, 1920),
    (100, 300, 'min', 736, 640, 1920),
    (30, 40, 'min', 736, 736, 992),
    (640, 640, 'min', 736, 736, 736),
    (2000, 300, 'min', 736, 1920, 288),
    (1500, 1000, 'min', 736, 1504, 992),
    (4000, 6000, 'min', 736, 1280, 1920),
    (480, 640, 'max', 960, 480, 640),
    (375, 500, 'max', 960, 384, 512),
    (1080, 1920, 'max', 960, 544, 960),
    (3648, 5472, 'max', 960, 640, 960),
    (100, 300, 'max', 960, 96, 288),
    (30, 40, 'max', 960, 32, 32),
    (640, 640, 'max', 960, 640, 640),
    (2000, 300, 'max', 960, 960, 128),
    (1500, 1000, 'max', 960, 960, 640),
    (4000, 6000, 'max', 960, 640, 960),
    (3648, 5472, 'max', 1280, 864, 1280),
    (2000, 300, 'max', 1280, 1280, 192),
    (30, 40, 'min', 64, 64, 96),
    (4000, 6000, 'min', 64, 1280, 1920),
]


@pytest.mark.parametrize(('h', 'w', 'limit', 'side', 'eh', 'ew'), REFERENCE)
def test_the_size_equals_the_reference(
    h: int, w: int, limit: str, side: int, eh: int, ew: int
) -> None:
    assert det_size(h, w, side, limit, 1920) == (eh, ew)


def test_an_unknown_limit_is_refused() -> None:
    with pytest.raises(ValueError, match='det_limit'):
        det_size(480, 640, 960, 'long', 1920)
