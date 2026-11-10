"""Export resize/crop decodes through the oriented helper.

A phone photo stores its pixels sideways and a tag; the boxes are drawn on the upright frame.
An export that resizes or crops from the raw pixels lands the boxes beside the object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from PIL import Image

from src.services.curation import export_support


if TYPE_CHECKING:
    from pathlib import Path


#: tag -> stored (w, h) for an upright 40x20 picture; 5-8 are stored transposed.
_STORED = {
    1: (40, 20),
    2: (40, 20),
    3: (40, 20),
    4: (40, 20),
    5: (20, 40),
    6: (20, 40),
    7: (20, 40),
    8: (20, 40),
}


def _tagged_jpeg(path: Path, tag: int) -> None:
    img = Image.new('RGB', _STORED[tag], (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = tag
    img.save(path, format='JPEG', quality=95, exif=exif)


@pytest.mark.parametrize('tag', range(1, 9))
def test_aspect_resize_output_is_upright(tmp_path: Path, tag: int) -> None:
    src, dest = tmp_path / 's.jpg', tmp_path / 'd.jpg'
    _tagged_jpeg(src, tag)
    _, ok, err = export_support._copy_or_resize_one(str(src), str(dest), 'aspect', 40)
    assert ok, err
    with Image.open(dest) as out:
        assert out.size == (40, 20)


def test_tag_one_crop_is_unchanged(tmp_path: Path) -> None:
    src, dest = tmp_path / 's.jpg', tmp_path / 'd.jpg'
    _tagged_jpeg(src, 1)
    _, ok, err = export_support._copy_or_resize_one(
        str(src), str(dest), None, 40, crop_norm=(0.0, 0.0, 0.5, 1.0)
    )
    assert ok, err
    with Image.open(dest) as out:
        assert out.size == (20, 20)
