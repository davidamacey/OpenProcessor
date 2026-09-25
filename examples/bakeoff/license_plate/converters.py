"""Public license-plate benchmark formats, as example dataset converters.

Registered into ``scripts.curation.bakeoff.datasets`` on import (the example
profile lists this module in ``converter_modules``). Every format here is
single-class, so boxes go to class 0 (``YoloWriter.write_abs``'s default).

* ``ccpd``     --- CCPD encodes the plate bbox in the filename.
* ``ufpr``     --- UFPR-ALPR: per-image ``.txt`` with ``position_plate: x y w h``.
* ``openalpr`` --- OpenALPR endtoend benchmark: ``<name> x y w h <text>`` lines.

Pascal VOC plate sets (e.g. the Kaggle car-plate set) use the built-in
generic ``voc`` converter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.curation.bakeoff.datasets import AbsBox, image_size, iter_images, register_converter


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from scripts.curation.bakeoff.datasets import YoloWriter


PROFILE = 'license_plate'


def parse_ccpd_filename(name: str) -> AbsBox | None:
    """Extract the plate bbox from a CCPD filename.

    CCPD fields are ``-``-separated; the 3rd is the box as ``x1&y1_x2&y2``
    in absolute pixels, e.g. ``...-154&383_386&473-...``.
    """
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    parts = stem.split('-')
    if len(parts) < 3:
        return None
    try:
        tl, br = parts[2].split('_')
        x1, y1 = (float(v) for v in tl.split('&'))
        x2, y2 = (float(v) for v in br.split('&'))
    except (ValueError, IndexError):
        return None
    return (x1, y1, x2, y2)


def parse_ufpr_annotation(text: str) -> list[AbsBox]:
    """``position_plate: x y w h`` lines (abs px, top-left + size) -> xyxy boxes."""
    boxes: list[AbsBox] = []
    for line in text.splitlines():
        if line.strip().lower().startswith('position_plate:'):
            nums = line.split(':', 1)[1].split()
            if len(nums) >= 4:
                try:
                    x, y, w, h = (float(v) for v in nums[:4])
                except ValueError:
                    continue
                boxes.append((x, y, x + w, y + h))
    return boxes


def parse_openalpr_annotation(text: str) -> list[AbsBox]:
    """``<name> x y w h <plate_text>`` lines (abs px) -> xyxy boxes."""
    boxes: list[AbsBox] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
    return boxes


def convert_ccpd(src: Path, writer: YoloWriter) -> int:
    n = 0
    for img in iter_images(src):
        box = parse_ccpd_filename(img.name)
        if box is None:
            continue
        dims = image_size(img)
        if dims is None:
            continue
        writer.write_abs(img.stem, img, [box], *dims)
        n += 1
    return n


def _convert_sidecar_txt(
    src: Path, writer: YoloWriter, parse: Callable[[str], list[AbsBox]]
) -> int:
    n = 0
    for img in iter_images(src):
        ann = img.with_suffix('.txt')
        if not ann.is_file():
            continue
        dims = image_size(img)
        if dims is None:
            continue
        boxes = parse(ann.read_text(encoding='utf-8', errors='ignore'))
        writer.write_abs(img.stem, img, boxes, *dims)
        n += 1
    return n


def convert_ufpr(src: Path, writer: YoloWriter) -> int:
    return _convert_sidecar_txt(src, writer, parse_ufpr_annotation)


def convert_openalpr(src: Path, writer: YoloWriter) -> int:
    return _convert_sidecar_txt(src, writer, parse_openalpr_annotation)


register_converter('ccpd', convert_ccpd, description='CCPD (bbox in filename)', example_for=PROFILE)
register_converter('ufpr', convert_ufpr, description='UFPR-ALPR sidecar .txt', example_for=PROFILE)
register_converter(
    'openalpr', convert_openalpr, description='OpenALPR endtoend .txt', example_for=PROFILE
)
