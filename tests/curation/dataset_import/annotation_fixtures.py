"""Offline writers for the CVAT, Label Studio, KITTI, CreateML and LabelMe
layouts (small hand-written public-style datasets, no downloads)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from curation.dataset_import.harness import write_image


if TYPE_CHECKING:
    from pathlib import Path

COLORS = ['red', 'green', 'blue', 'white']


def _image(path: Path, index: int, size: tuple[int, int]) -> tuple[int, int]:
    actual = (size[0] + index, size[1])  # distinct bytes per image: no dedup
    write_image(path, size=actual, color=COLORS[index % 4])
    return actual


CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><name>demo</name></task></meta>
  {images}
</annotations>
"""


def cvat_box(
    label: str, box: tuple[float, float, float, float], *, extra: str = '', occluded: int = 0
) -> str:
    x1, y1, x2, y2 = box
    return (
        f'<box label="{label}" source="manual" occluded="{occluded}" '
        f'xtl="{x1}" ytl="{y1}" xbr="{x2}" ybr="{y2}" z_order="0"{extra}></box>'
    )


def write_cvat(
    root: Path,
    images: dict[str, list[str]],
    *,
    size: tuple[int, int] = (100, 100),
    xml_size: tuple[int, int] | str | None = 'same',
    subsets: dict[str, str] | None = None,
) -> None:
    """``images``: file name (``a.jpg``) -> shape snippets. Writes
    ``annotations.xml`` and ``images/<name>``."""
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (name, shapes) in enumerate(images.items()):
        actual = _image(root / 'images' / name, i, size)
        declared = actual if xml_size == 'same' else xml_size
        dims = ''
        if isinstance(declared, tuple):
            dims = f' width="{declared[0]}" height="{declared[1]}"'
        subset = f' subset="{subsets[name]}"' if subsets and name in subsets else ''
        parts.append(f'<image id="{i}" name="{name}"{dims}{subset}>{"".join(shapes)}</image>')
    (root / 'annotations.xml').write_text(CVAT_XML.format(images='\n  '.join(parts)))


def ls_rect(
    label: str, x: float, y: float, w: float, h: float, *, rotation: float = 0, size=(100, 100)
) -> dict[str, Any]:
    """A Label Studio rectanglelabels result; x, y, w, h are percentages."""
    return {
        'original_width': size[0],
        'original_height': size[1],
        'image_rotation': 0,
        'value': {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'rotation': rotation,
            'rectanglelabels': [label],
        },
        'id': f'r{x}{y}',
        'from_name': 'label',
        'to_name': 'image',
        'type': 'rectanglelabels',
    }


def ls_task(image_ref: str, results: list[dict[str, Any]] | None, **annotation: Any) -> dict:
    """``results=None`` is a task with no annotation at all."""
    annotations = (
        []
        if results is None
        else [{'id': 1, 'result': results, 'was_cancelled': False, **annotation}]
    )
    return {'id': 1, 'data': {'image': image_ref}, 'annotations': annotations, 'predictions': []}


def write_labelstudio(
    root: Path,
    tasks: list[dict[str, Any]],
    *,
    images: dict[str, tuple[int, int]] | None = None,
    name: str = 'project-1-export.json',
) -> None:
    """Writes the export JSON and ``images/<file>`` for each ``images`` entry."""
    root.mkdir(parents=True, exist_ok=True)
    for i, (file, size) in enumerate((images or {}).items()):
        _image(root / 'images' / file, i, size)
    (root / name).write_text(json.dumps(tasks))


def write_kitti(
    root: Path,
    images: dict[str, list[str] | None],
    *,
    size: tuple[int, int] = (200, 100),
    splits: dict[str, list[str]] | None = None,
) -> None:
    """``images``: stem -> label lines (``None`` = no label file). Writes
    ``image_2/<stem>.png`` and ``label_2/<stem>.txt``."""
    (root / 'label_2').mkdir(parents=True, exist_ok=True)
    for i, (stem, lines) in enumerate(images.items()):
        actual = (size[0] + i, size[1])
        write_image(root / f'image_2/{stem}.png', size=actual, color=COLORS[i % 4])
        if lines is not None:
            (root / f'label_2/{stem}.txt').write_text(''.join(f'{line}\n' for line in lines))
    if splits:
        (root / 'ImageSets').mkdir(parents=True, exist_ok=True)
        for split, stems in splits.items():
            (root / f'ImageSets/{split}.txt').write_text(''.join(f'{s}\n' for s in stems))


def kitti_line(name: str, box: tuple[float, float, float, float], *, extra: str = '') -> str:
    x1, y1, x2, y2 = box
    return f'{name} 0.00 0 -1.57 {x1} {y1} {x2} {y2} 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59{extra}'


def cml_ann(label: str, cx: float, cy: float, w: float, h: float) -> dict[str, Any]:
    return {'label': label, 'coordinates': {'x': cx, 'y': cy, 'width': w, 'height': h}}


def write_createml(
    root: Path,
    records: dict[str, list[dict[str, Any]]],
    *,
    size: tuple[int, int] = (100, 100),
    json_name: str = '_annotations.createml.json',
) -> None:
    """``records``: image file name -> annotations; images sit beside the JSON."""
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (file, anns) in enumerate(records.items()):
        _image(root / file, i, size)
        out.append({'image': file, 'annotations': anns})
    (root / json_name).write_text(json.dumps(out))


def lm_rect(label: str, p1: tuple[float, float], p2: tuple[float, float]) -> dict[str, Any]:
    return {
        'label': label,
        'points': [list(p1), list(p2)],
        'group_id': None,
        'shape_type': 'rectangle',
        'flags': {},
    }


def write_labelme(
    root: Path,
    images: dict[str, list[dict[str, Any]]],
    *,
    size: tuple[int, int] = (100, 100),
    declare_size: bool = True,
) -> None:
    """``images``: stem -> shapes. Writes ``<stem>.jpg`` and ``<stem>.json``."""
    root.mkdir(parents=True, exist_ok=True)
    for i, (stem, shapes) in enumerate(images.items()):
        actual = _image(root / f'{stem}.jpg', i, size)
        doc = {
            'version': '5.2.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': f'{stem}.jpg',
            'imageData': None,
            'imageHeight': actual[1] if declare_size else None,
            'imageWidth': actual[0] if declare_size else None,
        }
        (root / f'{stem}.json').write_text(json.dumps(doc))
