"""LabelMe JSON dataset reading.

Layout::

    root/<name>.json              (or root/{train,val,test}/<name>.json)
    root/<name>.jpg               (beside its JSON, else in an ``images/`` directory there)

One JSON per image: ``shapes`` (each with ``label``, ``shape_type`` and
``points``), ``imagePath``, ``imageWidth`` and ``imageHeight``. Only
``rectangle`` shapes (two corner points, any order, pixels) are boxes; every
other shape type (polygon, circle, line, point, linestrip) is skipped with
``shape_not_box_skipped`` and still counts as present. Classes are the
``label`` texts (``DatasetScan.class_ids`` stays empty). An image with no
shape is a ``negative``. ``imagePath`` resolves through
:func:`~src.services.curation.dataset_import.paths.resolve_ref`; when it names
nothing inside the dataset (LabelMe writes ``..\\images\\a.jpg`` style paths)
the image is looked up by the JSON's own file stem. A split comes from the
``train`` / ``val`` / ``valid`` / ``validation`` / ``test`` directory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.json_sniff import has_json_of_kind, split_json_files
from src.services.curation.dataset_import.limits import MAX_LABELME_JSON_BYTES, preview_max_files
from src.services.curation.dataset_import.native_common import (
    SizeBudget,
    box_from_coords,
    declared_size,
    entry_state,
    find_image,
    finish_scan,
    load_json_file,
    rel_path,
    resolve_size,
    scan_too_large,
    stem_refs,
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.scan import DatasetScan, LabelBox, ScanEntry


if TYPE_CHECKING:
    from pathlib import Path


def is_labelme(root: Path) -> bool:
    return has_json_of_kind(root, 'labelme')


@dataclass
class _Ctx:
    root: Path
    issues: IssueCollector
    path_guard: PathGuard | None
    budget: SizeBudget


def _corner_box(
    shape: dict[str, Any], index: int, size: tuple[int, int], rel_file: str, ctx: _Ctx
) -> LabelBox | None:
    try:
        (xa, ya), (xb, yb) = shape['points']
        xs, ys = [float(xa), float(xb)], [float(ya), float(yb)]
        if not all(math.isfinite(v) for v in (*xs, *ys)):
            raise ValueError('not finite')
        label = shape['label']
        if not isinstance(label, str):
            raise TypeError('label')
    except (KeyError, TypeError, ValueError):
        ctx.issues.add(
            'label_row_malformed',
            file=rel_file,
            detail={'object': index, 'reason': 'bad rectangle'},
        )
        return None
    return box_from_coords(
        label.strip(),
        (min(xs), min(ys), max(xs), max(ys)),
        size=size,
        rel_file=rel_file,
        index=index,
        issues=ctx.issues,
    )


def _shapes(
    shapes: list[Any], size: tuple[int, int], rel_file: str, ctx: _Ctx
) -> tuple[list[LabelBox], bool]:
    boxes: list[LabelBox] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            ctx.issues.add(
                'label_row_malformed',
                file=rel_file,
                detail={'object': index, 'reason': 'shape is not an object'},
            )
            continue
        kind = shape.get('shape_type') or 'polygon'
        if kind != 'rectangle':
            ctx.issues.add(
                'shape_not_box_skipped', file=rel_file, detail={'object': index, 'shape': kind}
            )
            continue
        box = _corner_box(shape, index, size, rel_file, ctx)
        if box is not None:
            boxes.append(box)
    return boxes, bool(shapes)


def _read_file(path: Path, split: str | None, ctx: _Ctx) -> ScanEntry | None:
    rel_file = rel_path(path, ctx.root)
    doc = load_json_file(path, MAX_LABELME_JSON_BYTES, rel_file, ctx.issues, 'labelme_json_invalid')
    if doc is None:
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get('shapes'), list):
        ctx.issues.add('labelme_json_invalid', file=rel_file, detail={'reason': 'no shapes list'})
        return None
    image_path = doc.get('imagePath')
    refs = ([image_path] if isinstance(image_path, str) and image_path else []) + stem_refs(
        path.stem
    )
    image = find_image(
        [path.parent, path.parent / 'images'],
        refs,
        rel_file=rel_file,
        issues=ctx.issues,
        path_guard=ctx.path_guard,
    )
    if image is None:
        return None
    sized = resolve_size(
        declared_size(doc.get('imageWidth'), doc.get('imageHeight')),
        image,
        rel_file,
        ctx.issues,
        ctx.budget,
    )
    if sized is None:
        return None
    boxes, present = _shapes(doc['shapes'], sized[0], rel_file, ctx)
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=split,
        label_state=entry_state(present, boxes),
        boxes=boxes,
        declared_size=sized[1],
    )


def scan_labelme(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()
    files = split_json_files(root)
    if len(files) > preview_max_files():
        return scan_too_large('labelme', root, issues)
    ctx = _Ctx(root, issues, path_guard, SizeBudget())
    entries = [e for e in (_read_file(f, split, ctx) for f, split in files) if e is not None]
    if not files:
        issues.add('labelme_json_invalid', file=root.name, detail={'reason': 'no JSON file'})
    return finish_scan('labelme', root, entries, issues)


__all__ = ['is_labelme', 'scan_labelme']
