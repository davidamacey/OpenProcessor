"""KITTI object-detection dataset reading.

Layout (the root, or its ``training/`` directory)::

    root/label_2/<id>.txt         (one object per line, 15 fields, 16 with a score)
    root/image_2/<id>.png         (else images/, else the root)
    root/ImageSets/{train,val,test}.txt   (optional, for splits)

A line is ``type truncated occluded alpha left top right bottom h w l x y z
rotation_y``; the box is the four pixel fields and the type text is the class
name (KITTI has no class index, so ``DatasetScan.class_ids`` stays empty). The
3D fields are ignored. ``DontCare`` regions are skipped and count as present.
A label file is what makes an image part of the dataset: an empty one is a
``negative``, an image without one is not imported. Pixel coordinates need
the image size, so every image header is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.capped_io import read_text_capped
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_LABEL_FILE_BYTES, preview_max_files
from src.services.curation.dataset_import.native_common import (
    SizeBudget,
    box_from_coords,
    entry_state,
    find_image,
    finish_scan,
    read_split_lists,
    rel_path,
    resolve_size,
    scan_too_large,
    stem_refs,
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.scan import DatasetScan, LabelBox, ScanEntry


if TYPE_CHECKING:
    from pathlib import Path

_MIN_FIELDS = 15


def _base(root: Path) -> Path | None:
    for base in (root, root / 'training'):
        if (base / 'label_2').is_dir():
            return base
    return None


def is_kitti(root: Path) -> bool:
    base = _base(root) if root.is_dir() else None
    return base is not None and any((base / 'label_2').glob('*.txt'))


@dataclass
class _Ctx:
    root: Path
    image_dir: Path
    issues: IssueCollector
    path_guard: PathGuard | None
    splits: dict[str, str]


def _parse_lines(
    text: str, size: tuple[int, int], rel_file: str, ctx: _Ctx
) -> tuple[list[LabelBox], bool]:
    boxes: list[LabelBox] = []
    present = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        tokens = raw.split()
        if not tokens:
            continue
        present = True
        if tokens[0] == 'DontCare':
            ctx.issues.add(
                'shape_not_box_skipped', file=rel_file, line=lineno, detail={'shape': 'DontCare'}
            )
            continue
        try:
            if len(tokens) < _MIN_FIELDS:
                raise ValueError('too few fields')
            left, top, right, bottom = (float(t) for t in tokens[4:8])
            if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                raise ValueError('not finite')
        except ValueError:
            ctx.issues.add('label_row_malformed', file=rel_file, line=lineno)
            continue
        box = box_from_coords(
            tokens[0],
            (left, top, right, bottom),
            size=size,
            rel_file=rel_file,
            index=lineno,
            issues=ctx.issues,
            line=lineno,
        )
        if box is not None:
            boxes.append(box)
    return boxes, present


def _read_label(path: Path, ctx: _Ctx, budget: SizeBudget) -> ScanEntry | None:
    rel_file = rel_path(path, ctx.root)
    text = read_text_capped(path, MAX_LABEL_FILE_BYTES)
    if text is None:
        ctx.issues.add('dataset_file_too_large', file=rel_file)
        return None
    image = find_image(
        [ctx.image_dir],
        stem_refs(path.stem),
        rel_file=rel_file,
        issues=ctx.issues,
        path_guard=ctx.path_guard,
    )
    if image is None:
        return None
    sized = resolve_size(None, image, rel_file, ctx.issues, budget)
    if sized is None:
        return None
    boxes, present = _parse_lines(text, sized[0], rel_file, ctx)
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=ctx.splits.get(path.stem),
        label_state=entry_state(present, boxes),
        boxes=boxes,
    )


def scan_kitti(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()
    base = _base(root)
    if base is None:
        issues.add('label_file_missing', file='label_2')
        return finish_scan('kitti', root, [], issues)
    label_files = sorted((base / 'label_2').glob('*.txt'))
    if len(label_files) > preview_max_files():
        return scan_too_large('kitti', root, issues)
    image_dir = next((d for d in (base / 'image_2', base / 'images') if d.is_dir()), base)
    if path_guard is not None and not path_guard(image_dir):
        raise DatasetPathNotAllowedError(str(image_dir))
    ctx = _Ctx(root, image_dir, issues, path_guard, read_split_lists(base / 'ImageSets', issues))
    budget = SizeBudget()
    entries = [e for e in (_read_label(p, ctx, budget) for p in label_files) if e is not None]
    return finish_scan('kitti', root, entries, issues)


__all__ = ['is_kitti', 'scan_kitti']
