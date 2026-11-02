"""CreateML JSON dataset reading.

Layout (Create ML, and Roboflow's "CreateML" export)::

    root/<name>.json              (or root/{train,valid,test}/_annotations.createml.json)
    root/<image>                  (beside its JSON, else in an ``images/`` directory there)

A file is a list of ``{"image": "a.jpg", "annotations": [{"label": ...,
"coordinates": {"x", "y", "width", "height"}}]}``. ``x`` and ``y`` are the box
**centre** in pixels, so the image size is read from each image. Classes are
the ``label`` texts (``DatasetScan.class_ids`` stays empty). A record with no
annotation is a ``negative``. A split comes from the ``train`` / ``val`` /
``valid`` / ``validation`` / ``test`` directory holding the JSON. ``image``
resolves through :func:`~src.services.curation.dataset_import.paths.resolve_ref`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.json_sniff import has_json_of_kind, split_json_files
from src.services.curation.dataset_import.limits import MAX_CREATEML_JSON_BYTES, preview_max_files
from src.services.curation.dataset_import.native_common import (
    SizeBudget,
    box_from_coords,
    entry_state,
    find_image,
    finish_scan,
    load_json_file,
    rel_path,
    resolve_size,
    scan_too_large,
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.scan import DatasetScan, LabelBox, ScanEntry


if TYPE_CHECKING:
    from pathlib import Path


def is_createml(root: Path) -> bool:
    return has_json_of_kind(root, 'createml')


@dataclass
class _Ctx:
    root: Path
    issues: IssueCollector
    path_guard: PathGuard | None
    budget: SizeBudget


def _number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(value)
    return number


def _annotation(
    ann: Any, index: int, size: tuple[int, int], rel_file: str, ctx: _Ctx
) -> LabelBox | None:
    try:
        coords = ann['coordinates']
        cx, cy = _number(coords['x']), _number(coords['y'])
        w, h = _number(coords['width']), _number(coords['height'])
        label = ann['label']
        if not isinstance(label, str):
            raise TypeError('label')
    except (KeyError, TypeError, ValueError):
        ctx.issues.add(
            'label_row_malformed',
            file=rel_file,
            detail={'object': index, 'reason': 'bad annotation'},
        )
        return None
    return box_from_coords(
        label.strip(),
        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        size=size,
        rel_file=rel_file,
        index=index,
        issues=ctx.issues,
    )


def _read_record(
    record: dict[str, Any], json_path: Path, split: str | None, index: int, ctx: _Ctx
) -> ScanEntry | None:
    rel_file = f'{rel_path(json_path, ctx.root)}#{index}'
    ref = record['image']
    image = find_image(
        [json_path.parent, json_path.parent / 'images'],
        [ref],
        rel_file=rel_file,
        issues=ctx.issues,
        path_guard=ctx.path_guard,
    )
    if image is None:
        return None
    sized = resolve_size(None, image, rel_file, ctx.issues, ctx.budget)
    if sized is None:
        return None
    annotations = record['annotations']
    boxes = [
        b
        for b in (_annotation(a, i, sized[0], rel_file, ctx) for i, a in enumerate(annotations))
        if b is not None
    ]
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=split,
        label_state=entry_state(bool(annotations), boxes),
        boxes=boxes,
    )


def _load_records(path: Path, root: Path, issues: IssueCollector) -> list[dict[str, Any]] | None:
    rel = rel_path(path, root)
    data = load_json_file(path, MAX_CREATEML_JSON_BYTES, rel, issues, 'createml_json_invalid')
    if data is None:
        return None
    if not isinstance(data, list) or not all(
        isinstance(r, dict)
        and isinstance(r.get('image'), str)
        and isinstance(r.get('annotations'), list)
        for r in data
    ):
        issues.add(
            'createml_json_invalid',
            file=rel,
            detail={'reason': 'not a list of {image, annotations} records'},
        )
        return None
    return data


def scan_createml(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()
    files = split_json_files(root)
    if not files:
        issues.add('createml_json_invalid', file=root.name, detail={'reason': 'no JSON file'})
        return finish_scan('createml', root, [], issues)
    loaded = [(f, split, _load_records(f, root, issues)) for f, split in files]
    if sum(len(r) for _f, _s, r in loaded if r) > preview_max_files():
        return scan_too_large('createml', root, issues)
    ctx = _Ctx(root, issues, path_guard, SizeBudget())
    entries: list[ScanEntry] = []
    for file, split, records in loaded:
        for index, record in enumerate(records or []):
            entry = _read_record(record, file, split, index, ctx)
            if entry is not None:
                entries.append(entry)
    return finish_scan('createml', root, entries, issues)


__all__ = ['is_createml', 'scan_createml']
