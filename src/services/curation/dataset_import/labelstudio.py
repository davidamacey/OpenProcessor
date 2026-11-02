"""Label Studio JSON export (rectangle labels) dataset reading.

Layout::

    root/<export>.json            (the full JSON export: a list of tasks)
    root/images/<file>            (else the root)

Classes are the ``rectanglelabels`` texts, so ``DatasetScan.class_ids`` stays
empty. Coordinates are percentages of the image, so no image is decoded;
``original_width/height`` is carried as the declared size and the job checks
it. A task with no (non-cancelled) annotation is ``unlabeled``; one annotation
with no box is ``negative``; boxes make it ``labeled``. Several annotations on
one task are refused rather than guessed. A rotated rectangle and every other
geometry (polygon, brush, keypoint, ellipse) is skipped
(``shape_not_box_skipped``) and still counts as present. The image reference
is ``data.image`` with a ``/data/upload/<n>/`` or ``/data/local-files/?d=``
prefix removed; it resolves through
:func:`~src.services.curation.dataset_import.paths.resolve_ref`, and a remote
URL is never fetched. The minimized ``JSON_MIN`` export is not read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.json_sniff import json_files, sniff_json
from src.services.curation.dataset_import.limits import (
    MAX_LABELSTUDIO_JSON_BYTES,
    preview_max_files,
)
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
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.scan import DatasetScan, LabelBox, ScanEntry
from src.services.curation.dataset_import.yolo import IMAGE_EXTENSIONS


if TYPE_CHECKING:
    from pathlib import Path

_UPLOAD = re.compile(r'^/data/upload/\d+/(.+)$')
_LOCAL_FILES = '/data/local-files/'
_GEOMETRY_TYPES = frozenset(
    {
        'polygonlabels',
        'brushlabels',
        'keypointlabels',
        'ellipselabels',
        'rectangle',
        'polygon',
        'brush',
        'keypoint',
        'ellipse',
    }
)
# Values are percentages: 0.1 % is the same slack as 1e-3 on a unit square.
_PERCENT_TOLERANCE = 0.1


def is_labelstudio(root: Path) -> bool:
    return root.is_dir() and any(sniff_json(p) == 'labelstudio' for p in json_files(root))


@dataclass
class _Ctx:
    root: Path
    images_dirs: list[Path]
    issues: IssueCollector
    path_guard: PathGuard | None
    budget: SizeBudget


def _image_ref(data: dict[str, Any]) -> str | None:
    ref = data.get('image')
    if isinstance(ref, str) and ref:
        return ref
    candidates = [
        v
        for v in data.values()
        if isinstance(v, str) and any(v.lower().endswith(e) for e in IMAGE_EXTENSIONS)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _local_ref(ref: str) -> str | None:
    """The dataset-relative reference ``ref`` names, or ``None`` for a remote
    URL (never fetched)."""
    if '://' in ref.split('?', 1)[0]:
        return None
    if ref.startswith(_LOCAL_FILES):
        query = ref.split('?', 1)[1] if '?' in ref else ''
        for part in query.split('&'):
            if part.startswith('d='):
                return unquote(part[2:])
        return None
    match = _UPLOAD.match(ref)
    return match.group(1) if match else ref


def _number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(value)
    return number


def _rectangle(item: dict[str, Any], index: int, rel_file: str, ctx: _Ctx) -> LabelBox | None:
    value = item.get('value')
    labels = value.get('rectanglelabels') if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or not isinstance(labels, list)
        or len(labels) != 1
        or not isinstance(labels[0], str)
    ):
        ctx.issues.add(
            'label_row_malformed',
            file=rel_file,
            detail={'object': index, 'reason': 'not exactly one rectangle label'},
        )
        return None
    try:
        x, y = _number(value['x']), _number(value['y'])
        w, h = _number(value['width']), _number(value['height'])
        rotation = _number(value.get('rotation') or 0)
    except (KeyError, TypeError, ValueError):
        ctx.issues.add(
            'label_row_malformed',
            file=rel_file,
            detail={'object': index, 'reason': 'bad coordinates'},
        )
        return None
    if abs(rotation) > 1e-6:
        ctx.issues.add(
            'shape_not_box_skipped',
            file=rel_file,
            detail={'object': index, 'shape': 'rotated box'},
        )
        return None
    return box_from_coords(
        labels[0].strip(),
        (x, y, x + w, y + h),
        size=(100.0, 100.0),
        rel_file=rel_file,
        index=index,
        issues=ctx.issues,
        tolerance=_PERCENT_TOLERANCE,
    )


def _results(
    results: list[Any], rel_file: str, ctx: _Ctx
) -> tuple[list[LabelBox], bool, tuple[int, int] | None]:
    boxes: list[LabelBox] = []
    present = False
    declared: tuple[int, int] | None = None
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        kind = item.get('type')
        if kind == 'rectanglelabels':
            present = True
            if declared is None:
                declared = declared_size(item.get('original_width'), item.get('original_height'))
            box = _rectangle(item, index, rel_file, ctx)
            if box is not None:
                boxes.append(box)
        elif kind in _GEOMETRY_TYPES:
            present = True
            ctx.issues.add(
                'shape_not_box_skipped', file=rel_file, detail={'object': index, 'shape': kind}
            )
    return boxes, present, declared


def _read_task(task: dict[str, Any], rel_file: str, ctx: _Ctx) -> ScanEntry | None:
    ref = _image_ref(task['data'])
    if ref is None:
        ctx.issues.add('labelstudio_task_skipped', file=rel_file, detail={'reason': 'no image'})
        return None
    local = _local_ref(ref)
    if local is None:
        ctx.issues.add(
            'image_file_missing',
            file=rel_file,
            detail={'filename': ref[:200], 'reason': 'remote reference'},
        )
        return None
    annotations = task.get('annotations') or []
    active = [a for a in annotations if isinstance(a, dict) and not a.get('was_cancelled')]
    if len(active) > 1:
        ctx.issues.add(
            'labelstudio_task_skipped',
            file=rel_file,
            detail={'reason': 'several annotations on one task', 'count': len(active)},
        )
        return None
    image = find_image(
        ctx.images_dirs, [local], rel_file=rel_file, issues=ctx.issues, path_guard=ctx.path_guard
    )
    if image is None:
        return None
    boxes: list[LabelBox] = []
    present = False
    declared: tuple[int, int] | None = None
    if active:
        results = active[0].get('result') or []
        boxes, present, declared = _results(results, rel_file, ctx)
    if declared is not None:
        sized = resolve_size(declared, image, rel_file, ctx.issues, ctx.budget)
        if sized is None:
            return None
    state = 'unlabeled' if not active else entry_state(present, boxes)
    return ScanEntry(
        rel_path=rel_path(image, ctx.root),
        source_stem=image.stem,
        abs_image_path=image,
        split=None,
        label_state=state,
        boxes=boxes,
        declared_size=declared,
    )


def _load_tasks(path: Path, root: Path, issues: IssueCollector) -> list[dict[str, Any]] | None:
    rel = rel_path(path, root)
    data = load_json_file(path, MAX_LABELSTUDIO_JSON_BYTES, rel, issues, 'labelstudio_json_invalid')
    if data is None:
        return None
    if not isinstance(data, list) or not all(
        isinstance(t, dict) and isinstance(t.get('data'), dict) for t in data
    ):
        issues.add(
            'labelstudio_json_invalid',
            file=rel,
            detail={'reason': 'not a list of tasks with a data object'},
        )
        return None
    return data


def scan_labelstudio(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()
    files = json_files(root)
    if not files:
        issues.add('labelstudio_json_invalid', file=root.name, detail={'reason': 'no JSON file'})
        return finish_scan('labelstudio', root, [], issues)
    loaded = [(f, _load_tasks(f, root, issues)) for f in files]
    if sum(len(t) for _f, t in loaded if t) > preview_max_files():
        return scan_too_large('labelstudio', root, issues)
    images_dirs = [d for d in (root / 'images', root) if d.is_dir()]
    if path_guard is not None and not all(path_guard(d) for d in images_dirs):
        raise DatasetPathNotAllowedError(str(root / 'images'))
    ctx = _Ctx(root, images_dirs, issues, path_guard, SizeBudget())
    entries: list[ScanEntry] = []
    for file, tasks in loaded:
        for index, task in enumerate(tasks or []):
            entry = _read_task(task, f'{rel_path(file, root)}#{index}', ctx)
            if entry is not None:
                entries.append(entry)
    return finish_scan('labelstudio', root, entries, issues)


__all__ = ['is_labelstudio', 'scan_labelstudio']
