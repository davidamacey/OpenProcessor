"""COCO-format dataset reading (W10.2.1).

The annotation JSON is untrusted input: its structure is validated before
use, ``file_name`` values resolve through
:func:`~src.services.curation.dataset_import.paths.resolve_ref` (no ``..``,
no absolute path, no symlink out), and a file over the size cap is refused
unread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_COCO_JSON_BYTES, preview_max_files
from src.services.curation.dataset_import.paths import (
    DatasetPathNotAllowedError,
    PathGuard,
    resolve_ref,
)
from src.services.curation.dataset_import.safe_json import loads_json
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
    exif_transposed_size,
)


# The preview decodes at most this many image headers to check declared
# sizes; the job checks every image as it decodes it.
PREVIEW_SIZE_CHECKS = 200


@dataclass(frozen=True)
class CocoAnnotationFile:
    path: Path
    images_dir: Path
    split: str | None = None


def _infer_split(path: Path) -> str | None:
    name = path.stem.lower()
    if 'train' in name:
        return 'train'
    if 'val' in name:
        return 'val'
    if 'test' in name:
        return 'test'
    return None


def _clamp_bbox(
    x1: float, y1: float, x2: float, y2: float, *, tolerance: float = 1e-3
) -> list[float] | None:
    out = [x1, y1, x2, y2]
    for i, v in enumerate(out):
        if -tolerance <= v < 0.0 or 1.0 < v <= 1.0 + tolerance:
            out[i] = max(0.0, min(1.0, v))
        elif v < -tolerance or v > 1.0 + tolerance:
            return None
    return out


def _load_json(f: CocoAnnotationFile, issues: IssueCollector) -> dict[str, Any] | None:
    name = f.path.name
    try:
        size = f.path.stat().st_size
        if size > MAX_COCO_JSON_BYTES:
            issues.add('dataset_file_too_large', file=name)
            return None
        data = loads_json(f.path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        issues.add('coco_json_invalid', file=name, detail={'error': str(exc)[:200]})
        return None
    if not isinstance(data, dict) or not all(
        isinstance(data.get(k), list) for k in ('images', 'annotations', 'categories')
    ):
        issues.add('coco_json_invalid', file=name)
        return None
    return data


def _categories(data: dict[str, Any], issues: IssueCollector, file_name: str) -> dict[Any, str]:
    categories: dict[Any, str] = {}
    name_counts: dict[str, int] = {}
    for c in data['categories']:
        if not isinstance(c, dict) or 'id' not in c or not isinstance(c.get('name'), str):
            issues.add('coco_json_invalid', file=file_name, detail={'reason': 'bad category'})
            continue
        categories[c['id']] = c['name']
        name_counts[c['name']] = name_counts.get(c['name'], 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            issues.add('coco_category_duplicate_name', file=file_name, detail={'name': name})
    return categories


def _annotation_box(
    ann: dict[str, Any], categories: dict[Any, str], width: Any, height: Any
) -> LabelBox | None:
    """The normalized box for one annotation, or ``None`` when its
    category, size or coordinates are unusable."""
    try:
        bx, by, bw, bh = (float(v) for v in ann['bbox'])
        cat_name = categories.get(ann['category_id'])
        w, h = float(width), float(height)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (bx, by, bw, bh, w, h)):
        return None
    if cat_name is None or w <= 0 or h <= 0:
        return None
    clamped = _clamp_bbox(bx / w, by / h, (bx + bw) / w, (by + bh) / h)
    if clamped is None or clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return LabelBox(
        dataset_class=cat_name, bbox_norm=(clamped[0], clamped[1], clamped[2], clamped[3])
    )


def read_coco(
    files: list[CocoAnnotationFile],
    *,
    issues: IssueCollector,
    path_guard: PathGuard | None = None,
    size_checks: int = PREVIEW_SIZE_CHECKS,
    class_ids: dict[str, int] | None = None,
) -> list[ScanEntry]:
    """Read one or more COCO annotation files into ``ScanEntry`` rows.

    Each file's ``categories[].name`` are the dataset classes;
    ``category_id`` is never compared with a registry id (W10.2.1: "never
    by index"). The first ``size_checks`` images have their declared
    ``width``/``height`` compared with the EXIF-transposed header size
    (``image_size_mismatch``: the image is skipped).
    """
    entries: list[ScanEntry] = []
    checked = 0
    for f in files:
        if path_guard is not None and not path_guard(f.images_dir):
            raise DatasetPathNotAllowedError(str(f.images_dir))
        data = _load_json(f, issues)
        if data is None:
            continue
        categories = _categories(data, issues, f.path.name)
        if class_ids is not None:
            for cat_id, cat_name in categories.items():
                if isinstance(cat_id, int):
                    class_ids[cat_name] = cat_id

        anns_by_image: dict[Any, list[dict[str, Any]]] = {}
        for ann in data['annotations']:
            if isinstance(ann, dict) and 'image_id' in ann:
                anns_by_image.setdefault(ann['image_id'], []).append(ann)

        split = f.split or _infer_split(f.path)

        for img in data['images']:
            if not isinstance(img, dict) or not isinstance(img.get('file_name'), str):
                issues.add('coco_json_invalid', file=f.path.name, detail={'reason': 'bad image'})
                continue
            file_name = img['file_name']
            abs_path = resolve_ref(f.images_dir, file_name)
            if abs_path is None or (path_guard is not None and not path_guard(abs_path)):
                issues.add('image_path_not_servable', file=file_name)
                continue
            width, height = img.get('width'), img.get('height')
            declared = _declared_size(width, height)
            if declared is not None and checked < size_checks:
                checked += 1
                actual = exif_transposed_size(abs_path)
                if actual is not None and actual != declared:
                    issues.add(
                        'image_size_mismatch',
                        file=file_name,
                        detail={'declared': list(declared), 'actual': list(actual)},
                    )
                    continue
            boxes: list[LabelBox] = []
            has_non_crowd = False
            for ann in anns_by_image.get(img.get('id'), []):
                if ann.get('iscrowd'):
                    issues.add('coco_crowd_skipped', file=file_name)
                    continue
                has_non_crowd = True
                box = _annotation_box(ann, categories, width, height)
                if box is None:
                    issues.add('coco_bbox_out_of_image', file=file_name)
                    continue
                boxes.append(box)

            label_state: LabelState
            if not has_non_crowd:
                label_state = 'negative'
            elif boxes:
                label_state = 'labeled'
            else:
                label_state = 'unlabeled'

            entries.append(
                ScanEntry(
                    rel_path=file_name,
                    source_stem=Path(file_name).stem,
                    abs_image_path=abs_path,
                    split=split,
                    label_state=label_state,
                    boxes=boxes,
                    declared_size=declared,
                )
            )
            if len(entries) > preview_max_files():
                issues.add('dataset_too_large', file=f.path.name)
                return entries
    return entries


def _declared_size(width: Any, height: Any) -> tuple[int, int] | None:
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def scan_coco(
    files: list[CocoAnnotationFile], *, path_guard: PathGuard | None = None
) -> DatasetScan:
    issues = IssueCollector()
    class_ids: dict[str, int] = {}
    entries = read_coco(files, issues=issues, path_guard=path_guard, class_ids=class_ids)
    root = files[0].images_dir if files else Path()
    return DatasetScan(
        format='coco',
        root=root,
        entries=entries,
        issues=issues,
        class_ids=class_ids,
        class_box_counts=class_box_counts(entries),
    )


__all__ = ['PREVIEW_SIZE_CHECKS', 'CocoAnnotationFile', 'read_coco', 'scan_coco']
