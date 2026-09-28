"""COCO-format dataset reading (W10.2.1)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
)


PathGuard = Callable[[Path], bool]


class DatasetPathNotAllowedError(Exception):
    pass


class CocoJsonInvalidError(Exception):
    pass


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


def read_coco(
    files: list[CocoAnnotationFile], *, issues: IssueCollector, path_guard: PathGuard | None = None
) -> list[ScanEntry]:
    """Read one or more COCO annotation files into ``ScanEntry`` rows.

    Each file's ``categories[].name`` are the dataset classes;
    ``category_id`` is never compared with a registry id (W10.2.1: "never
    by index").
    """
    entries: list[ScanEntry] = []
    for f in files:
        if path_guard is not None and not path_guard(f.images_dir):
            raise DatasetPathNotAllowedError(str(f.images_dir))
        try:
            data = json.loads(f.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            issues.add('coco_json_invalid', file=str(f.path.name), detail={'error': str(exc)})
            continue
        if not all(k in data for k in ('images', 'annotations', 'categories')):
            issues.add('coco_json_invalid', file=str(f.path.name))
            continue

        categories = {c['id']: c['name'] for c in data['categories']}
        name_counts: dict[str, int] = {}
        for c in data['categories']:
            name_counts[c['name']] = name_counts.get(c['name'], 0) + 1
        for name, count in name_counts.items():
            if count > 1:
                issues.add(
                    'coco_category_duplicate_name', file=str(f.path.name), detail={'name': name}
                )

        images_by_id = {img['id']: img for img in data['images']}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        split = f.split or _infer_split(f.path)

        for image_id, img in images_by_id.items():
            file_name = img['file_name']
            abs_path = f.images_dir / file_name
            if path_guard is not None and not path_guard(abs_path):
                issues.add('image_path_not_servable', file=file_name)
                continue
            width = img.get('width')
            height = img.get('height')
            boxes: list[LabelBox] = []
            has_non_crowd = False
            for ann in anns_by_image.get(image_id, []):
                if ann.get('iscrowd'):
                    issues.add('coco_crowd_skipped', file=file_name)
                    continue
                has_non_crowd = True
                bx, by, bw, bh = ann['bbox']
                cat_name = categories.get(ann['category_id'])
                if cat_name is None or not width or not height:
                    issues.add('coco_bbox_out_of_image', file=file_name)
                    continue
                x1, y1, x2, y2 = bx / width, by / height, (bx + bw) / width, (by + bh) / height
                clamped = _clamp_bbox(x1, y1, x2, y2)
                if clamped is None:
                    issues.add('coco_bbox_out_of_image', file=file_name)
                    continue
                if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                    continue
                boxes.append(
                    LabelBox(
                        dataset_class=cat_name,
                        bbox_norm=(clamped[0], clamped[1], clamped[2], clamped[3]),
                    )
                )

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
                )
            )
    return entries


def scan_coco(
    files: list[CocoAnnotationFile], *, path_guard: PathGuard | None = None
) -> DatasetScan:
    issues = IssueCollector()
    entries = read_coco(files, issues=issues, path_guard=path_guard)
    root = files[0].images_dir if files else Path()
    return DatasetScan(
        format='coco',
        root=root,
        entries=entries,
        issues=issues,
        class_box_counts=class_box_counts(entries),
    )


__all__ = [
    'CocoAnnotationFile',
    'CocoJsonInvalidError',
    'DatasetPathNotAllowedError',
    'read_coco',
    'scan_coco',
]
