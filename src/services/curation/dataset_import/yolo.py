"""YOLO-format dataset reading (W10.2.1).

Discovery (``data.yaml`` / ``images/<split>`` tree) is a fresh
implementation here rather than a straight move of
``scripts/curation/yolo_dataset.py`` (deviation — see the wave report:
``yolo_dataset.py`` stays in place, still used by
``scripts/curation/eval_regions_vs_gt.py``, which this pass does not
touch). The row-level rules (clamp tolerance, polygon-to-box,
class-range checks) are new — the pre-W10 per-image importer (now
deleted, along with the rest of the ``label_import`` module) had none
of them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    FormatUndetectedError,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
)


IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
_SPLIT_KEYS = ('train', 'val', 'valid', 'test')
_CLAMP_TOLERANCE = 1e-3

PathGuard = Callable[[Path], bool]


class DatasetPathNotAllowedError(Exception):
    pass


def _names_list(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(n) for n in raw]
    return [str(raw[k]) for k in sorted(raw, key=int)]


def _find_yaml(root: Path) -> Path | None:
    if root.is_file():
        return root
    for name in ('data.yaml', 'data.yml', 'dataset.yaml'):
        if (root / name).is_file():
            return root / name
    return None


def _images_under(entry: Path, base: Path) -> list[Path]:
    if entry.is_dir():
        return sorted(
            p for p in entry.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    out = []
    for line in entry.read_text(encoding='utf-8').splitlines():
        if line.strip():
            p = Path(line.strip())
            out.append(p if p.is_absolute() else (base / p))
    return sorted(out)


def _resolve_entry(entry: str, yaml_dir: Path, root_field: str | None) -> Path | None:
    candidates: list[Path] = []
    if Path(entry).is_absolute():
        candidates.append(Path(entry))
    else:
        candidates.append(yaml_dir / entry)
        if root_field:
            root = Path(root_field)
            candidates.append((root if root.is_absolute() else yaml_dir / root) / entry)
    for c in candidates:
        if c.exists():
            return c
    return None


def discover_yolo(root: Path, issues: IssueCollector) -> tuple[dict[str, list[Path]], list[str]]:
    """Return ``({split: [image paths]}, class names)``. Emits
    ``data_yaml_invalid`` / ``data_yaml_names_missing`` / ``split_dir_missing``."""
    yaml_path = _find_yaml(root)
    if yaml_path is None:
        # images/<split> or <split>/images tree, no data.yaml.
        raise FormatUndetectedError(str(root))

    data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    names = _names_list(data.get('names'))
    if names is None:
        issues.add('data_yaml_names_missing', file=str(yaml_path.name))
        return {}, []
    nc = data.get('nc')
    if nc is not None and int(nc) != len(names):
        issues.add(
            'data_yaml_invalid', file=str(yaml_path.name), detail={'reason': 'nc != len(names)'}
        )
        return {}, names

    splits: dict[str, list[Path]] = {}
    for key in _SPLIT_KEYS:
        value = data.get(key)
        if not value:
            continue
        stored_key = 'val' if key == 'valid' else key
        entries = value if isinstance(value, list) else [value]
        images: list[Path] = []
        for entry in entries:
            resolved = _resolve_entry(str(entry), yaml_path.parent, data.get('path'))
            if resolved is None:
                issues.add('split_dir_missing', file=str(entry))
                continue
            images.extend(_images_under(resolved, resolved.parent))
        splits.setdefault(stored_key, []).extend(images)
    for key, value_images in splits.items():
        splits[key] = sorted(set(value_images))
    return splits, names


def label_path_for(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == 'images':
            parts[i] = 'labels'
            return Path(*parts).with_suffix('.txt')
    return image.with_suffix('.txt')


def read_yolo_labels(
    txt_path: Path, *, names: list[str], rel_file: str, issues: IssueCollector
) -> tuple[list[LabelBox], bool]:
    """Parse one YOLO ``.txt`` -> ``(boxes, label_file_exists)``.

    Row rules (W10.2.1): 5 fields is a plain box; ``1 + 2k`` fields
    (``k >= 3``) is a YOLO-seg polygon / YOLO-OBB row, imported as its
    axis-aligned bounding box (``yolo_polygon_to_box``, info). Any other
    field count is ``label_row_malformed`` (row skipped). A coordinate
    outside ``[0, 1]`` by <= 1e-3 is clamped (info); further out is
    ``label_coords_out_of_range`` (row skipped). ``cls < 0 or cls >= nc``
    is ``label_class_out_of_range`` (row skipped).
    """
    if not txt_path.exists():
        return [], False

    boxes: list[LabelBox] = []
    lines = txt_path.read_text(encoding='utf-8').splitlines()
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        n = len(parts)
        is_polygon = n >= 7 and n % 2 == 1
        if n != 5 and not is_polygon:
            issues.add('label_row_malformed', file=rel_file, line=lineno)
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            issues.add('label_row_malformed', file=rel_file, line=lineno)
            continue
        if cls_id < 0 or cls_id >= len(names):
            issues.add(
                'label_class_out_of_range', file=rel_file, line=lineno, detail={'cls': cls_id}
            )
            continue
        class_name = names[cls_id]

        if is_polygon:
            xs = coords[0::2]
            ys = coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            issues.add('yolo_polygon_to_box', file=rel_file, line=lineno)
        else:
            cx, cy, bw, bh = coords
            bbox = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

        clamped = False
        out_of_range = False
        for i, v in enumerate(bbox):
            if -_CLAMP_TOLERANCE <= v < 0.0 or 1.0 < v <= 1.0 + _CLAMP_TOLERANCE:
                bbox[i] = max(0.0, min(1.0, v))
                clamped = True
            elif v < -_CLAMP_TOLERANCE or v > 1.0 + _CLAMP_TOLERANCE:
                out_of_range = True
        if out_of_range:
            issues.add('label_coords_out_of_range', file=rel_file, line=lineno)
            continue
        if clamped:
            issues.add('label_coords_clamped', file=rel_file, line=lineno)

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            issues.add('label_box_degenerate', file=rel_file, line=lineno)
            continue

        boxes.append(
            LabelBox(dataset_class=class_name, bbox_norm=(bbox[0], bbox[1], bbox[2], bbox[3]))
        )
    return boxes, True


def scan_yolo(
    root: Path,
    *,
    missing_label: str = 'unlabeled',
    path_guard: PathGuard | None = None,
) -> DatasetScan:
    """Scan a YOLO-format dataset directory into a :class:`DatasetScan`."""
    issues = IssueCollector()
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))

    splits, names = discover_yolo(root, issues)
    entries: list[ScanEntry] = []
    seen_labels: set[Path] = set()

    for split, images in splits.items():
        for image in images:
            if path_guard is not None and not path_guard(image):
                issues.add('image_path_not_servable', file=str(image))
                continue
            label_path = label_path_for(image)
            seen_labels.add(label_path)
            rel = (
                str(image.relative_to(root))
                if root in image.parents or image == root
                else str(image)
            )
            boxes, exists = read_yolo_labels(label_path, names=names, rel_file=rel, issues=issues)
            label_state: LabelState
            if not exists:
                issues.add('label_file_missing', file=rel)
                label_state = 'negative' if missing_label == 'negative' else 'unlabeled'
            elif not boxes:
                label_state = 'negative'
            else:
                label_state = 'labeled'
            entries.append(
                ScanEntry(
                    rel_path=rel,
                    source_stem=image.stem,
                    abs_image_path=image,
                    split=split,
                    label_state=label_state,
                    boxes=boxes,
                )
            )

    # Orphan label files: a labels/ dir entry with no matching image.
    for images in splits.values():
        if not images:
            continue
        labels_root = label_path_for(images[0]).parent
        if not labels_root.is_dir():
            continue
        for txt in sorted(labels_root.glob('*.txt')):
            if txt not in seen_labels:
                issues.add('label_file_orphan', file=str(txt.name))

    return DatasetScan(
        format='yolo',
        root=root,
        entries=entries,
        issues=issues,
        class_box_counts=class_box_counts(entries),
    )


__all__ = [
    'DatasetPathNotAllowedError',
    'discover_yolo',
    'label_path_for',
    'read_yolo_labels',
    'scan_yolo',
]
