"""YOLO-format dataset reading (W10.2.1).

Discovery (``data.yaml`` / ``images/<split>`` tree) is a fresh
implementation here rather than a straight move of
``scripts/curation/yolo_dataset.py`` (which stays in place, still used by
``scripts/curation/eval_regions_vs_gt.py``). The row-level rules (clamp
tolerance, polygon-to-box, class-range checks) are new: the pre-W10
per-image importer had none of them.

A dataset is untrusted input. Every path it names goes through
:mod:`~src.services.curation.dataset_import.paths`, every text file is
size-capped before it is read, and a malformed ``data.yaml`` is a blocking
issue, never an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import (
    MAX_LABEL_FILE_BYTES,
    MAX_YAML_BYTES,
    preview_max_files,
)
from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError, PathGuard
from src.services.curation.dataset_import.safe_yaml import YamlTooComplexError, load_bounded_yaml
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


def _scalar_name(value: Any) -> str:
    if isinstance(value, (list, dict, set)):
        raise ValueError('a class name must be a scalar')
    return str(value)


def _names_map(raw: Any) -> dict[int, str] | None:
    """Parse a YOLO ``data.yaml`` ``names`` field into ``{index: name}``.

    A list is dense (``enumerate``); a dict may be sparse (a class pruned
    from ``data.yaml`` leaves a gap). A missing index must never be filled
    in from iteration order -- that crosses a label at that index onto the
    next *present* name, mislabeling it (the exact bug class this wave
    exists to close). Absent indices are simply not present in the map, so
    a lookup miss is a clean ``label_class_out_of_range``.

    Raises ``ValueError`` for any other shape (a scalar, a dict with a
    non-integer key, a negative index).
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return {i: _scalar_name(n) for i, n in enumerate(raw)}
    if not isinstance(raw, dict):
        raise ValueError('names must be a list or a mapping')
    out = {int(k): _scalar_name(v) for k, v in raw.items()}
    if any(k < 0 for k in out):
        raise ValueError('names has a negative index')
    return out


def _find_yaml(root: Path) -> Path | None:
    if root.is_file():
        return root
    for name in ('data.yaml', 'data.yml', 'dataset.yaml'):
        if (root / name).is_file():
            return root / name
    return None


def _read_text_capped(path: Path, cap: int) -> str | None:
    """The file's text, or ``None`` when it is over ``cap`` bytes or
    unreadable. Never reads more than ``cap + 1`` bytes."""
    try:
        with path.open('rb') as fh:
            raw = fh.read(cap + 1)
    except OSError:
        return None
    if len(raw) > cap:
        return None
    return raw.decode('utf-8', errors='replace')


def _images_under(entry: Path, base: Path) -> list[Path]:
    if entry.is_dir():
        return sorted(
            p for p in entry.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    text = _read_text_capped(entry, MAX_LABEL_FILE_BYTES)
    if text is None:
        return []
    out = []
    for line in text.splitlines():
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


def _load_data_yaml(yaml_path: Path, issues: IssueCollector) -> dict[str, Any] | None:
    """The parsed ``data.yaml`` mapping, or ``None`` after recording why not."""
    name = yaml_path.name
    text = _read_text_capped(yaml_path, MAX_YAML_BYTES)
    if text is None:
        issues.add('data_yaml_invalid', file=name, detail={'reason': 'unreadable or too large'})
        return None
    try:
        data = load_bounded_yaml(text)
    except (yaml.YAMLError, YamlTooComplexError) as exc:
        issues.add(
            'data_yaml_invalid', file=name, detail={'reason': f'not valid YAML: {exc}'[:200]}
        )
        return None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        issues.add('data_yaml_invalid', file=name, detail={'reason': 'top level is not a mapping'})
        return None
    return data


def _nc_matches(nc: Any, names: dict[int, str]) -> bool:
    """``nc`` is the class count: ``len(names)``, or ``max(index) + 1`` for a
    pruned ``data.yaml`` whose ``names`` keeps its original (sparse) indices."""
    try:
        value = int(nc)
    except (TypeError, ValueError):
        return False
    return value in (len(names), max(names) + 1 if names else 0)


def discover_yolo(
    root: Path, issues: IssueCollector, path_guard: PathGuard | None = None
) -> tuple[dict[str, list[Path]], dict[int, str]]:
    """Return ``({split: [image paths]}, {class index: class name})``. Emits
    ``data_yaml_invalid`` / ``data_yaml_names_missing`` / ``split_dir_missing``
    / ``data_yaml_names_sparse`` / ``dataset_path_not_allowed``.

    With a ``path_guard``, every split entry (directory or list file) must
    pass it after symlink resolution; one that does not is recorded as
    ``dataset_path_not_allowed`` and skipped.
    """
    yaml_path = _find_yaml(root)
    if yaml_path is None:
        # images/<split> or <split>/images tree, no data.yaml.
        raise FormatUndetectedError(str(root))

    data = _load_data_yaml(yaml_path, issues)
    if data is None:
        return {}, {}
    try:
        names = _names_map(data.get('names'))
    except (TypeError, ValueError) as exc:
        issues.add(
            'data_yaml_invalid',
            file=yaml_path.name,
            detail={'reason': f'names: {exc}'[:200]},
        )
        return {}, {}
    if names is None:
        issues.add('data_yaml_names_missing', file=yaml_path.name)
        return {}, {}
    nc = data.get('nc')
    if nc is not None and not _nc_matches(nc, names):
        issues.add(
            'data_yaml_invalid',
            file=yaml_path.name,
            detail={'reason': 'nc matches neither len(names) nor max(index) + 1'},
        )
        return {}, names
    if names and (max(names) + 1 != len(names)):
        # A gap in the index range (e.g. a class pruned from data.yaml).
        # Never fill it in from iteration order -- missing indices stay
        # missing and reject at the label-row lookup.
        issues.add(
            'data_yaml_names_sparse',
            file=yaml_path.name,
            detail={'present': sorted(names)},
        )

    root_field = data.get('path') if isinstance(data.get('path'), str) else None
    splits: dict[str, list[Path]] = {}
    for key in _SPLIT_KEYS:
        value = data.get(key)
        if not value:
            continue
        stored_key = 'val' if key == 'valid' else key
        entries = value if isinstance(value, list) else [value]
        images: list[Path] = []
        for entry in entries:
            if not isinstance(entry, str):
                issues.add(
                    'data_yaml_invalid',
                    file=yaml_path.name,
                    detail={'reason': f'{key}: an entry is not a path string'},
                )
                continue
            resolved = _resolve_entry(entry, yaml_path.parent, root_field)
            if resolved is None:
                issues.add('split_dir_missing', file=str(entry))
                continue
            if path_guard is not None and not path_guard(resolved):
                issues.add('dataset_path_not_allowed', file=str(entry))
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
    txt_path: Path, *, names: dict[int, str], rel_file: str, issues: IssueCollector
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
    if not txt_path.is_file():
        return [], False

    boxes: list[LabelBox] = []
    text = _read_text_capped(txt_path, MAX_LABEL_FILE_BYTES)
    if text is None:
        issues.add('dataset_file_too_large', file=rel_file)
        return [], True
    lines = text.splitlines()
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
        if cls_id not in names:
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

    splits, names = discover_yolo(root, issues, path_guard)
    if sum(len(images) for images in splits.values()) > preview_max_files():
        issues.add('dataset_too_large', file=str(root.name))
        return DatasetScan(format='yolo', root=root, entries=[], issues=issues)
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
            boxes: list[LabelBox]
            if path_guard is not None and label_path.exists() and not path_guard(label_path):
                # A label file (or a symlink) outside every allowed root is
                # never opened.
                issues.add('dataset_path_not_allowed', file=rel)
                boxes, exists = [], False
            else:
                boxes, exists = read_yolo_labels(
                    label_path, names=names, rel_file=rel, issues=issues
                )
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
        class_ids={name: idx for idx, name in names.items()},
        class_box_counts=class_box_counts(entries),
    )


__all__ = [
    'DatasetPathNotAllowedError',
    'discover_yolo',
    'label_path_for',
    'read_yolo_labels',
    'scan_yolo',
]
