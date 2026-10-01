"""Dataset scanning (W10.2, W10.4): read a directory (YOLO or COCO
format) into an ordered, format-neutral :class:`DatasetScan` — no writes,
no OpenSearch, no Triton. ``POST /datasets/preview`` and the import job
both call :func:`scan_dataset`; the job re-runs the same scan so preview
stays advisory and the job authoritative (W10.4).

``scan_dataset`` is the one entry point: it applies the path policy to the
root, detects the format, and dispatches to the reader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.services.curation.dataset_import.paths import DatasetPathNotAllowedError


if TYPE_CHECKING:
    from src.services.curation.dataset_import.coco import CocoAnnotationFile
    from src.services.curation.dataset_import.issues import IssueCollector
    from src.services.curation.dataset_import.op_export import OpExportInfo
    from src.services.curation.dataset_import.options import DatasetSource
    from src.services.curation.dataset_import.paths import PathGuard


LabelState = Literal['labeled', 'negative', 'unlabeled']
DatasetFormat = Literal['yolo', 'coco', 'openprocessor_export']


class FormatUndetectedError(Exception):
    pass


@dataclass(frozen=True)
class LabelBox:
    dataset_class: str
    bbox_norm: tuple[float, float, float, float]


@dataclass
class ScanEntry:
    """One dataset image, format-neutral. Consumed by
    ``dataset_import/job.py``'s per-image import flow and (W10.19) by
    P4's project-source reader."""

    rel_path: str
    source_stem: str
    abs_image_path: Path
    split: str | None
    label_state: LabelState
    boxes: list[LabelBox] = field(default_factory=list)
    stratum: str | None = None
    hard_negative: bool = False
    """An OpenProcessor-export ``neg:`` frame: a human marked a detector
    region false-positive there (W10.2.2)."""
    declared_size: tuple[int, int] | None = None
    """A COCO image's declared ``(width, height)``; the job compares it with
    the decoded size (``image_size_mismatch``)."""
    stem_kind: Literal['frame', 'item_crop'] | None = None
    """How an OpenProcessor-export stem resolves to existing docs: a
    whole frame (``image_id``) or an item crop (``crop_id``). ``None`` for
    every other source."""


@dataclass
class DatasetScan:
    format: DatasetFormat
    root: Path
    entries: list[ScanEntry]
    issues: IssueCollector
    class_ids: dict[str, int] = field(default_factory=dict)
    """dataset_class -> the dataset's own integer id for it (YOLO index, COCO
    category id, export dense id). Reported only: never compared with a
    registry id."""
    op_export: OpExportInfo | None = None
    class_box_counts: dict[str, int] = field(default_factory=dict)
    """dataset_class -> total box count across every entry (only classes
    with >=1 box need a mapping decision, W10.5)."""


def exif_transposed_size(path: Path) -> tuple[int, int] | None:
    """The ``(width, height)`` the ingest decode would produce (header read
    plus the EXIF orientation swap), or ``None`` when unreadable."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            width, height = img.size
            orientation = img.getexif().get(0x0112, 1)
    except Exception:
        return None
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def detect_format(root: Path) -> DatasetFormat:
    """First match wins (W10.2.1)."""
    from src.services.curation.dataset_import.op_export import is_op_export

    if is_op_export(root):
        return 'openprocessor_export'
    if root.is_file() and root.suffix in ('.yaml', '.yml'):
        return 'yolo'
    for name in ('data.yaml', 'data.yml', 'dataset.yaml'):
        if (root / name).is_file():
            return 'yolo'
    if (root / 'images').is_dir():
        return 'yolo'
    for child in sorted(root.iterdir()) if root.is_dir() else ():
        if child.is_dir() and (child / 'images').is_dir():
            return 'yolo'
    annotations_dir = root / 'annotations'
    if annotations_dir.is_dir() and any(annotations_dir.glob('*.json')):
        return 'coco'
    raise FormatUndetectedError(str(root))


def source_sha(scan: DatasetScan) -> str:
    """sha256 over sorted (rel_path, size) per image, plus the ordered
    dataset class names with their box counts, plus the label content
    (encoded as the boxes themselves, since a full second read of every
    label file's raw bytes is the job's job, not the scan's).

    Simplification note: any_domain_plan.md W10.11 hashes label FILE
    BYTES directly (so an edit that doesn't change parsed boxes, e.g. a
    reordered row, still changes the hash) and separately hashes image
    (path, size). This implementation hashes over the parsed
    ``ScanEntry`` list instead — deterministic and idempotency-preserving
    for this pass's purposes (unchanged input -> unchanged key), but a
    label file edited in a way that reparsess to the same boxes would
    not change this hash where the spec's would. See the wave report.
    """
    hasher = hashlib.sha256()
    for entry in sorted(scan.entries, key=lambda e: e.rel_path):
        try:
            size = entry.abs_image_path.stat().st_size
        except OSError:
            size = -1
        hasher.update(f'{entry.rel_path}|{size}|{entry.split}|{entry.label_state}'.encode())
        for box in entry.boxes:
            hasher.update(
                f'|{box.dataset_class}|{box.bbox_norm[0]:.6f},{box.bbox_norm[1]:.6f},'
                f'{box.bbox_norm[2]:.6f},{box.bbox_norm[3]:.6f}'.encode()
            )
    for name in sorted(scan.class_box_counts):
        hasher.update(f'#{name}={scan.class_box_counts[name]}'.encode())
    return hasher.hexdigest()


def class_box_counts(entries: list[ScanEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        for box in entry.boxes:
            counts[box.dataset_class] = counts.get(box.dataset_class, 0) + 1
    return counts


def _coco_files(source: DatasetSource, root: Path) -> list[CocoAnnotationFile]:
    from src.services.curation.dataset_import.coco import CocoAnnotationFile
    from src.services.curation.dataset_import.paths import resolve_ref

    if source.coco_annotations:
        return [
            CocoAnnotationFile(Path(f.path), Path(f.images_dir), f.split)
            for f in source.coco_annotations
        ]
    images_dir = root / 'images' if (root / 'images').is_dir() else root
    return [
        CocoAnnotationFile(ann, images_dir)
        for ann in sorted((root / 'annotations').glob('*.json'))
        if resolve_ref(root, f'annotations/{ann.name}') is not None
    ]


def scan_dataset(
    source: DatasetSource,
    *,
    path_guard: PathGuard,
    missing_label: str = 'unlabeled',
) -> DatasetScan:
    """Scan ``source`` (any supported format) into a :class:`DatasetScan`.

    Raises :class:`DatasetPathNotAllowedError` when the root (or an explicit
    COCO annotation/images path) fails ``path_guard``, and
    :class:`FormatUndetectedError` when no format rule matches. Nothing is
    written and no image is decoded beyond the COCO size spot-check.
    """
    from src.services.curation.dataset_import.coco import scan_coco
    from src.services.curation.dataset_import.yolo import scan_yolo

    root = Path(source.path)
    if not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    fmt: DatasetFormat = source.format if source.format != 'auto' else _auto_format(source, root)
    if fmt == 'yolo':
        return scan_yolo(root, missing_label=missing_label, path_guard=path_guard)
    if fmt == 'coco':
        files = _coco_files(source, root)
        for f in files:
            if not path_guard(f.path):
                raise DatasetPathNotAllowedError(str(f.path))
        return scan_coco(files, path_guard=path_guard)
    from src.services.curation.dataset_import.op_export import read_op_export

    return read_op_export(root, path_guard=path_guard)


def _auto_format(source: DatasetSource, root: Path) -> DatasetFormat:
    if source.coco_annotations:
        return 'coco'
    return detect_format(root)


__all__ = [
    'DatasetFormat',
    'DatasetScan',
    'FormatUndetectedError',
    'LabelBox',
    'LabelState',
    'ScanEntry',
    'class_box_counts',
    'detect_format',
    'scan_dataset',
    'source_sha',
]
