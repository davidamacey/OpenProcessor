"""Dataset scanning (W10.2, W10.4): read a directory (YOLO or COCO
format) into an ordered, format-neutral :class:`DatasetScan` — no writes,
no OpenSearch, no Triton. ``POST /datasets/preview`` and the import job
both call :func:`scan_dataset`; the job re-runs the same scan so preview
stays advisory and the job authoritative (W10.4).

Scope note: this pass implements ``yolo`` and ``coco`` detection +
reading. ``openprocessor_export`` detection/reading (``op_export.py``)
is deferred — see the wave report — so :func:`detect_format` raises
``FormatUndetectedError`` for an OP-export-shaped directory rather than
silently misreading it as ``yolo``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.dataset_import.issues import IssueCollector


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


@dataclass
class DatasetScan:
    format: DatasetFormat
    root: Path
    entries: list[ScanEntry]
    issues: IssueCollector
    class_box_counts: dict[str, int] = field(default_factory=dict)
    """dataset_class -> total box count across every entry (only classes
    with >=1 box need a mapping decision, W10.5)."""


def detect_format(root: Path) -> DatasetFormat:
    """First match wins (W10.2.1)."""
    if (root / 'manifest.json').is_file() and (root / 'class_registry.json').is_file():
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


__all__ = [
    'DatasetFormat',
    'DatasetScan',
    'FormatUndetectedError',
    'LabelBox',
    'LabelState',
    'ScanEntry',
    'class_box_counts',
    'detect_format',
    'source_sha',
]
