"""Helpers shared by the native format readers (Pascal VOC, CVAT, Label Studio,
KITTI, CreateML, LabelMe): image lookup under the path policy, pixel boxes,
declared-versus-decoded size checks, split lists and the scan envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.services.curation.dataset_import import coco
from src.services.curation.dataset_import.box_norm import CLAMP_TOLERANCE_PX, normalize_box
from src.services.curation.dataset_import.capped_io import read_bytes_capped, read_text_capped
from src.services.curation.dataset_import.limits import MAX_LABEL_FILE_BYTES
from src.services.curation.dataset_import.paths import PathGuard, resolve_ref
from src.services.curation.dataset_import.safe_json import loads_json
from src.services.curation.dataset_import.scan import (
    DatasetFormat,
    DatasetScan,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
    exif_transposed_size,
)
from src.services.curation.dataset_import.yolo import IMAGE_EXTENSIONS


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.services.curation.dataset_import.issues import IssueCollector

_SPLIT_ORDER = ('test', 'val', 'train')
_SPLIT_WORDS = {'train': 'train', 'val': 'val', 'valid': 'val', 'validation': 'val', 'test': 'test'}


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def entry_state(has_objects: bool, boxes: Sequence[LabelBox]) -> LabelState:
    """No object at all is a reviewed negative; objects but no kept box is
    unlabeled; otherwise labeled."""
    if not has_objects:
        return 'negative'
    return 'labeled' if boxes else 'unlabeled'


def split_of_dirname(name: str) -> str | None:
    return _SPLIT_WORDS.get(name.lower())


def read_split_lists(directory: Path, issues: IssueCollector) -> dict[str, str]:
    """``{stem: split}`` from ``train.txt`` / ``val.txt`` / ``test.txt`` in
    ``directory``; the first of test, val, train naming a stem owns it."""
    owner: dict[str, str] = {}
    for split in _SPLIT_ORDER:
        path = directory / f'{split}.txt'
        if not path.is_file():
            continue
        text = read_text_capped(path, MAX_LABEL_FILE_BYTES)
        if text is None:
            issues.add('dataset_file_too_large', file=path.name)
            continue
        for line in text.splitlines():
            tokens = line.split()
            if tokens:
                owner.setdefault(tokens[0], split)
    return owner


def read_head(path: Path, size: int = 65536) -> str:
    """The first ``size`` bytes decoded leniently, or ``''`` when unreadable."""
    try:
        with path.open('rb') as fh:
            return fh.read(size).decode('utf-8-sig', errors='replace')
    except OSError:
        return ''


def load_json_file(path: Path, cap: int, rel: str, issues: IssueCollector, code: str) -> Any | None:
    """The parsed JSON of ``path``, or ``None`` after recording ``code``."""
    raw = read_bytes_capped(path, cap)
    if raw is None:
        if path.is_file() and path.stat().st_size > cap:
            issues.add('dataset_file_too_large', file=rel)
        else:
            issues.add(code, file=rel, detail={'reason': 'unreadable'})
        return None
    try:
        return loads_json(raw.decode('utf-8-sig'))
    except (ValueError, UnicodeDecodeError) as exc:
        issues.add(code, file=rel, detail={'reason': str(exc)[:200]})
        return None


def find_image(
    bases: Sequence[Path],
    refs: Sequence[str],
    *,
    rel_file: str,
    issues: IssueCollector,
    path_guard: PathGuard | None,
) -> Path | None:
    """The first existing image among ``refs`` under any of ``bases``. A
    reference that escapes its base (``..``, an absolute path, a symlink out)
    never resolves; none found is ``image_file_missing``."""
    for base in bases:
        for ref in refs:
            candidate = resolve_ref(base, ref)
            if (
                candidate is not None
                and candidate.suffix.lower() in IMAGE_EXTENSIONS
                and candidate.is_file()
            ):
                if path_guard is not None and not path_guard(candidate):
                    issues.add('image_path_not_servable', file=rel_file)
                    return None
                return candidate
    issues.add(
        'image_file_missing', file=rel_file, detail={'filename': (refs[0] if refs else '')[:200]}
    )
    return None


def stem_refs(stem: str) -> list[str]:
    return [f'{stem}{ext}' for ext in sorted(IMAGE_EXTENSIONS)]


@dataclass
class SizeBudget:
    """Declared sizes are checked against the decoded header for the first
    ``PREVIEW_SIZE_CHECKS`` images; the job checks the rest."""

    left: int = field(default_factory=lambda: coco.PREVIEW_SIZE_CHECKS)


def resolve_size(
    declared: tuple[int, int] | None,
    image: Path,
    rel_file: str,
    issues: IssueCollector,
    budget: SizeBudget,
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """``(size used for normalizing, declared size to carry)``, or ``None``
    when the image must be skipped."""
    if declared is not None:
        if budget.left > 0:
            budget.left -= 1
            actual = exif_transposed_size(image)
            if actual is not None and actual != declared:
                issues.add(
                    'image_size_mismatch',
                    file=rel_file,
                    detail={'declared': list(declared), 'actual': list(actual)},
                )
                return None
        return declared, declared
    actual = exif_transposed_size(image)
    if actual is None:
        issues.add('image_unreadable', file=rel_file)
        return None
    return actual, None


def declared_size(width: Any, height: Any) -> tuple[int, int] | None:
    try:
        w, h = int(float(width)), int(float(height))
    except (TypeError, ValueError, OverflowError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def box_from_coords(
    name: str,
    coords: tuple[float, float, float, float],
    *,
    size: tuple[float, float],
    rel_file: str,
    index: int,
    issues: IssueCollector,
    tolerance: float = CLAMP_TOLERANCE_PX,
    line: int | None = None,
) -> LabelBox | None:
    """A :class:`LabelBox` from ``x1, y1, x2, y2`` in the units of ``size``,
    or ``None`` after recording why not."""
    if not name:
        issues.add(
            'label_row_malformed',
            file=rel_file,
            line=line,
            detail={'object': index, 'reason': 'no name'},
        )
        return None
    norm = normalize_box(*coords, width=size[0], height=size[1], tolerance=tolerance)
    if norm.box is None:
        code = (
            'label_coords_out_of_range'
            if norm.problem == 'out_of_range'
            else 'label_box_degenerate'
        )
        issues.add(code, file=rel_file, line=line, detail={'object': index})
        return None
    return LabelBox(dataset_class=name, bbox_norm=norm.box)


def finish_scan(
    fmt: DatasetFormat, root: Path, entries: list[ScanEntry], issues: IssueCollector
) -> DatasetScan:
    return DatasetScan(
        format=fmt,
        root=root,
        entries=entries,
        issues=issues,
        class_ids={},
        class_box_counts=class_box_counts(entries),
    )


def scan_too_large(fmt: DatasetFormat, root: Path, issues: IssueCollector) -> DatasetScan:
    issues.add('dataset_too_large', file=root.name)
    return DatasetScan(format=fmt, root=root, entries=[], issues=issues)
