"""Open Images CSV dataset reading.

Layout::

    root/class-descriptions-boxable.csv        (MID,DisplayName)
    root/<split>-annotations-bbox.csv          (one or more)
    root/images/<split>/<ImageID>.jpg          (or <split>/, images/, or the root)

The classes are the ``DisplayName`` texts; a MID is a lookup key into the
descriptions file and never leaves this module. Open Images has no class
index, so ``DatasetScan.class_ids`` stays empty. An ``ImageID`` is a
dictionary key only: an image file is found by listing the image
directories, never by joining the id into a path. Every CSV is size-capped
before it is read; a malformed file is an issue, never an exception.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.services.curation.dataset_import.box_norm import CLAMP_TOLERANCE_NORM, normalize_box
from src.services.curation.dataset_import.capped_io import read_bytes_capped
from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import (
    MAX_LABEL_FILE_BYTES,
    MAX_OPENIMAGES_CSV_BYTES,
    preview_max_files,
)
from src.services.curation.dataset_import.paths import (
    DatasetPathNotAllowedError,
    PathGuard,
    resolve_ref,
)
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    LabelBox,
    ScanEntry,
    class_box_counts,
)
from src.services.curation.dataset_import.yolo import IMAGE_EXTENSIONS


if TYPE_CHECKING:
    from pathlib import Path

_IMAGE_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
_REQUIRED_COLUMNS = ('ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax')
_SPLIT_RANK = {'train': 0, 'val': 1, 'test': 2}
_DESCRIPTIONS_HEADER = ['LabelName', 'DisplayName']


def _csv_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_file())


def is_openimages(root: Path) -> bool:
    return (
        root.is_dir()
        and any(root.glob('*annotations-bbox*.csv'))
        and any(root.glob('*class-descriptions*.csv'))
    )


def _split_of(name: str) -> tuple[str | None, str | None]:
    """``(split, directory word)`` from a boxes file name."""
    lowered = name.lower()
    for word, split in (
        ('train', 'train'),
        ('validation', 'val'),
        ('val', 'val'),
        ('test', 'test'),
    ):
        if word in lowered:
            return split, word
    return None, None


@dataclass
class _OiImage:
    split: str | None
    split_word: str | None
    boxes: list[LabelBox] = field(default_factory=list)
    has_objects: bool = False


@dataclass
class _Ctx:
    root: Path
    classes: dict[str, str]
    issues: IssueCollector
    limit: int
    images: dict[str, _OiImage] = field(default_factory=dict)
    used_mids: dict[str, set[str]] = field(default_factory=dict)
    too_large: bool = False


def _pick_descriptions(root: Path, issues: IssueCollector) -> Path | None:
    files = _csv_files(root, '*class-descriptions-boxable*.csv') or _csv_files(
        root, '*class-descriptions*.csv'
    )
    if len(files) == 1:
        return files[0]
    issues.add(
        'openimages_csv_invalid',
        file=root.name,
        detail={'reason': 'several class-description files', 'files': [f.name for f in files]},
    )
    return None


def _read_classes(path: Path, issues: IssueCollector) -> dict[str, str] | None:
    raw = read_bytes_capped(path, MAX_LABEL_FILE_BYTES)
    if raw is None:
        issues.add('dataset_file_too_large', file=path.name)
        return None
    classes: dict[str, str] = {}
    try:
        reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
        for row in reader:
            if not row or (reader.line_num == 1 and row == _DESCRIPTIONS_HEADER):
                continue
            mid, name = (row[0].strip(), row[1].strip()) if len(row) == 2 else ('', '')
            if not mid or not name:
                issues.add('label_row_malformed', file=path.name, line=reader.line_num)
            elif classes.setdefault(mid, name) != name:
                issues.add(
                    'openimages_csv_invalid',
                    file=path.name,
                    line=reader.line_num,
                    detail={'reason': 'MID listed twice with different names'},
                )
                return None
    except (csv.Error, UnicodeDecodeError) as exc:
        issues.add('openimages_csv_invalid', file=path.name, detail={'error': str(exc)[:200]})
        return None
    return classes


def _box_row(
    row: dict[str | None, str | None],
    line: int,
    file_name: str,
    ctx: _Ctx,
    split: tuple[str | None, str | None],
) -> None:
    """Fold one boxes row into ``ctx``; ``split`` is ``(split, word)``."""
    if None in row or any(row.get(c) is None for c in _REQUIRED_COLUMNS):
        ctx.issues.add('label_row_malformed', file=file_name, line=line)
        return
    image_id = (row['ImageID'] or '').strip()
    mid = (row['LabelName'] or '').strip()
    if not _IMAGE_ID.match(image_id):
        ctx.issues.add('label_row_malformed', file=file_name, line=line)
        return
    try:
        coords = [float(row[c] or '') for c in ('XMin', 'YMin', 'XMax', 'YMax')]
    except ValueError:
        ctx.issues.add('label_row_malformed', file=file_name, line=line)
        return
    image = ctx.images.get(image_id)
    if image is None:
        if len(ctx.images) >= ctx.limit:
            ctx.issues.add('dataset_too_large', file=file_name)
            ctx.too_large = True
            return
        image = ctx.images[image_id] = _OiImage(split=split[0], split_word=split[1])
    elif image.split != split[0]:
        ctx.issues.add(
            'openimages_image_split_conflict',
            file=file_name,
            line=line,
            detail={'image_id': image_id, 'split': image.split},
        )
        return
    name = ctx.classes.get(mid)
    if name is None:
        ctx.issues.add('openimages_class_unknown', file=file_name, line=line, detail={'mid': mid})
        return
    image.has_objects = True
    if (row.get('IsGroupOf') or '').strip() == '1':
        ctx.issues.add('openimages_group_skipped', file=file_name, line=line)
        return
    norm = normalize_box(*coords, tolerance=CLAMP_TOLERANCE_NORM)
    if norm.box is None:
        code = (
            'label_coords_out_of_range'
            if norm.problem == 'out_of_range'
            else 'label_box_degenerate'
        )
        ctx.issues.add(code, file=file_name, line=line)
        return
    if norm.clamped:
        ctx.issues.add('label_coords_clamped', file=file_name, line=line)
    image.boxes.append(LabelBox(dataset_class=name, bbox_norm=norm.box))
    ctx.used_mids.setdefault(name, set()).add(mid)


def _read_boxes_file(path: Path, ctx: _Ctx) -> None:
    if path.stat().st_size > MAX_OPENIMAGES_CSV_BYTES:
        ctx.issues.add('dataset_file_too_large', file=path.name)
        return
    split = _split_of(path.name)
    line = 0
    try:
        with path.open(newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh, strict=True)
            missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                ctx.issues.add(
                    'openimages_csv_invalid', file=path.name, detail={'missing': missing}
                )
                return
            for row in reader:
                line = reader.line_num
                _box_row(row, line, path.name, ctx, split)
                if ctx.too_large:
                    return
    except (csv.Error, UnicodeDecodeError, OSError) as exc:
        ctx.issues.add(
            'openimages_csv_invalid',
            file=path.name,
            line=line or None,
            detail={'error': str(exc)[:200]},
        )


def _boxes_files(root: Path) -> list[Path]:
    files = _csv_files(root, '*annotations-bbox*.csv')
    return sorted(files, key=lambda p: (_SPLIT_RANK.get(_split_of(p.name)[0] or '', 3), p.name))


class _ImageIndex:
    """Stem -> file maps of the candidate image directories, built lazily."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._maps: dict[Path, dict[str, Path]] = {}

    def _map(self, directory: Path) -> dict[str, Path]:
        if directory not in self._maps:
            found: dict[str, Path] = {}
            if directory.is_dir():
                for p in sorted(directory.iterdir()):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        found.setdefault(p.stem, p)
            self._maps[directory] = found
        return self._maps[directory]

    def candidates(self, split_word: str | None) -> list[Path]:
        root = self.root
        out = [root / 'images', root]
        if split_word:
            out = [root / 'images' / split_word, root / split_word, *out]
        return out

    def find(self, image_id: str, split_word: str | None) -> Path | None:
        for directory in self.candidates(split_word):
            hit = self._map(directory).get(image_id)
            if hit is not None:
                return resolve_ref(directory, hit.name)
        return None

    def unreferenced(self, referenced: set[str]) -> list[Path]:
        return [p for m in self._maps.values() for stem, p in m.items() if stem not in referenced]


def _check_duplicate_names(ctx: _Ctx) -> None:
    for name, mids in ctx.used_mids.items():
        if len(mids) > 1:
            ctx.issues.add(
                'openimages_class_duplicate_name',
                file=ctx.root.name,
                detail={'name': name, 'mids': sorted(mids)},
            )


def _entries(ctx: _Ctx, path_guard: PathGuard | None) -> list[ScanEntry]:
    index = _ImageIndex(ctx.root)
    root_resolved = ctx.root.resolve()
    entries: list[ScanEntry] = []
    for image_id, image in ctx.images.items():
        found = index.find(image_id, image.split_word)
        if found is None:
            ctx.issues.add('image_file_missing', file=ctx.root.name, detail={'image_id': image_id})
            continue
        if path_guard is not None and not path_guard(found):
            ctx.issues.add('image_path_not_servable', file=found.name)
            continue
        try:
            rel = str(found.relative_to(root_resolved))
        except ValueError:
            rel = str(found)
        entries.append(
            ScanEntry(
                rel_path=rel,
                source_stem=image_id,
                abs_image_path=found,
                split=image.split,
                label_state='labeled' if image.boxes else 'unlabeled',
                boxes=image.boxes,
            )
        )
    for stray in index.unreferenced(set(ctx.images)):
        ctx.issues.add('openimages_image_unreferenced', file=stray.name)
    return entries


def scan_openimages(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    if path_guard is not None and not path_guard(root):
        raise DatasetPathNotAllowedError(str(root))
    issues = IssueCollector()

    def empty() -> DatasetScan:
        return DatasetScan(format='openimages', root=root, entries=[], issues=issues)

    descriptions = _pick_descriptions(root, issues)
    classes = None if descriptions is None else _read_classes(descriptions, issues)
    if classes is None:
        return empty()
    ctx = _Ctx(root=root, classes=classes, issues=issues, limit=preview_max_files())
    for path in _boxes_files(root):
        _read_boxes_file(path, ctx)
        if ctx.too_large:
            return empty()
    _check_duplicate_names(ctx)
    entries = _entries(ctx, path_guard)
    return DatasetScan(
        format='openimages',
        root=root,
        entries=entries,
        issues=issues,
        class_ids={},
        class_box_counts=class_box_counts(entries),
    )


__all__ = ['is_openimages', 'scan_openimages']
