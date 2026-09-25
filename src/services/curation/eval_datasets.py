"""Evaluation datasets for the model comparison (bake-off): discovery, ids, identity.

Two sources (plan ``generic_model_comparison_plan.md`` section 3.2):

* ``export`` -- every real directory (not a symlink) at depth 1 or 2 under
  ``CurationConfig.export_root`` that holds ``manifest.json``, ``data.yaml``
  and a non-empty ``labels/test/``. Depth 2 covers single-class exports at
  ``<export_root>/<profile_name>/<timestamp>/``. ``is_current`` is true when
  the ``current`` symlink at that level points at it.
* ``external`` -- a frozen third-party set: a directory with
  ``TEST_FROZEN.json`` under ``CurationConfig.bakeoff_eval_root/{curated,public,sample}``.

Ids are ``export:<path relative to export_root>`` and
``external:<group>/<name>``. :func:`resolve_dataset_id` refuses anything that
does not resolve inside its root.

Everything the comparison relies on is computed from the files (class counts,
test-split identity and content hashes), not trusted from manifest counts;
only ``dataset_sha`` / ``exported_at`` / ``unlabeled_items_on_exported_images``
come from the export manifest. Results are cached per directory, keyed by the
test labels' and the manifest's mtimes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.curation.bakeoff.class_map import read_export_id_map, read_names
from scripts.curation.bakeoff.freeze import LOCK_NAME, test_sha as freeze_test_sha
from src.config import get_curation_config
from src.services.curation.export_support import frozen_test_sha_of


_config = get_curation_config()
# Module globals (not re-read per call) so tests can point them at tmp dirs.
EXPORT_ROOT: Path = _config.export_root
EXTERNAL_ROOT: Path = _config.bakeoff_eval_root
EXTERNAL_GROUPS: tuple[str, ...] = ('curated', 'public', 'sample')

DATASET_ID_RE = re.compile(r'^(export|external):[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)?$')
_IMAGE_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'})


class UnknownDatasetError(ValueError):
    """A dataset id that is malformed, escapes its root, or names no eval dataset."""


@dataclass(frozen=True)
class EvalClassCount:
    eval_class_id: int
    name: str
    registry_class_id: int | None
    n_objects: int
    n_images: int


@dataclass(frozen=True)
class EvalDatasetRecord:
    """One eval dataset with everything computed from its files (plan 7.2)."""

    id: str
    source: str  # 'export' | 'external'
    group: str | None
    name: str
    path: Path
    is_current: bool
    dataset_kind: str  # 'multi_class' | 'single_class' | 'external'
    class_names: dict[int, str]
    export_id_map: dict[int, int]
    classes: tuple[EvalClassCount, ...]
    n_images: int
    n_objects: int
    n_background_images: int
    frozen_test_sha: str | None
    test_label_sha: str | None
    sha_source: str  # 'manifest' | 'computed'
    dataset_sha: str | None = None
    exported_at: str | None = None
    unlabeled_items_on_exported_images: int | None = None
    frozen_ok: bool | None = None
    test_stems: frozenset[str] = field(default_factory=frozenset, repr=False)

    @property
    def present_class_ids(self) -> list[int]:
        return [c.eval_class_id for c in self.classes]

    @property
    def dir_name(self) -> str:
        """Per-dataset output dir name in a job: ``:`` and ``/`` become ``__``."""
        return self.id.replace(':', '__').replace('/', '__')

    def to_wire(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'source': self.source,
            'group': self.group,
            'name': self.name,
            'path': str(self.path),
            'is_current': self.is_current,
            'dataset_kind': self.dataset_kind,
            'nc': len(self.class_names),
            'classes': [c.__dict__ for c in self.classes],
            'n_images': self.n_images,
            'n_objects': self.n_objects,
            'n_background_images': self.n_background_images,
            'frozen_test_sha': self.frozen_test_sha,
            'test_label_sha': self.test_label_sha,
            'sha_source': self.sha_source,
            'dataset_sha': self.dataset_sha,
            'exported_at': self.exported_at,
            'unlabeled_items_on_exported_images': self.unlabeled_items_on_exported_images,
            'frozen_ok': self.frozen_ok,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _has_test_labels(d: Path) -> bool:
    labels = d / 'labels' / 'test'
    return labels.is_dir() and any(labels.rglob('*.txt'))


def _is_export_dir(d: Path) -> bool:
    return (
        d.is_dir()
        and not d.is_symlink()
        and (d / 'manifest.json').is_file()
        and (d / 'data.yaml').is_file()
        and _has_test_labels(d)
    )


def _is_current(level: Path, d: Path) -> bool:
    cur = level / 'current'
    return cur.is_symlink() and cur.resolve() == d.resolve()


def _label_files(d: Path) -> list[Path]:
    return sorted((d / 'labels' / 'test').rglob('*.txt'))


def _cache_key(d: Path, lock: Path) -> tuple[Any, ...]:
    """Directory + newest label mtime + manifest/lock mtime.

    The newest label-file mtime is included (not only the dir mtime, which a
    content edit does not change) so an edited test label invalidates the
    cached content hash instead of serving a stale ``test_label_sha``.
    """
    labels = d / 'labels' / 'test'
    files = _label_files(d)
    newest = max((f.stat().st_mtime_ns for f in files), default=0)
    lock_mtime = lock.stat().st_mtime_ns if lock.is_file() else 0
    return (str(d.resolve()), labels.stat().st_mtime_ns, newest, len(files), lock_mtime)


def _count_split(d: Path) -> tuple[Counter[int], Counter[int], int, int, frozenset[str]]:
    """Per-class objects/images, image count, background count, label stems."""
    objects: Counter[int] = Counter()
    images_per_class: Counter[int] = Counter()
    positive_stems: set[str] = set()
    stems: set[str] = set()
    for f in _label_files(d):
        stems.add(f.stem)
        seen: set[int] = set()
        for line in f.read_text(encoding='utf-8').splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            objects[cid] += 1
            seen.add(cid)
        if seen:
            positive_stems.add(f.stem)
        for cid in seen:
            images_per_class[cid] += 1
    images_dir = d / 'images' / 'test'
    image_stems = (
        {p.stem for p in images_dir.rglob('*') if p.suffix.lower() in _IMAGE_SUFFIXES}
        if images_dir.is_dir()
        else set(stems)
    )
    n_images = len(image_stems)
    n_background = n_images - len(image_stems & positive_stems)
    return objects, images_per_class, n_images, n_background, frozenset(stems)


_CACHE: dict[tuple[Any, ...], EvalDatasetRecord] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _build(
    d: Path,
    *,
    dataset_id: str,
    source: str,
    group: str | None,
    is_current: bool,
) -> EvalDatasetRecord:
    lock = d / ('manifest.json' if source == 'export' else LOCK_NAME)
    key = (_cache_key(d, lock), dataset_id, is_current)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    meta = _read_json(lock)
    names = read_names(d / 'data.yaml')
    export_id_map = read_export_id_map(d)
    registry_of = {e: r for r, e in export_id_map.items()}
    objects, images_per_class, n_images, n_background, stems = _count_split(d)
    classes = tuple(
        EvalClassCount(
            eval_class_id=cid,
            name=names.get(cid, str(cid)),
            registry_class_id=registry_of.get(cid),
            n_objects=objects[cid],
            n_images=images_per_class[cid],
        )
        for cid in sorted(objects)
        if objects[cid] > 0
    )

    current_label_sha = freeze_test_sha(d)[0]
    if source == 'export':
        recorded_label_sha = meta.get('test_label_sha')
        frozen_test_sha = meta.get('frozen_test_sha') or frozen_test_sha_of(d)
        kind = 'single_class' if meta.get('dataset_kind') == 'single_class' else 'multi_class'
        frozen_ok = None
    else:
        # freeze.py wrote the content hash under the legacy key before W1.
        recorded_label_sha = meta.get('test_label_sha') or meta.get('frozen_test_sha')
        frozen_test_sha = frozen_test_sha_of(d)
        kind = 'external'
        frozen_ok = bool(recorded_label_sha) and recorded_label_sha == current_label_sha
    record = EvalDatasetRecord(
        id=dataset_id,
        source=source,
        group=group,
        name=d.name,
        path=d,
        is_current=is_current,
        dataset_kind=kind,
        class_names=names,
        export_id_map=export_id_map,
        classes=classes,
        n_images=n_images,
        n_objects=sum(objects.values()),
        n_background_images=n_background,
        frozen_test_sha=frozen_test_sha or None,
        test_label_sha=recorded_label_sha or current_label_sha,
        sha_source='manifest' if recorded_label_sha else 'computed',
        dataset_sha=meta.get('dataset_sha') if source == 'export' else None,
        exported_at=meta.get('exported_at') if source == 'export' else None,
        unlabeled_items_on_exported_images=(
            meta.get('unlabeled_items_on_exported_images') if source == 'export' else None
        ),
        frozen_ok=frozen_ok,
        test_stems=stems,
    )
    _CACHE[key] = record
    return record


def _export_record(root: Path, d: Path, level: Path) -> EvalDatasetRecord:
    rel = d.relative_to(root).as_posix()
    return _build(
        d, dataset_id=f'export:{rel}', source='export', group=None, is_current=_is_current(level, d)
    )


def _iter_export_dirs(root: Path) -> list[tuple[Path, Path]]:
    """``(export_dir, level_dir)`` for real export dirs at depth 1 or 2."""
    out: list[tuple[Path, Path]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        if _is_export_dir(child):
            out.append((child, root))
            continue
        out.extend((g, child) for g in sorted(child.iterdir()) if _is_export_dir(g))
    return out


def _is_external_dir(d: Path, root: Path) -> bool:
    return (
        d.is_dir()
        and (d / LOCK_NAME).is_file()
        and d.resolve().is_relative_to(root.resolve())
        and _has_test_labels(d)
    )


def list_eval_datasets(source: str | None = None) -> list[EvalDatasetRecord]:
    """Every eval dataset; exports newest ``exported_at`` first, then external by id."""
    exports: list[EvalDatasetRecord] = []
    external: list[EvalDatasetRecord] = []
    if source in (None, 'export'):
        root = EXPORT_ROOT
        exports = [_export_record(root, d, level) for d, level in _iter_export_dirs(root)]
        exports.sort(key=lambda r: r.id)
        exports.sort(key=lambda r: r.exported_at or '', reverse=True)
    if source in (None, 'external'):
        root = EXTERNAL_ROOT
        for group in EXTERNAL_GROUPS:
            gdir = root / group
            if not gdir.is_dir():
                continue
            external.extend(
                _build(
                    d,
                    dataset_id=f'external:{group}/{d.name}',
                    source='external',
                    group=group,
                    is_current=False,
                )
                for d in sorted(gdir.iterdir())
                if _is_external_dir(d, root)
            )
        external.sort(key=lambda r: r.id)
    return exports + external


def resolve_dataset_id(dataset_id: str) -> EvalDatasetRecord:
    """The record for an ``export:`` / ``external:`` id; :class:`UnknownDatasetError` otherwise."""
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise UnknownDatasetError(f'invalid dataset id {dataset_id!r}')
    source, rel = dataset_id.split(':', 1)
    root = EXPORT_ROOT if source == 'export' else EXTERNAL_ROOT
    candidate = root / rel
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise UnknownDatasetError(f'dataset id {dataset_id!r} is outside its root')
    if source == 'export':
        if not _is_export_dir(candidate) or any(p.is_symlink() for p in _parents(candidate, root)):
            raise UnknownDatasetError(f'unknown eval dataset {dataset_id!r}')
        return _export_record(root, candidate, candidate.parent)
    group = rel.split('/', 1)[0]
    if '/' not in rel or group not in EXTERNAL_GROUPS or not _is_external_dir(candidate, root):
        raise UnknownDatasetError(f'unknown eval dataset {dataset_id!r}')
    return _build(
        candidate, dataset_id=dataset_id, source='external', group=group, is_current=False
    )


def _parents(d: Path, root: Path) -> list[Path]:
    """``d`` and its ancestors below ``root``."""
    out: list[Path] = []
    while d not in (root, d.parent):
        out.append(d)
        d = d.parent
    return out


def export_id_for_dir(export_dir: str | Path) -> str:
    """The ``export:`` id of a directory under the export root (symlinks resolved).

    Raises :class:`UnknownDatasetError` when it is not under the export root
    at depth 1 or 2.
    """
    root = EXPORT_ROOT.resolve()
    resolved = Path(export_dir).resolve()
    if not resolved.is_relative_to(root):
        raise UnknownDatasetError(f'{export_dir} is not under the export root {EXPORT_ROOT}')
    rel = resolved.relative_to(root).as_posix()
    if not rel or rel == '.' or rel.count('/') > 1:
        raise UnknownDatasetError(f'{export_dir} is not an export directory')
    return f'export:{rel}'


def train_test_overlap(train_export_dir: Path, dataset: EvalDatasetRecord) -> dict[str, Any]:
    """Eval test images whose stem is also in the training export's train/val labels.

    A leakage warning, never a block (plan 3.5). ``fraction`` is over the eval
    split's images.
    """
    train_stems: set[str] = set()
    for split in ('train', 'val'):
        labels = Path(train_export_dir) / 'labels' / split
        if labels.is_dir():
            train_stems.update(p.stem for p in labels.rglob('*.txt'))
    n = len(train_stems & dataset.test_stems)
    return {'n_images': n, 'fraction': (n / dataset.n_images) if dataset.n_images else 0.0}


__all__ = [
    'DATASET_ID_RE',
    'EXPORT_ROOT',
    'EXTERNAL_ROOT',
    'EvalClassCount',
    'EvalDatasetRecord',
    'UnknownDatasetError',
    'clear_cache',
    'export_id_for_dir',
    'list_eval_datasets',
    'resolve_dataset_id',
    'train_test_overlap',
]
