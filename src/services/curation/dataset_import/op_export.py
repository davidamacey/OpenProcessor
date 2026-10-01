"""Read an OpenProcessor export back as a dataset (W10.2.2).

An export directory (``data.yaml``, ``images/``, ``labels/``,
``manifest.json``, ``class_registry.json``, optionally ``stratum_map.json``
and ``TEST_FROZEN.json``) is the one format whose class identities, frozen
test split and provenance are fully recorded, so the import is a faithful
round trip: classes by NAME from ``class_registry.json``, the frozen test
split kept exactly, backgrounds as reviewed negatives, strata preserved.

An export directory is untrusted input like any other dataset (it may have
been edited, copied or assembled by hand): every file is read through the
size caps and path policy of :mod:`~src.services.curation.dataset_import.paths`
and :mod:`~src.services.curation.dataset_import.limits`, a malformed file is an
issue and never an exception, and a stem from ``stratum_map.json`` is only a
dictionary key, never a path.

Stem resolution against existing project docs (``resolve_stems``) lives in
:mod:`~src.services.curation.dataset_import.op_export_stems`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.services.curation.dataset_import.issues import IssueCollector
from src.services.curation.dataset_import.limits import MAX_MANIFEST_BYTES, preview_max_files
from src.services.curation.dataset_import.paths import resolve_ref
from src.services.curation.dataset_import.scan import (
    DatasetScan,
    LabelBox,
    LabelState,
    ScanEntry,
    class_box_counts,
)
from src.services.curation.dataset_import.yolo import (
    IMAGE_EXTENSIONS,
    _load_data_yaml,
    _names_map,
    read_yolo_labels,
)
from src.services.curation.export_support import label_content_sha


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.curation.dataset_import.paths import PathGuard

StemKind = Literal['frame', 'item_crop']

MANIFEST_NAME = 'manifest.json'
CLASS_REGISTRY_NAME = 'class_registry.json'
STRATUM_MAP_NAME = 'stratum_map.json'
LOCK_NAME = 'TEST_FROZEN.json'
SPLITS = ('train', 'val', 'test')


@dataclass(frozen=True)
class TestFrozenInfo:
    __test__ = False  # not a pytest class

    present: bool
    verified: bool
    test_label_sha: str | None = None
    message: str = ''


@dataclass(frozen=True)
class StratumMapInfo:
    present: bool
    entries: int = 0


@dataclass
class OpExportInfo:
    """What the preview and the import record say about an export."""

    dataset_kind: str
    box_source: str
    image_mode: str
    manifest_dataset_sha: str | None
    recomputed_dataset_sha: str | None
    frozen_test_sha: str | None
    test_frozen: TestFrozenInfo
    stratum_map: StratumMapInfo
    source_classes: dict[str, int]
    """Class name -> the registry id the SOURCE project gave it. Never
    compared with a registry id here; a suggestion only when the name matches
    too (``same_registry``)."""
    source_manifest: dict[str, Any]
    source_class_registry: dict[str, Any]
    test_frozen_content: dict[str, Any] | None
    parent_classes: list[str]
    """Region export: the parent-class filter the export was built with."""
    freeze_test_split_default: bool

    @property
    def is_region_export(self) -> bool:
        return self.box_source == 'region'


# --------------------------------------------------------------------- reading


def is_op_export(root: Path) -> bool:
    """Rule 1 of format detection: ``manifest.json`` + ``class_registry.json``
    + ``data.yaml`` at the root."""
    return root.is_dir() and all(
        (root / name).is_file() for name in (MANIFEST_NAME, CLASS_REGISTRY_NAME, 'data.yaml')
    )


def read_json_object(root: Path, name: str) -> dict[str, Any] | None:
    """A top-level JSON object file under ``root``, or ``None`` when it is
    missing, a symlink, over the manifest cap, unparsable or not an object."""
    path = resolve_ref(root, name)
    if path is None or (root / name).is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def has_symlinks(directory: Path) -> bool:
    """Whether anything under ``directory`` is a symlink (never followed)."""
    try:
        return any(p.is_symlink() for p in directory.rglob('*'))
    except OSError:
        return True


def verify_test_frozen(root: Path) -> TestFrozenInfo:
    """Verify ``TEST_FROZEN.json`` with the bake-off freeze check.

    Wraps :func:`scripts.curation.bakeoff.freeze.verify` so a hostile lock or
    label tree degrades to "not verified": a lock that is a symlink,
    oversize, not an object or unparsable, or a test label tree containing
    symlinks, never raises and never reads outside ``root``.
    """
    from scripts.curation.bakeoff import freeze

    lock_path = root / LOCK_NAME
    if not lock_path.exists() and not lock_path.is_symlink():
        return TestFrozenInfo(False, False, None, f'no {LOCK_NAME}')
    lock = read_json_object(root, LOCK_NAME)
    if lock is None:
        return TestFrozenInfo(True, False, None, f'{LOCK_NAME} is unreadable')
    raw_sha = lock.get('test_label_sha', lock.get('frozen_test_sha'))
    sha = raw_sha if isinstance(raw_sha, str) else None
    test_labels = root / 'labels' / 'test'
    if test_labels.is_dir() and has_symlinks(test_labels):
        return TestFrozenInfo(True, False, sha, 'the test label tree contains symlinks')
    try:
        ok, message = freeze.verify(root)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return TestFrozenInfo(True, False, sha, f'{LOCK_NAME} could not be verified: {exc}'[:200])
    return TestFrozenInfo(True, ok, sha, message[:200])


def _names_by_registry_id(classes: Any) -> dict[int, str] | None:
    if not isinstance(classes, list):
        return None
    out: dict[int, str] = {}
    for c in classes:
        if not isinstance(c, dict) or not isinstance(c.get('class_name'), str):
            return None
        try:
            out[int(c['class_id'])] = c['class_name']
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _invert_export_id_map(
    id_map: Any, name_by_rid: dict[int, str]
) -> tuple[dict[int, str], dict[str, int]] | None:
    """``({dense id: name}, {name: source registry id})`` from
    ``export_id_map`` (``{"<registry id>": dense id}``) over ``classes``."""
    if not isinstance(id_map, dict):
        return None
    dense: dict[int, str] = {}
    source_ids: dict[str, int] = {}
    try:
        for rid_raw, dense_raw in id_map.items():
            rid, dense_id = int(rid_raw), int(dense_raw)
            if dense_id < 0 or dense_id in dense:
                return None
            dense[dense_id] = name_by_rid[rid]
            source_ids[name_by_rid[rid]] = rid
    except (KeyError, TypeError, ValueError):
        return None
    return dense, source_ids


def _dense_names(
    registry: dict[str, Any],
) -> tuple[dict[int, str], dict[str, int], list[str]] | None:
    """``({dense id: name}, {name: source registry id}, parent class names)``
    from ``class_registry.json``, or ``None`` when it is inconsistent.

    Names always come from the registry file's own ``classes``/``names``,
    never from a position in the *importing* project's registry:

    * region export: ``names`` holds the one region class; ``classes`` is the
      parent filter (its ``export_id_map`` indexes parents, not names);
    * single-class / subset item export: ``names`` is the dense order;
    * multi-class export: invert ``export_id_map`` over ``classes``.
    """
    name_by_rid = _names_by_registry_id(registry.get('classes'))
    if name_by_rid is None:
        return None
    raw_names = registry.get('names')
    names: list[str] | None = (
        [str(n) for n in raw_names]
        if isinstance(raw_names, list) and all(isinstance(n, str) for n in raw_names)
        else None
    )
    if registry.get('box_source') == 'region':
        if names is None:
            return None
        return dict(enumerate(names)), {}, list(name_by_rid.values())
    inverted = _invert_export_id_map(registry.get('export_id_map'), name_by_rid)
    if inverted is None or (names is not None and dict(enumerate(names)) != inverted[0]):
        return None
    return inverted[0], inverted[1], []


def _stem_kind(dataset_kind: str, image_mode: str) -> StemKind:
    return 'item_crop' if dataset_kind != 'multi_class' and image_mode == 'item_crop' else 'frame'


def _guarded_images(
    root: Path, split: str, issues: IssueCollector, path_guard: PathGuard | None
) -> dict[str, Path]:
    """``{stem: resolved image path}`` for one split; a symlinked or
    unguarded image is an issue and absent from the result."""
    images: dict[str, Path] = {}
    for p in sorted((root / 'images' / split).glob('*')):
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = f'images/{split}/{p.name}'
        resolved = None if p.is_symlink() or not p.is_file() else resolve_ref(root, rel)
        if resolved is None or (path_guard is not None and not path_guard(resolved)):
            issues.add('image_path_not_servable', file=rel)
            continue
        images.setdefault(p.stem, resolved)
    return images


def _scan_entries(
    root: Path,
    dense: dict[int, str],
    strata: dict[str, str],
    stem_kind: StemKind,
    issues: IssueCollector,
    path_guard: PathGuard | None,
) -> list[ScanEntry]:
    """One entry per image/label stem under ``images/<split>`` and
    ``labels/<split>``; symlinked members are never followed."""
    entries: list[ScanEntry] = []
    for split in SPLITS:
        label_stems = {p.stem for p in (root / 'labels' / split).glob('*.txt')}
        images = _guarded_images(root, split, issues, path_guard)
        for stem in sorted(label_stems | set(images)):
            label_rel = f'labels/{split}/{stem}.txt'
            image = images.get(stem)
            if image is None:
                issues.add('label_file_orphan', file=label_rel)
                continue
            label_path = root / label_rel
            boxes: list[LabelBox] = []
            exists = False
            if label_path.is_symlink():
                issues.add('dataset_path_not_allowed', file=label_rel)
            else:
                boxes, exists = read_yolo_labels(
                    label_path, names=dense, rel_file=label_rel, issues=issues
                )
            state: LabelState = 'labeled' if boxes else 'negative'
            if not exists:
                issues.add('label_file_missing', file=label_rel)
                state = 'unlabeled'
            stratum = strata.get(stem)
            entries.append(
                ScanEntry(
                    rel_path=f'images/{split}/{image.name}',
                    source_stem=stem,
                    abs_image_path=image,
                    split=split,
                    label_state=state,
                    boxes=boxes,
                    stratum=stratum,
                    hard_negative=bool(stratum and stratum.startswith('neg:')),
                    stem_kind=stem_kind,
                )
            )
            if len(entries) > preview_max_files():
                issues.add('dataset_too_large', file=str(root.name))
                return entries
    return entries


def _read_strata(root: Path, issues: IssueCollector) -> tuple[dict[str, str], StratumMapInfo]:
    path = root / STRATUM_MAP_NAME
    if not path.exists() and not path.is_symlink():
        return {}, StratumMapInfo(False)
    raw = read_json_object(root, STRATUM_MAP_NAME)
    if raw is None:
        issues.add('stratum_map_partial', file=STRATUM_MAP_NAME, detail={'reason': 'unreadable'})
        return {}, StratumMapInfo(True, 0)
    strata = {str(k): v for k, v in raw.items() if isinstance(v, str)}
    return strata, StratumMapInfo(True, len(strata))


def _sha_pair(
    root: Path, manifest: dict[str, Any], names: list[str], issues: IssueCollector
) -> tuple[str | None, str | None]:
    """``(manifest sha, recomputed sha)``; a mismatch is the warning
    ``dataset_sha_mismatch`` (the directory was edited after export)."""
    declared = manifest.get('dataset_sha')
    declared = declared if isinstance(declared, str) and declared else None
    labels = root / 'labels'
    if declared is None or not labels.is_dir():
        return declared, None
    if has_symlinks(labels):
        issues.add('dataset_sha_mismatch', file=MANIFEST_NAME, detail={'reason': 'symlinks'})
        return declared, None
    # Multi-class exports record the full 64-hex digest over the dense names;
    # single-class ones a 16-hex digest over the dataset's names.
    if len(declared) == 64:
        recomputed = label_content_sha(root, names, truncate=None)
    else:
        recomputed = label_content_sha(root, names)
    if recomputed != declared:
        issues.add(
            'dataset_sha_mismatch',
            file=MANIFEST_NAME,
            detail={'manifest': declared, 'recomputed': recomputed},
        )
    return declared, recomputed


def read_op_export(root: Path, *, path_guard: PathGuard | None = None) -> DatasetScan:
    """Scan an OpenProcessor export directory into a :class:`DatasetScan`
    (``format='openprocessor_export'``, ``op_export`` populated)."""
    issues = IssueCollector()
    manifest = read_json_object(root, MANIFEST_NAME)
    registry = read_json_object(root, CLASS_REGISTRY_NAME)
    if manifest is None or registry is None:
        issues.add(
            'manifest_unreadable', file=MANIFEST_NAME if manifest is None else CLASS_REGISTRY_NAME
        )
        return DatasetScan(format='openprocessor_export', root=root, entries=[], issues=issues)
    resolved = _dense_names(registry)
    if resolved is None:
        issues.add(
            'manifest_unreadable', file=CLASS_REGISTRY_NAME, detail={'reason': 'inconsistent'}
        )
        return DatasetScan(format='openprocessor_export', root=root, entries=[], issues=issues)
    dense, source_classes, parent_classes = resolved

    yaml_path = root / 'data.yaml'
    data = _load_data_yaml(yaml_path, issues) if yaml_path.is_file() else None
    if data is None:
        if not yaml_path.is_file():
            issues.add('data_yaml_invalid', file='data.yaml', detail={'reason': 'missing'})
        return DatasetScan(format='openprocessor_export', root=root, entries=[], issues=issues)
    try:
        yaml_names = _names_map(data.get('names'))
    except (TypeError, ValueError):
        yaml_names = None
    if yaml_names != dense:
        issues.add(
            'names_mismatch',
            file='data.yaml',
            detail={
                'data_yaml': yaml_names and sorted(yaml_names.items()),
                'registry': sorted(dense.items()),
            },
        )
        return DatasetScan(format='openprocessor_export', root=root, entries=[], issues=issues)

    dataset_kind = str(
        registry.get('dataset_kind') or manifest.get('dataset_kind') or 'multi_class'
    )
    box_source = str(registry.get('box_source') or manifest.get('box_source') or 'item')
    image_mode = str(manifest.get('image_mode') or 'whole_frame')
    stem_kind = _stem_kind(dataset_kind, image_mode)
    issues.add(
        'op_export_detected',
        file=MANIFEST_NAME,
        detail={'dataset_kind': dataset_kind, 'box_source': box_source, 'image_mode': image_mode},
    )

    strata, strata_info = _read_strata(root, issues)
    entries = _scan_entries(root, dense, strata, stem_kind, issues, path_guard)
    stems = {e.source_stem for e in entries}
    missing = sorted(s for s in strata if s not in stems)
    for stem in missing:
        issues.add('stratum_map_partial', file=STRATUM_MAP_NAME, detail={'stem': stem[:200]})

    ordered_names = [dense[i] for i in sorted(dense)]
    declared_sha, recomputed_sha = _sha_pair(root, manifest, ordered_names, issues)

    frozen = verify_test_frozen(root)
    manifest_holdout = manifest.get('frozen_holdout_sha')
    if frozen.present and not frozen.verified:
        issues.add('test_split_changed', file=LOCK_NAME, detail={'message': frozen.message})
        freeze_default = False
    elif frozen.present:
        freeze_default = True
    elif manifest_holdout:
        issues.add('test_frozen_from_manifest', file=MANIFEST_NAME)
        freeze_default = True
    else:
        issues.add('test_frozen_missing', file=MANIFEST_NAME)
        freeze_default = False

    source_manifest = {k: v for k, v in manifest.items() if k != 'image_copy'}
    image_copy = manifest.get('image_copy')
    if isinstance(image_copy, dict):
        source_manifest['image_copy'] = {k: v for k, v in image_copy.items() if k != 'errors'}
    frozen_sha = manifest.get('frozen_test_sha')
    info = OpExportInfo(
        dataset_kind=dataset_kind,
        box_source=box_source,
        image_mode=image_mode,
        manifest_dataset_sha=declared_sha,
        recomputed_dataset_sha=recomputed_sha,
        frozen_test_sha=frozen_sha if isinstance(frozen_sha, str) else None,
        test_frozen=frozen,
        stratum_map=strata_info,
        source_classes=source_classes,
        source_manifest=source_manifest,
        source_class_registry=registry,
        test_frozen_content=read_json_object(root, LOCK_NAME),
        parent_classes=parent_classes,
        freeze_test_split_default=freeze_default,
    )
    return DatasetScan(
        format='openprocessor_export',
        root=root,
        entries=entries,
        issues=issues,
        class_ids={name: i for i, name in dense.items()},
        op_export=info,
        class_box_counts=class_box_counts(entries),
    )


__all__ = [
    'OpExportInfo',
    'StratumMapInfo',
    'TestFrozenInfo',
    'has_symlinks',
    'is_op_export',
    'read_json_object',
    'read_op_export',
    'verify_test_frozen',
]
