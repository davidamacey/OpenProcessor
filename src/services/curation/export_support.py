"""Split-assignment and image-pixel helpers for :mod:`src.services.curation.export`.

Pulled out of ``export.py`` to keep that module under the repo's 700-LOC
file-size ratchet (plan §6 R4) — this module has no public surface of its
own; everything here is imported straight back into ``export.py`` and
re-exported from there, so callers only ever need ``from
src.services.curation.export import ...``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404 - only used with a fixed argv + resolved executable, see _code_sha
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.clients.curation_opensearch import RegistryClassEntry
    from src.config import CurationConfig


logger = get_logger(__name__)


@dataclass
class _ExportRow:
    """One label row scrolled off the items index, en route to a label file.

    Structurally satisfies ``src.services.detection.frame_dedup._DedupRow``
    (``image_id`` + ``has_test_crop``) so the same row objects can be
    handed straight to :func:`~src.services.detection.frame_dedup.dedup_rows_by_embedding`.
    """

    item_id: str
    image_id: str
    image_path: str
    bbox_norm: list[float]
    class_id: int
    class_name: str
    # Named to match frame_dedup._DedupRow's protocol attribute directly
    # (rather than a differently-named field + an adapter property) -- this
    # is also this export's frozen ``test_holdout`` flag: a row with
    # has_test_crop=True forces its whole split group to 'test'.
    has_test_crop: bool = False
    cluster_id: int | None = None
    export_class_id: int = -1


def hash_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    """Deterministic ``(seed, key)`` -> ``{'train','val','test'}`` bucket.

    Stable across export re-runs with the same seed. Retained as a
    standalone primitive (used by :func:`stratified_split` for its
    per-stratum ordering key) and for any caller that only needs a single
    unstratified bucket assignment.
    """
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    if frac < train_ratio:
        return 'train'
    if frac < train_ratio + val_ratio:
        return 'val'
    return 'test'


def stratified_split(
    rows: list[_ExportRow],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    group_key: str | None = 'cluster_id',
) -> dict[str, str]:
    """Per-class, per-group deterministic stratified split.

    Rows sharing a ``group_key`` value (e.g. a near-duplicate burst's
    ``cluster_id``) are always assigned to the split together — a group
    can never straddle train/val/test, which is the general data-leakage
    guard the reference exporter's unstratified ``hash_split`` didn't
    have. Rows with no ``group_key`` value fall back to their own
    ``item_id`` as a singleton group.

    Within each class stratum, groups are ordered by
    ``sha256(seed:stratum:group_id)`` (deterministic, so the same seed
    always reproduces the same split) and sliced by exact count
    (``round(n * ratio)``) rather than an independent per-item
    probabilistic hash bucket — this is what makes the *actual* per-class
    ratio converge to the target even for small classes, where
    :func:`hash_split` applied independently per item can drift far from
    the target by chance.

    A row with ``has_test_crop=True`` (the frozen ``test_holdout`` flag)
    forces its entire group to ``test``, taking priority over the
    stratified assignment.

    Returns ``{item_id: split}`` for every row.
    """

    def _group_of(row: _ExportRow) -> str:
        if group_key is None:
            return f'item:{row.item_id}'
        value = getattr(row, group_key, None)
        if value in (None, ''):
            return f'item:{row.item_id}'
        return f'{group_key}:{value}'

    groups: dict[str, list[_ExportRow]] = {}
    for row in rows:
        groups.setdefault(_group_of(row), []).append(row)

    forced_groups: set[str] = set()
    class_to_groups: dict[int, list[str]] = {}
    for gid, members in groups.items():
        if any(m.has_test_crop for m in members):
            forced_groups.add(gid)
            continue
        # Stratum = the group's most common class_id (mode); ties broken
        # toward the smallest class_id for determinism.
        counts: dict[int, int] = {}
        for m in members:
            counts[m.class_id] = counts.get(m.class_id, 0) + 1
        best_count = max(counts.values())
        stratum = min(cid for cid, c in counts.items() if c == best_count)
        class_to_groups.setdefault(stratum, []).append(gid)

    group_split: dict[str, str] = dict.fromkeys(forced_groups, 'test')
    for stratum, gids in class_to_groups.items():
        ordered = sorted(
            gids, key=lambda gid: hashlib.sha256(f'{seed}:{stratum}:{gid}'.encode()).hexdigest()
        )
        n = len(ordered)
        n_train = min(round(n * train_ratio), n)
        n_val = min(round(n * val_ratio), n - n_train)
        for i, gid in enumerate(ordered):
            if i < n_train:
                group_split[gid] = 'train'
            elif i < n_train + n_val:
                group_split[gid] = 'val'
            else:
                group_split[gid] = 'test'

    item_split: dict[str, str] = {}
    for gid, members in groups.items():
        split = group_split[gid]
        for m in members:
            item_split[m.item_id] = split
    return item_split


def dataset_checksum(item_ids: list[str]) -> str:
    """Checksum over the sorted set of item ids that went into an export."""
    return hashlib.sha256('\n'.join(sorted(item_ids)).encode()).hexdigest()


def _build_export_id_map(classes: list[RegistryClassEntry]) -> dict[int, int]:
    """Registry ``class_id`` -> contiguous dense export id (0-indexed).

    Deprecated classes are skipped entirely — they never occupy a dense
    slot. Order is ascending registry ``class_id``, so the mapping only
    changes when the live registry's set of non-deprecated classes
    changes, not on export-to-export scroll-order noise (the previous
    behavior assigned dense ids in first-seen scroll order, which was
    silently non-deterministic).
    """
    live_ids = sorted(c.class_id for c in classes if not c.deprecated)
    return {registry_id: dense_id for dense_id, registry_id in enumerate(live_ids)}


def _remap_rows_to_export_ids(rows: list[_ExportRow], id_map: dict[int, int]) -> list[_ExportRow]:
    """Apply ``id_map`` to every row's ``class_id``, in place.

    Rows whose ``class_id`` has no entry in ``id_map`` (deprecated or
    otherwise not in the live registry) are dropped — there is no valid
    dense id to write into a label file for them.
    """
    kept: list[_ExportRow] = []
    for row in rows:
        dense = id_map.get(row.class_id)
        if dense is None:
            logger.warning(
                'export_row_dropped_unmapped_class',
                item_id=row.item_id,
                class_id=row.class_id,
            )
            continue
        row.export_class_id = dense
        kept.append(row)
    return kept


def _resolve_source_path(image_path: str, config: CurationConfig) -> Path | None:
    """Best-effort resolve an item's stored ``image_path`` to a real file.

    Tries the path as-is (if absolute), then under the configured
    ``source_root``, then under each ``source_path_aliases`` root.
    Returns ``None`` (never raises) when nothing resolves — a missing
    source image should skip that one file's pixel copy, not abort the
    whole export (labels for it were already written).
    """
    if not image_path:
        return None
    candidate = Path(image_path)
    search: list[Path] = [candidate] if candidate.is_absolute() else []
    if not candidate.is_absolute():
        search.append(config.source_root / candidate)
        search.extend(Path(alias) / candidate for alias in config.source_path_aliases.values())
    for path in search:
        if path.is_file():
            return path
    return None


def _letterbox_pil(img: Any, target: int) -> Any:
    from PIL import Image as _Image

    w, h = img.size
    scale = min(target / w, target / h) if w and h else 1.0
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), _Image.LANCZOS)
    canvas = _Image.new('RGB', (target, target), (114, 114, 114))
    canvas.paste(resized, ((target - new_w) // 2, (target - new_h) // 2))
    return canvas


def _copy_or_resize_one(
    src_path: str, dest_path: str, resize_mode: str | None, target_size: int
) -> tuple[str, bool, str | None]:
    """Worker-process body for the image copy/resize stage.

    Module-level (not a method / closure) so it is picklable for
    ``ProcessPoolExecutor``. Never raises — a per-image failure is
    reported back as ``(dest_path, False, error)`` so one bad source file
    can't abort the whole export.
    """
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        if resize_mode is None:
            shutil.copyfile(src_path, dest_path)
            return dest_path, True, None
        from PIL import Image

        with Image.open(src_path) as img:
            rgb = img.convert('RGB')
            if resize_mode == 'letterbox':
                rgb = _letterbox_pil(rgb, target_size)
            elif resize_mode == 'aspect':
                rgb.thumbnail((target_size, target_size), Image.LANCZOS)
            else:
                msg = f'unknown resize_mode: {resize_mode!r}'
                raise ValueError(msg)
            rgb.save(dest_path, format='JPEG', quality=90)
        return dest_path, True, None
    except Exception as exc:
        return dest_path, False, str(exc)


def _code_sha() -> str:
    """Best-effort code provenance for the manifest.

    ``OP_BUILD_SHA`` lets a container image built outside a git checkout
    (no ``.git`` dir) still stamp a real commit sha baked in at build
    time; falls back to a live ``git rev-parse HEAD`` in a dev checkout,
    and to ``'unknown'`` if neither is available.
    """
    override = os.environ.get('OP_BUILD_SHA')
    if override:
        return override
    git_bin = shutil.which('git')
    if not git_bin:
        return 'unknown'
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv, resolved executable, no user input
            [git_bin, 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return 'unknown'


__all__ = [
    'dataset_checksum',
    'hash_split',
    'stratified_split',
]
