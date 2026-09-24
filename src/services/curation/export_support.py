"""Split-assignment, atomic-write and image-pixel helpers shared by the
curation dataset exporters.

Pulled out of ``export.py`` to keep that module under the repo's 700-LOC
file-size ratchet (plan §6 R4). Everything here is imported straight back
into ``export.py`` and re-exported from there, so callers of the
multi-class exporter only ever need ``from
src.services.curation.export import ...``. The single-class / class-subset
exporter (:mod:`src.services.curation.export_single_class`) imports the
same building blocks directly — deliberately, so the two exporters share
one sampler, one splitter, one resize worker and one atomic-write
primitive rather than growing divergent copies.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404 - only used with a fixed argv + resolved executable, see _code_sha
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from src.core.logging import get_logger


if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Sequence

    from src.clients.curation_opensearch import RegistryClassEntry
    from src.config import CurationConfig


logger = get_logger(__name__)

_T = TypeVar('_T')


class SplittableRow(Protocol):
    """Minimum surface :func:`stratified_split` needs from a row.

    Declared structurally rather than as ``_ExportRow`` so the
    single-class exporter's own frame-level row type can be handed to
    the same splitter without either module having to fake the other's
    fields. ``group_key`` is read via ``getattr``, so any additional
    grouping attribute (e.g. ``image_id``) is reachable too.
    """

    item_id: str
    class_id: int
    has_test_crop: bool


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


def even_stratified_sample(
    rows: list[_T], n: int | None, key_fn: Callable[[_T], str], rng: random.Random
) -> list[_T]:
    """Round-robin cap ``rows`` to ~``n`` with EVEN per-stratum representation.

    A *sampler*, not a splitter — the counterpart to :func:`stratified_split`,
    which answers a different question (which split does each kept row land
    in). Each stratum (the ``key_fn`` value, e.g. an item's class id)
    contributes one row per pass, so small strata survive a cap intact and
    large strata are the ones trimmed. Plain ``rows[:n]`` truncation instead
    lets whatever sorts first monopolize the budget and can drop a rare
    stratum entirely.

    Args:
        rows: Candidate rows to sample from.
        n: Target count. ``None`` or ``>= len(rows)`` keeps everything.
        key_fn: Maps a row to its stratum key.
        rng: Seeded RNG, so the same export seed reproduces the same sample.

    Returns:
        At most ``n`` rows, every non-empty stratum represented as long as
        ``n >= `` the number of strata.
    """
    if n is None or n >= len(rows):
        return list(rows)
    if n <= 0:
        return []
    buckets: dict[str, list[_T]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = list(buckets)
    rng.shuffle(order)  # no key-order bias on the final, partial pass
    picked: list[_T] = []
    while len(picked) < n:
        progressed = False
        for key in order:
            bucket = buckets[key]
            if bucket:
                picked.append(bucket.pop())
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:  # pragma: no cover - unreachable: n < len(rows)
            break
    return picked


def stratified_split(
    rows: Sequence[SplittableRow],
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

    def _group_of(row: SplittableRow) -> str:
        if group_key is None:
            return f'item:{row.item_id}'
        value = getattr(row, group_key, None)
        if value in (None, ''):
            return f'item:{row.item_id}'
        return f'{group_key}:{value}'

    groups: dict[str, list[SplittableRow]] = {}
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


async def scroll_hits(
    opensearch: Any,
    *,
    index: str,
    query: dict[str, Any],
    source: list[str],
    # F-26: export _source is ~8 small fields (no bbox docvalues switch —
    # see the audit doc's warning that multi-valued numeric docvalues come
    # back sorted+deduplicated, which would corrupt bbox_norm arrays), so
    # a bigger scroll page is safe and cuts round trips on large exports.
    page_size: int = 2000,
    scroll_ttl: str = '5m',
    cap: int | None = None,
) -> list[dict[str, Any]]:
    """Scroll ``index`` and return every raw hit.

    Shared by both exporters so there is exactly one place that gets the
    scroll-context lifecycle right (``clear_scroll`` in a ``finally``, a
    failed clear logged but never fatal).

    ``cap`` stops early once that many hits are collected — used to
    bounded-sample a pool that is far larger than the export needs
    (e.g. every region-free frame in the index), instead of paying for a
    full scroll and discarding almost all of it.
    """
    body: dict[str, Any] = {'size': page_size, 'query': query, '_source': source}
    resp = await opensearch.search(index=index, body=body, scroll=scroll_ttl)
    scroll_id = resp.get('_scroll_id')
    hits = list((resp.get('hits') or {}).get('hits') or [])
    out: list[dict[str, Any]] = []
    try:
        while hits:
            out.extend(hits)
            if cap is not None and len(out) >= cap:
                break
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll=scroll_ttl)
            scroll_id = resp.get('_scroll_id')
            hits = list((resp.get('hits') or {}).get('hits') or [])
    finally:
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.warning('export_clear_scroll_failed', err=str(exc))
    return out


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


def _crop_pil(img: Any, crop_norm: tuple[float, float, float, float]) -> Any:
    """Crop ``img`` to a normalized ``(x1, y1, x2, y2)`` box.

    Normalized (not pixel) coordinates so the box is scale-independent —
    the same stored ``bbox_norm`` crops correctly whatever resolution the
    source frame happens to be. A degenerate or out-of-bounds box leaves
    the image untouched rather than raising: the caller already validated
    geometry upstream, and an unexpected edge case should cost one
    uncropped training image, not the whole export.
    """
    w, h = img.size
    x1, y1, x2, y2 = crop_norm
    left, top = max(0, int(x1 * w)), max(0, int(y1 * h))
    right, bottom = min(w, int(x2 * w)), min(h, int(y2 * h))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def _copy_or_resize_one(
    src_path: str,
    dest_path: str,
    resize_mode: str | None,
    target_size: int,
    crop_norm: tuple[float, float, float, float] | None = None,
) -> tuple[str, bool, str | None]:
    """Worker-process body for the image copy/resize stage.

    Module-level (not a method / closure) so it is picklable for
    ``ProcessPoolExecutor``. Never raises — a per-image failure is
    reported back as ``(dest_path, False, error)`` so one bad source file
    can't abort the whole export.

    ``crop_norm`` (normalized ``x1,y1,x2,y2``) crops before resizing, for
    the single-class exporter's ``item_crop`` image mode. It forces a
    decode even when ``resize_mode`` is ``None``, since a byte copy
    obviously can't crop.
    """
    try:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        if resize_mode is None and crop_norm is None:
            shutil.copyfile(src_path, dest_path)
            return dest_path, True, None
        from PIL import Image

        with Image.open(src_path) as img:
            rgb = img.convert('RGB')
            if crop_norm is not None:
                rgb = _crop_pil(rgb, crop_norm)
            if resize_mode == 'letterbox':
                rgb = _letterbox_pil(rgb, target_size)
            elif resize_mode == 'aspect':
                rgb.thumbnail((target_size, target_size), Image.LANCZOS)
            elif resize_mode is not None:
                msg = f'unknown resize_mode: {resize_mode!r}'
                raise ValueError(msg)
            rgb.save(dest_path, format='JPEG', quality=90)
        return dest_path, True, None
    except Exception as exc:
        return dest_path, False, str(exc)


def atomic_write_text(path: Path, payload: str) -> None:
    """tmp-write + fsync + rename.

    Never leaves a partially-written file visible to a concurrent reader
    (e.g. a preflight scan reading ``manifest.json`` while an export
    rewrites it).
    """
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def atomic_symlink_flip(symlink_path: Path, target: Path) -> None:
    """Point ``symlink_path`` at ``target`` via a write-then-rename.

    A reader that resolves the link mid-flip always sees either the old
    or the new target, never a missing/half-written symlink. The tmp name
    carries the pid so two processes flipping the same link concurrently
    can't clobber each other's staging entry.
    """
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_symlink = symlink_path.with_name(f'.{symlink_path.name}.tmp.{os.getpid()}')
    if tmp_symlink.exists() or tmp_symlink.is_symlink():
        tmp_symlink.unlink()
    tmp_symlink.symlink_to(target, target_is_directory=True)
    tmp_symlink.replace(symlink_path)


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
    'SplittableRow',
    'atomic_symlink_flip',
    'atomic_write_text',
    'dataset_checksum',
    'even_stratified_sample',
    'hash_split',
    'scroll_hits',
    'stratified_split',
]
