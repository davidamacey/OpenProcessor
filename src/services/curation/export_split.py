"""Group-aware stratified split shared by the curation dataset exporters.

Split out of :mod:`src.services.curation.export_support` along a real seam:
this module answers "which split does each source image land in", including
the stored splits a dataset import filed frames under (W10.9), and knows
nothing about label files, checksums or pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


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


DEFAULT_SPLIT_GROUP_KEY = 'image_id'
"""The leakage unit :func:`stratified_split` groups on by default.

Items cut from the same source image share pixels, so a group never
straddles train/val/test. ``cluster_id`` is NOT a leakage unit: clustering
assigns class-sized semantic clusters (``cluster_id == class_id`` for every
validated item), so grouping on it put a whole class into one group.
"""

_SPLIT_PRIORITY = ('train', 'val', 'test')
# Below this a ratio counts as zero (``1 - 0.89 - 0.11`` is ~1e-17, not 0).
_RATIO_EPSILON = 1e-9


def _allocate_group_counts(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``n`` groups across the positive-weight splits.

    Every active split (weight > 0) gets one group first, in
    ``train -> val -> test`` priority order, as far as ``n`` reaches; the
    rest go one at a time to whichever split is furthest below its target
    ``n * weight`` (ties to the higher-priority split). Pure arithmetic on
    counts, so it's deterministic and never depends on group identity.
    """
    active = [s for s in _SPLIT_PRIORITY if weights.get(s, 0.0) > _RATIO_EPSILON]
    counts = dict.fromkeys(_SPLIT_PRIORITY, 0)
    if n <= 0 or not active:
        return counts
    total = sum(weights[s] for s in active)
    target = {s: n * weights[s] / total for s in active}
    for split in active[:n]:
        counts[split] = 1
    for _ in range(n - min(n, len(active))):
        best = max(active, key=lambda s: (target[s] - counts[s], -active.index(s)))
        counts[best] += 1
    return counts


SPLITS = ('train', 'val', 'test')

SplitMode = Literal['keep_imported', 'recompute']


@dataclass
class SplitStats:
    """How :func:`stratified_split_with_stats` assigned each group."""

    pinned_groups: int = 0
    computed_groups: int = 0
    pinned_split_conflicts: int = 0
    pinned_overridden_by_holdout: int = 0

    def to_manifest(self, split_mode: str) -> dict[str, Any]:
        return {'split_mode': split_mode, **asdict(self)}


def _most_common_split(pins: Sequence[str]) -> str:
    """Most common split; ties ``test`` > ``val`` > ``train``."""
    counts = {split: pins.count(split) for split in SPLITS}
    return max(reversed(SPLITS), key=lambda sp: counts[sp])


def pins_from_rows(rows: Iterable[Any], split_mode: SplitMode) -> dict[str, str] | None:
    """``{item_id: dataset_split}`` for ``keep_imported``; ``None`` for
    ``recompute`` (stored splits ignored). The one place a row's stored split
    becomes a pin, shared by both exporters."""
    if split_mode == 'recompute':
        return None
    return {r.item_id: r.dataset_split for r in rows if getattr(r, 'dataset_split', None) in SPLITS}


def stratified_split_with_stats(
    rows: Sequence[SplittableRow],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    group_key: str | None = DEFAULT_SPLIT_GROUP_KEY,
    pinned: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], SplitStats]:
    """Per-class, per-group deterministic stratified split.

    **Groups.** Rows sharing a ``group_key`` value (default ``image_id``:
    every item cut from one source image) are always assigned to the same
    split, so near-identical pixels never straddle train/val/test. A row
    with no ``group_key`` value is its own singleton group (keyed by
    ``item_id``). ``group_key=None`` makes every row a singleton.

    **Frozen holdout.** A row with ``has_test_crop=True`` (the frozen
    ``test_holdout`` flag) goes to ``test``, and so does every other row
    of its group (a same-image mate), whatever its class — otherwise the
    mate would leak the holdout's pixels into training.

    **Strata.** Every remaining group belongs to the class stratum of its
    most common ``class_id`` (ties to the smallest id). Within a stratum,
    groups are ordered by ``sha256(seed:stratum:group_id)`` and cut by
    exact counts, not an independent per-item hash bucket, so each class's
    actual ratio tracks the target even when the class is small.

    **Per-class allocation of the remaining ``n`` groups:**

    * a class with at least one frozen holdout row: its test split IS the
      frozen holdout, so the remaining groups are split between train and
      val only, in the ratio ``train_ratio : val_ratio``;
    * a class with no frozen holdout: train / val / test in the ratio
      ``train_ratio : val_ratio : (1 - train_ratio - val_ratio)``.

    Each split with a positive ratio gets one group before any split gets
    a second, in ``train -> val -> test`` priority order; the remainder
    follows the target ratio (:func:`_allocate_group_counts`). So:

    * ``n == 0`` — the class contributes only its holdout rows (to test);
    * ``n == 1`` — train;
    * ``n == 2`` — one train, one val;
    * ``n >= 3`` — at least one train and one val; with no holdout, also
      at least one test.

    **Pinned splits (W10.9).** ``pinned`` maps ``item_id`` to a stored
    split (``train``/``val``/``test``; anything else is ignored), the split
    a dataset import filed the frame under. A group with pinned members
    takes the most common pinned split (ties ``test`` > ``val`` > ``train``;
    more than one distinct value counts in ``pinned_split_conflicts``) and
    is excluded from the per-class allocation. A frozen-holdout group
    still goes to ``test`` whatever its pin says (counted in
    ``pinned_overridden_by_holdout`` when the pin disagreed): the holdout
    always wins. Unpinned groups follow the rules above.

    Deterministic: the same rows (in any order) and seed always give the
    same assignment. Returns ``({item_id: split}, stats)``.
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

    holdout_classes = {row.class_id for row in rows if row.has_test_crop}
    forced_groups: set[str] = set()
    pinned_split: dict[str, str] = {}
    class_to_groups: dict[int, list[str]] = {}
    stats = SplitStats()
    for gid, members in groups.items():
        pins = [pinned[m.item_id] for m in members if pinned and pinned.get(m.item_id) in SPLITS]
        chosen = _most_common_split(pins) if pins else None
        if any(m.has_test_crop for m in members):
            forced_groups.add(gid)
            stats.computed_groups += 1
            if chosen is not None and chosen != 'test':
                stats.pinned_overridden_by_holdout += 1
            continue
        if chosen is not None:
            pinned_split[gid] = chosen
            stats.pinned_groups += 1
            if len(set(pins)) > 1:
                stats.pinned_split_conflicts += 1
            continue
        stats.computed_groups += 1
        counts: dict[int, int] = {}
        for m in members:
            counts[m.class_id] = counts.get(m.class_id, 0) + 1
        best_count = max(counts.values())
        stratum = min(cid for cid, c in counts.items() if c == best_count)
        class_to_groups.setdefault(stratum, []).append(gid)

    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    group_split: dict[str, str] = {**pinned_split, **dict.fromkeys(forced_groups, 'test')}
    for stratum, gids in class_to_groups.items():
        ordered = sorted(
            gids, key=lambda gid: hashlib.sha256(f'{seed}:{stratum}:{gid}'.encode()).hexdigest()
        )
        weights = {
            'train': train_ratio,
            'val': val_ratio,
            'test': 0.0 if stratum in holdout_classes else test_ratio,
        }
        allocation = _allocate_group_counts(len(ordered), weights)
        cursor = 0
        for split in _SPLIT_PRIORITY:
            for gid in ordered[cursor : cursor + allocation[split]]:
                group_split[gid] = split
            cursor += allocation[split]

    item_split: dict[str, str] = {}
    for gid, members in groups.items():
        split = group_split[gid]
        for m in members:
            item_split[m.item_id] = split
    return item_split, stats


def stratified_split(
    rows: Sequence[SplittableRow],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    group_key: str | None = DEFAULT_SPLIT_GROUP_KEY,
    pinned: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """:func:`stratified_split_with_stats` without the stats."""
    return stratified_split_with_stats(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        group_key=group_key,
        pinned=pinned,
    )[0]


__all__ = [
    'DEFAULT_SPLIT_GROUP_KEY',
    'SPLITS',
    'SplitMode',
    'SplitStats',
    'SplittableRow',
    'pins_from_rows',
    'stratified_split',
    'stratified_split_with_stats',
]
