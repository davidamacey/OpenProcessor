"""Export readiness: refuse an export with nothing in it, and tell a
training preflight whether an export still describes the current dataset.

DQ-M9. ``POST /export/yolo`` used to write (and flip ``current`` to) an
empty dataset when no item was exportable, and a training preflight
passed a months-old export built from an items index that had since been
rebuilt.

**Staleness signal: the items index's identity, not item timestamps.**
Every export records the items index it was built from
(:func:`items_index_generation`: the concrete index name, its OpenSearch
``uuid`` and ``creation_date``) in its manifest as ``items_index``.
Preflight compares that with the live index:

* same ``uuid`` → current;
* different ``uuid`` → the index was deleted and recreated (a rebuild or
  a restore) since the export: the export's item ids, splits and lineage
  no longer resolve against the data, so training on it is blocked;
* no recorded stamp (an export made before stamps existed) → fall back to
  the manifest's ``exported_at`` against the live index's
  ``creation_date``;
* the live index can't be read, or the manifest has neither → ``unknown``
  (never a silent pass, never a false block).

A ``uuid`` is regenerated on every index creation and is immune to clock
skew between the exporter and OpenSearch, which a timestamp comparison is
not. Label edits *after* an export deliberately do not count as stale:
exports are snapshots, and retraining on an older snapshot is a supported
workflow (``GET /export/datasets``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from src.core.logging import get_logger
from src.services.curation.dataset_thresholds import (
    MIN_TRAIN_INSTANCES_PER_CLASS,
    MIN_VAL_INSTANCES_PER_CLASS,
)


logger = get_logger(__name__)

MANIFEST_GENERATION_KEY = 'items_index'

Severity = Literal['ok', 'warn', 'block', 'unknown']
CheckResult = tuple[Severity, str, dict[str, Any]]


class NothingToExportError(Exception):
    """No item survived the export's selection; ``str(exc)`` says why."""


async def items_index_generation(opensearch: Any, index: str) -> dict[str, Any] | None:
    """``{index, uuid, created_at}`` of the concrete items index, or
    ``None`` when it can't be read (logged; callers treat it as unknown)."""
    try:
        resp = await opensearch.indices.get_settings(index=index)
    except Exception as exc:
        logger.warning('items_index_generation_unavailable', index=index, error=str(exc))
        return None
    if not isinstance(resp, dict) or len(resp) != 1:
        return None
    ((name, body),) = resp.items()
    settings = ((body or {}).get('settings') or {}).get('index') or {}
    uuid = settings.get('uuid')
    created_ms = settings.get('creation_date')
    if not isinstance(uuid, str) or not uuid:
        return None
    created_at: str | None = None
    if isinstance(created_ms, str | int) and str(created_ms).isdigit():
        created_at = datetime.fromtimestamp(int(created_ms) / 1000, UTC).isoformat()
    return {'index': str(name), 'uuid': uuid, 'created_at': created_at}


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def export_generation_check(
    manifest: dict[str, Any], current: dict[str, Any] | None
) -> CheckResult:
    """``(severity, message, detail)`` for the ``export_generation`` preflight row."""
    recorded = manifest.get(MANIFEST_GENERATION_KEY)
    recorded = recorded if isinstance(recorded, dict) else {}
    exported_at = manifest.get('exported_at') or manifest.get('finished_at')
    detail = {'export': recorded or None, 'exported_at': exported_at, 'current': current}
    if current is None:
        return 'unknown', 'could not read the current items index to compare', detail
    if recorded.get('uuid'):
        if recorded['uuid'] == current['uuid']:
            return 'ok', 'export was built from the current items index', detail
        return (
            'block',
            (
                f'export was built from a previous items index (uuid {recorded["uuid"]}); '
                f'the index was rebuilt on {current.get("created_at") or "an unknown date"} '
                f'(uuid {current["uuid"]}). Re-export before training.'
            ),
            detail,
        )
    exported = _parse_ts(exported_at)
    created = _parse_ts(current.get('created_at'))
    if exported is None or created is None:
        return 'unknown', 'export records no index stamp or export time to compare', detail
    if exported < created:
        return (
            'block',
            (
                f'export was made at {exported.isoformat()}, before the current items index '
                f'was created ({created.isoformat()}). Re-export before training.'
            ),
            detail,
        )
    return 'ok', 'export was made after the current items index was created', detail


def export_size_check(manifest: dict[str, Any]) -> CheckResult:
    """``(severity, message, detail)`` for the ``export_not_empty`` preflight row."""
    if not manifest:
        return 'unknown', 'no readable manifest.json in the export directory', {}
    count = manifest.get('image_count')
    if not isinstance(count, int):
        splits = manifest.get('split_counts') or {}
        values = [v for v in splits.values() if isinstance(v, int)]
        count = sum(values) if values else None
    detail = {'image_count': count}
    if count is None:
        return 'unknown', 'export manifest records no image count', detail
    if count == 0:
        return 'block', 'export contains 0 images — nothing to train on', detail
    return 'ok', f'export contains {count:,} images', detail


_TRAINING_SPLITS = ('train', 'val')


def export_splits_check(manifest: dict[str, Any]) -> CheckResult:
    """``(severity, message, detail)`` for the ``export_splits_nonempty`` row.

    Blocks when the export has no instance in ``train`` or in ``val`` —
    a run needs both, and a non-empty export can still have neither (every
    item in ``test``).
    """
    splits = manifest.get('split_counts')
    if not isinstance(splits, dict) or not all(
        isinstance(splits.get(s), int) for s in _TRAINING_SPLITS
    ):
        return 'unknown', 'export manifest records no train/val split counts', {}
    counts = {s: int(v) for s, v in splits.items() if isinstance(v, int)}
    empty = [s for s in _TRAINING_SPLITS if counts[s] <= 0]
    detail = {'split_counts': counts, 'empty_splits': empty}
    summary = ', '.join(f'{s}={n:,}' for s, n in counts.items())
    if empty:
        names = ' and '.join(repr(s) for s in empty)
        return (
            'block',
            f'export has 0 images in {names} ({summary}); training needs both train and val. '
            'Re-export once there are enough non-holdout validated items.',
            detail,
        )
    return 'ok', f'export has train and val images ({summary})', detail


def export_class_split_check(
    manifest: dict[str, Any], include_classes: list[int] | None
) -> CheckResult:
    """``(severity, message, detail)`` for the ``export_class_split_coverage`` row.

    Judges every class the run trains on — ``include_classes`` (registry
    ids), else every class in the export — against the manifest's
    ``class_split_counts``: fewer than :data:`MIN_TRAIN_INSTANCES_PER_CLASS`
    train or :data:`MIN_VAL_INSTANCES_PER_CLASS` val instances blocks. An
    ``include_classes`` id the export doesn't have is left to the
    ``include_classes_resolvable`` check.
    """
    rows = manifest.get('class_split_counts')
    if not isinstance(rows, list):
        return (
            'unknown',
            'export manifest records no per-class split counts (exported before they were '
            'recorded); re-export to check per-class train/val coverage',
            {},
        )
    wanted = set(include_classes) if include_classes else None
    judged = [
        r for r in rows if isinstance(r, dict) and (wanted is None or r.get('class_id') in wanted)
    ]
    minimums = {'train': MIN_TRAIN_INSTANCES_PER_CLASS, 'val': MIN_VAL_INSTANCES_PER_CLASS}
    gaps: list[dict[str, Any]] = []
    for row in judged:
        missing = [s for s in _TRAINING_SPLITS if int(row.get(s) or 0) < minimums[s]]
        if missing:
            gaps.append(
                {
                    'class_id': row.get('class_id'),
                    'class_name': row.get('class_name'),
                    'train': int(row.get('train') or 0),
                    'val': int(row.get('val') or 0),
                    'test': int(row.get('test') or 0),
                    'missing_splits': missing,
                }
            )
    detail = {
        'classes': gaps,
        'judged_classes': len(judged),
        'min_train_per_class': MIN_TRAIN_INSTANCES_PER_CLASS,
        'min_val_per_class': MIN_VAL_INSTANCES_PER_CLASS,
    }
    if gaps:
        listed = '; '.join(
            f'{g["class_name"]} (class {g["class_id"]}): '
            + ', '.join(f'{s}={g[s]}' for s in g['missing_splits'])
            for g in gaps
        )
        return (
            'block',
            f'{len(gaps)} class(es) this run trains on are below the per-class minimum of '
            f'{MIN_TRAIN_INSTANCES_PER_CLASS} train / {MIN_VAL_INSTANCES_PER_CLASS} val '
            f'instance(s): {listed}. Validate more items for them, or leave them out via '
            'include_classes, then re-export.',
            detail,
        )
    return (
        'ok',
        f'all {len(judged)} class(es) this run trains on have train and val instances',
        detail,
    )


def export_unlabeled_objects_check(manifest: dict[str, Any]) -> CheckResult:
    """``(severity, message, detail)`` for the ``export_unlabeled_objects`` row.

    Warns when exported images also hold objects the export did not label
    (unreviewed, or on a class with no dense id) — the detector learns
    those as background. Never blocks: exporting partially labeled frames
    is the default policy, and ``require_fully_labeled_images`` is the
    opt-out. ``unknown`` for an export that predates these counts.
    """
    unlabeled = manifest.get('unlabeled_items_on_exported_images')
    images_with = manifest.get('images_with_unlabeled_items')
    detail = {
        'image_count': manifest.get('image_count'),
        'unlabeled_items_on_exported_images': unlabeled,
        'images_with_unlabeled_items': images_with,
        'require_fully_labeled_images': manifest.get('require_fully_labeled_images'),
        'images_dropped_not_fully_labeled': manifest.get('images_dropped_not_fully_labeled'),
    }
    if not isinstance(unlabeled, int) or not isinstance(images_with, int):
        return (
            'unknown',
            'export manifest records no unlabeled-object counts (exported before per-image '
            'labels); re-export to check for partially labeled images',
            detail,
        )
    if unlabeled == 0:
        return 'ok', 'every object on every exported image is labeled', detail
    total = detail['image_count']
    of_total = f'/{total:,}' if isinstance(total, int) else ''
    return (
        'warn',
        f'{images_with:,}{of_total} exported image(s) hold {unlabeled:,} unlabeled object(s) '
        '(unreviewed, or on a class this export leaves out). Training learns an unlabeled '
        'object in a training image as background. Review them, or re-export with '
        'require_fully_labeled_images=true to leave those images out.',
        detail,
    )


__all__ = [
    'MANIFEST_GENERATION_KEY',
    'CheckResult',
    'NothingToExportError',
    'Severity',
    'export_class_split_check',
    'export_generation_check',
    'export_size_check',
    'export_splits_check',
    'export_unlabeled_objects_check',
    'items_index_generation',
]
