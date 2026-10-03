"""The ``region`` scope (W10.13): regenerate an item's machine region boxes.

``redetect`` (default) removes every unlocked machine box, keeps locked ones
(human-owned or imported) and queues the item for the detection worker
(``pending_detection``), which re-proposes around the locked boxes.
``reverify`` turns unlocked ``rejected`` boxes back into ``proposed`` and
queues ``pending_verification``. The engine is
:mod:`~src.services.curation.region_requeue`; this module resolves the
targets to it and does the lock accounting. A validated region set is
never touched (:func:`~...reprocess_locks.region_set_locked`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import RegionStatus, get_curation_config
from src.config.region_fields import get_region_fields
from src.services.curation.region_requeue import (
    REQUEUEABLE_STATUSES,
    RequeueSelection,
    apply_requeue,
    apply_requeue_ids,
    requeue_breakdown,
)
from src.services.curation.region_scope import parent_classes_clause
from src.services.curation.reprocess_locks import region_set_locked
from src.services.curation.reprocess_models import BreakdownRow, ReprocessScopeResult
from src.services.curation.reprocess_targets import (
    ReprocessTargetsError,
    has_profile_selector,
    items_by_terms,
    selector_clauses,
)
from src.services.detection.profile_registry import get_active_region_profile


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.reprocess_models import RegionMode, ReprocessFilter


def _target_for(mode: RegionMode) -> RegionStatus:
    return (
        RegionStatus.PENDING_DETECTION if mode == 'redetect' else RegionStatus.PENDING_VERIFICATION
    )


def region_selections(f: ReprocessFilter, mode: RegionMode) -> list[RequeueSelection]:
    """One :class:`RequeueSelection` per source status the filter names.

    Raises :class:`ReprocessTargetsError` for a region filter that names no
    status, ``include_detected`` without a profile selector, or
    ``detected`` with ``reverify`` (accepted boxes have nothing to re-verify).
    """
    if f.include_detected and not has_profile_selector(f):
        raise ReprocessTargetsError(
            'include_detected needs profile_not or profile_revision_below: a detected set is '
            'only re-run when it was produced by another profile or revision'
        )
    statuses: list[RegionStatus | None]
    if f.missing_status:
        if f.region_status:
            raise ReprocessTargetsError('missing_status and region_status are exclusive')
        statuses = [None]
    else:
        try:
            parsed = [RegionStatus(s) for s in f.region_status]
        except ValueError as exc:
            raise ReprocessTargetsError(f'unknown region status: {exc}') from exc
        if not parsed:
            if not has_profile_selector(f):
                raise ReprocessTargetsError(
                    'the region scope needs region_status, missing_status or a profile selector'
                )
            parsed = list(REQUEUEABLE_STATUSES)
        if f.include_detected and RegionStatus.DETECTED not in parsed:
            parsed.append(RegionStatus.DETECTED)
        statuses = list(parsed)
    detected = RegionStatus.DETECTED in statuses
    if detected and mode == 'reverify':
        raise ReprocessTargetsError('a detected set cannot be re-verified; use redetect')
    if detected and not f.include_detected:
        raise ReprocessTargetsError('detected is only selectable with include_detected')
    extra = tuple(selector_clauses(f))
    # An item with no status is only ever seeded when its class is one of the
    # active profile's parent classes, so only those can be queued.
    active = get_active_region_profile()
    in_scope = parent_classes_clause(active.parent_classes) if active is not None else None
    try:
        return [
            RequeueSelection(
                status=status,
                target=_target_for(mode),
                detectors=tuple(f.detector),
                reasons=tuple(f.reason),
                missing_provenance=f.missing_provenance,
                extra=(*extra, in_scope) if status is None and in_scope is not None else extra,
                allow_detected=detected,
            )
            for status in statuses
        ]
    except ValueError as exc:
        raise ReprocessTargetsError(str(exc)) from exc


async def plan_region_filter(
    opensearch: AsyncOpenSearch, f: ReprocessFilter, mode: RegionMode
) -> ReprocessScopeResult:
    cfg = get_curation_config()
    result = ReprocessScopeResult(scope='region')
    rows: dict[tuple[str, str], int] = {}
    for sel in region_selections(f, mode):
        with_locked = await requeue_breakdown(opensearch, sel, config=cfg, include_locked=True)
        unlocked = await requeue_breakdown(opensearch, sel, config=cfg)
        result.selected += int(with_locked['total'])
        result.locked_skipped += int(with_locked['total']) - int(unlocked['total'])
        # Items with no box at all are invisible to the per-box breakdown.
        result.add_count('no_box', int(unlocked['no_box']))
        for det in unlocked['by_detector']:
            for reason in det['reasons']:
                key = (det['detector'], reason['reason'])
                rows[key] = rows.get(key, 0) + int(reason['count'])
    result.breakdown = [
        BreakdownRow(detector=d, reason=r, count=c) for (d, r), c in sorted(rows.items())
    ]
    return result


async def apply_region_filter(
    opensearch: AsyncOpenSearch, f: ReprocessFilter, mode: RegionMode
) -> int:
    queued = 0
    for sel in region_selections(f, mode):
        totals = await apply_requeue(
            opensearch, sel, clear_detection=mode == 'redetect', config=get_curation_config()
        )
        queued += totals['updated']
    return queued


async def region_items(
    opensearch: AsyncOpenSearch, field: str, ids: list[str]
) -> list[tuple[str, dict[str, Any]]]:
    F = get_region_fields()
    return await items_by_terms(
        opensearch,
        field,
        ids,
        index=get_curation_config().items_index,
        includes=['image_id', F.validated, F.status],
    )


def split_locked(docs: list[tuple[str, dict[str, Any]]]) -> tuple[list[str], list[str]]:
    """``(unlocked ids, locked ids)`` by the region-set lock."""
    unlocked = [cid for cid, src in docs if not region_set_locked(src)]
    locked = [cid for cid, src in docs if region_set_locked(src)]
    return unlocked, locked


async def apply_region_ids(opensearch: AsyncOpenSearch, ids: list[str], mode: RegionMode) -> int:
    if not ids:
        return 0
    totals = await apply_requeue_ids(
        opensearch,
        ids,
        target=_target_for(mode),
        clear_detection=mode == 'redetect',
        config=get_curation_config(),
    )
    return totals['updated']


__all__ = [
    'apply_region_filter',
    'apply_region_ids',
    'plan_region_filter',
    'region_items',
    'region_selections',
    'split_locked',
]
