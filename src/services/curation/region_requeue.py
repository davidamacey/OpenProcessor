"""Requeue regions parked in a terminal failure status back to pending.

The detection worker only picks up items whose ``RegionFields.status`` is
pending (see ``scripts/curation/worker/cascade.py``); once it writes a
terminal failure (``detection_failed``, ``verify_rejected``,
``no_region_box``, ``no_region_visible``) the item is never retried. That is
correct until something upstream changes — bbox sanity thresholds, a
detector engine swap, a tightened verify prompt — at which point the parked
cohort deserves a second pass without re-ingesting the source images.

This module is the one generic tool for that:

- :func:`requeue_breakdown` — counts per ``RegionFields.detector`` x
  ``RegionFields.rejection_reason`` for the selected cohort (the dry run).
- :func:`apply_requeue` — flips the cohort to a pending status through the
  OCC skip-on-conflict bulk writer, so a concurrent worker or human write
  always wins. The prior status is stashed in ``RegionFields.status_legacy``
  (first requeue only, so the original verdict survives repeated passes),
  the rejection reason is cleared, and with ``clear_detection`` every
  box/verify/text/embedding field is nulled so the cascade starts fresh.

The same tool also backfills items that carry **no** region status at all
(``RequeueSelection(status=None)``): items ingested before ingest seeded
``pending_detection`` for an active region profile are otherwise invisible
to the worker forever.

Human-validated regions (``RegionFields.validated=true``) are never
selected: a human "no region visible" or a human box is a verdict, not a
failure. ``false_positive`` and ``detected`` are not requeueable statuses.

All field names route through :class:`~src.config.region_fields.RegionFields`
and the index through :class:`~src.config.curation.CurationConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import CurationConfig, RegionStatus, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

REQUEUEABLE_STATUSES: tuple[RegionStatus, ...] = (
    RegionStatus.DETECTION_FAILED,
    RegionStatus.VERIFY_REJECTED,
    RegionStatus.NO_REGION_BOX,
    RegionStatus.NO_REGION_VISIBLE,
)
REQUEUE_TARGETS: tuple[RegionStatus, ...] = (
    RegionStatus.PENDING_DETECTION,
    RegionStatus.PENDING_VERIFICATION,
)
# Bucket label for items that carry no detector / no rejection reason
# (e.g. rows written before provenance existed). Also accepted as a filter
# value to select exactly those rows.
NONE_BUCKET = '(none)'


@dataclass(frozen=True)
class RequeueSelection:
    """Which parked regions to requeue.

    Attributes:
        status: The terminal status to requeue from, or ``None`` to select
            items with no region status at all (the unseeded backfill).
        target: The pending status to requeue to. ``pending_verification``
            re-runs only the verify step on the existing box, so it selects
            only items that still have one.
        detectors: Only these ``RegionFields.detector`` values
            (:data:`NONE_BUCKET` = no detector recorded).
        reasons: Only these ``RegionFields.rejection_reason`` values
            (:data:`NONE_BUCKET` = no reason recorded).
        missing_provenance: Only items without a
            ``RegionFields.detector_chain`` (pre-provenance writes).
    """

    status: RegionStatus | None
    target: RegionStatus = RegionStatus.PENDING_DETECTION
    detectors: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    missing_provenance: bool = False

    def __post_init__(self) -> None:
        if self.status is not None and self.status not in REQUEUEABLE_STATUSES:
            raise ValueError(
                f'{self.status!s} is not requeueable; expected one of '
                f'{[s.value for s in REQUEUEABLE_STATUSES]}'
            )
        if self.target not in REQUEUE_TARGETS:
            raise ValueError(f'{self.target!s} is not a pending status')


def _value_filter(field: str, values: tuple[str, ...]) -> dict[str, Any]:
    named = [v for v in values if v != NONE_BUCKET]
    options: list[dict[str, Any]] = []
    if named:
        options.append({'terms': {field: named}})
    if NONE_BUCKET in values:
        options.append({'bool': {'must_not': [{'exists': {'field': field}}]}})
    return {'bool': {'should': options, 'minimum_should_match': 1}}


def requeue_query(sel: RequeueSelection, fields: RegionFields | None = None) -> dict[str, Any]:
    F = fields or get_region_fields()
    # F-19: every clause is a pure predicate (term/exists/should-of-terms
    # via _value_filter) -- filter context, not must.
    filt: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = [{'term': {F.validated: True}}]
    if sel.status is None:
        must_not.append({'exists': {'field': F.status}})
    else:
        filt.append({'term': {F.status: sel.status.value}})
    if sel.detectors:
        filt.append(_value_filter(F.detector, sel.detectors))
    if sel.reasons:
        filt.append(_value_filter(F.rejection_reason, sel.reasons))
    if sel.missing_provenance:
        must_not.append({'exists': {'field': F.detector_chain}})
    if sel.target == RegionStatus.PENDING_VERIFICATION:
        filt.append({'exists': {'field': F.bbox_norm}})
    return {'bool': {'filter': filt, 'must_not': must_not}}


def detection_fields(fields: RegionFields | None = None) -> tuple[str, ...]:
    """Every field a fresh ``pending_detection`` item would not carry yet."""
    F = fields or get_region_fields()
    return (
        F.bbox_norm,
        F.bbox_frame,
        F.bbox_correct,
        F.score,
        F.confidence,
        F.reason,
        F.source,
        F.verified,
        F.verified_at,
        F.verifier,
        F.verifier_version,
        F.auto_confirmed,
        F.visible,
        F.detector,
        F.detector_version,
        F.detector_chain,
        F.detected_at,
        F.skip_verify,
        F.candidate_bbox_norm,
        F.candidate_score,
        F.candidate_detector,
        F.candidate_detector_version,
        F.candidate_source,
        F.text,
        F.text_raw,
        F.text_confidence,
        F.text_source,
        F.text_engine_version,
        F.text_vlm,
        F.text_ocr,
        F.text_disagreement,
        F.embedding,
        F.cluster_id,
        F.cluster_subid,
        F.cluster_distance,
    )


async def requeue_breakdown(
    opensearch: AsyncOpenSearch,
    sel: RequeueSelection,
    *,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
    max_buckets: int = 50,
) -> dict[str, Any]:
    """Count the selected cohort, grouped by detector then rejection reason."""
    cfg = config or get_curation_config()
    F = fields or get_region_fields()
    body = {
        'size': 0,
        'track_total_hits': True,
        'query': requeue_query(sel, F),
        'aggs': {
            'by_detector': {
                'terms': {'field': F.detector, 'size': max_buckets, 'missing': NONE_BUCKET},
                'aggs': {
                    'by_reason': {
                        'terms': {
                            'field': F.rejection_reason,
                            'size': max_buckets,
                            'missing': NONE_BUCKET,
                        }
                    }
                },
            }
        },
    }
    resp = await opensearch.search(index=cfg.items_index, body=body)
    total = int(((resp.get('hits') or {}).get('total') or {}).get('value', 0))
    detectors = []
    for det in ((resp.get('aggregations') or {}).get('by_detector') or {}).get('buckets') or []:
        reasons = [
            {'reason': str(r['key']), 'count': int(r['doc_count'])}
            for r in (det.get('by_reason') or {}).get('buckets') or []
        ]
        detectors.append(
            {'detector': str(det['key']), 'count': int(det['doc_count']), 'reasons': reasons}
        )
    return {
        'status': sel.status.value if sel.status is not None else NONE_BUCKET,
        'target': sel.target.value,
        'total': total,
        'by_detector': detectors,
    }


async def apply_requeue(
    opensearch: AsyncOpenSearch,
    sel: RequeueSelection,
    *,
    clear_detection: bool = False,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
    page_size: int = 500,
    max_docs: int = 0,
) -> dict[str, int]:
    """Move the selected cohort to ``sel.target``.

    Args:
        opensearch: AsyncOpenSearch client.
        sel: The cohort to requeue.
        clear_detection: Null every detection/verify/text/embedding field
            (:func:`detection_fields`) so the cascade starts from scratch.
            Not allowed with a ``pending_verification`` target, which needs
            the existing box.
        config: Supplies ``items_index``.
        fields: RegionFields naming (defaults to the process-wide one).
        page_size: Items per ``search_after`` page / OCC bulk batch.
        max_docs: Stop after requeueing this many (0 = no limit).

    Returns:
        ``{'updated', 'skipped', 'errors'}`` — ``skipped`` counts items a
        concurrent writer changed first (their write wins).
    """
    if clear_detection and sel.target == RegionStatus.PENDING_VERIFICATION:
        raise ValueError('clear_detection would drop the box pending_verification needs')
    cfg = config or get_curation_config()
    F = fields or get_region_fields()
    to_clear = detection_fields(F) if clear_detection else ()
    now = datetime.now(UTC).isoformat()
    expected = sel.status.value if sel.status is not None else None

    def _merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        if current.get(F.status) != expected or current.get(F.validated) is True:
            return {}
        update: dict[str, Any] = dict.fromkeys(to_clear)
        update[F.status] = sel.target.value
        if expected is not None:
            # Unseeded items have no prior verdict to stash or reason to clear.
            update[F.rejection_reason] = None
            if current.get(F.status_legacy) is None:
                update[F.status_legacy] = expected
        update['updated_at'] = now
        return update

    totals = {'updated': 0, 'skipped': 0, 'errors': 0}
    cursor: list[Any] | None = None
    while not max_docs or totals['updated'] < max_docs:
        size = page_size if not max_docs else min(page_size, max_docs - totals['updated'])
        body: dict[str, Any] = {
            'size': size,
            '_source': False,
            'query': requeue_query(sel, F),
            'sort': [{'crop_id': 'asc'}],
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await opensearch.search(index=cfg.items_index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        cursor = hits[-1].get('sort')
        result = await occ_skip_on_conflict_bulk(
            opensearch,
            doc_ids=[h['_id'] for h in hits],
            merger=_merge,
            index=cfg.items_index,
            refresh=False,
            writer_id='region_requeue',
        )
        totals['updated'] += int(result.get('updated', 0))
        totals['skipped'] += int(result.get('skipped_due_to_conflict', 0))
        totals['errors'] += len(result.get('errors') or [])
        logger.info('region_requeue_page', status=expected, cursor=cursor, **totals)
        if cursor is None or len(hits) < size:
            break

    if totals['updated']:
        await opensearch.indices.refresh(index=cfg.items_index)
    return totals


__all__ = [
    'NONE_BUCKET',
    'REQUEUEABLE_STATUSES',
    'REQUEUE_TARGETS',
    'RequeueSelection',
    'apply_requeue',
    'detection_fields',
    'requeue_breakdown',
    'requeue_query',
]
