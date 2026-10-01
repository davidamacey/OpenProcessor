"""Requeue regions parked in a terminal failure status back to pending.

The detection worker only picks up items whose ``RegionFields.status`` is
pending (see ``scripts/curation/worker/cascade.py``); once it writes a
terminal failure (``detection_failed``, ``verify_rejected``,
``no_region_box``, ``no_region_visible``) the item is never retried. That is
correct until something upstream changes — bbox sanity thresholds, a
detector engine swap, a tightened verify prompt — at which point the parked
cohort deserves a second pass without re-ingesting the source images.

This module is the one generic tool for that:

- :func:`requeue_breakdown` — counts per box ``detector`` x
  ``rejection_reason`` (W8c: nested fields on the ``region_boxes`` list,
  not the deleted item-level scalars) for the selected cohort (the dry
  run).
- :func:`apply_requeue` — flips the cohort to a pending status through the
  OCC skip-on-conflict bulk writer, so a concurrent worker or human write
  always wins. The prior status is stashed in ``RegionFields.status_legacy``
  (first requeue only, so the original verdict survives repeated passes),
  the rejection reason is cleared, and with ``clear_detection`` every box
  a human hasn't touched (:func:`~src.services.curation.region_boxes.
  is_human_owned` — created OR explicitly accepted/rejected/transcribed
  via the W8a per-box edit routes, W8c M3 fix) is dropped from
  ``region_boxes`` so the cascade starts fresh; the worker's own
  fresh-detection write later replaces whatever machine-sourced boxes
  this leaves behind with its new candidates, keeping any human-owned
  ones (W8c M1 fix).
  For ``target=pending_verification`` (re-verify only), each selected
  item's own non-human ``rejected`` box(es) are first rewritten to
  ``proposed`` (reason cleared) so the worker's Path 1 has something to
  re-verify — an item that ends up with nothing to re-propose (e.g. its
  only box is human-owned, or it has none at all) is left untouched
  instead of moving to ``pending_verification`` with no re-verifiable
  box (W8c M2 fix).

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

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import CurationConfig, RegionStatus, get_curation_config
from src.config.region_fields import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.curation.region_box_embeddings import prune_box_embeddings
from src.services.curation.region_boxes import (
    box_query,
    boxes_write_fields,
    has_any_box_query,
    is_human_owned,
    read_boxes,
)


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


def _box_value_filter(F: RegionFields, box_attr: str, values: tuple[str, ...]) -> dict[str, Any]:
    """W8c: ``_value_filter`` over a per-box attribute (e.g. ``detector``,
    ``rejection_reason``), wrapped in the one nested-query shape every
    ``region_boxes`` reader uses (:func:`~src.services.curation.region_boxes.box_query`).

    A ``nested`` query can never match a parent with ZERO elements in the
    nested list -- there is no element to run the inner query against,
    even a ``must_not exists`` one (real OpenSearch semantics, not a fake
    limitation). So :data:`NONE_BUCKET` ("no value recorded") is widened
    with an explicit "this item has no box at all" branch alongside the
    nested "has a box but this attribute is unset on it" case -- both are
    "no value recorded", just at different granularities.
    """
    nested = box_query(_value_filter(f'{F.boxes}.{box_attr}', values), F)
    if NONE_BUCKET not in values:
        return nested
    no_box = {'bool': {'must_not': [has_any_box_query(F)]}}
    return {'bool': {'should': [nested, no_box], 'minimum_should_match': 1}}


def requeue_query(sel: RequeueSelection, fields: RegionFields | None = None) -> dict[str, Any]:
    """W8c: selects on the ``region_boxes`` list, never the deleted
    single-scalar fields (``bbox_norm``, item-level ``detector`` /
    ``rejection_reason``) — a fresh W8 write never populates those, so the
    pre-W8c query silently matched nothing for any item this worker wrote.
    """
    F = fields or get_region_fields()
    # Every clause is a pure predicate (term/exists/should-of-terms
    # via _value_filter) -- filter context, not must.
    filt: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = [{'term': {F.validated: True}}]
    if sel.status is None:
        must_not.append({'exists': {'field': F.status}})
    else:
        filt.append({'term': {F.status: sel.status.value}})
    if sel.detectors:
        filt.append(_box_value_filter(F, 'detector', sel.detectors))
    if sel.reasons:
        filt.append(_box_value_filter(F, 'rejection_reason', sel.reasons))
    if sel.missing_provenance:
        must_not.append({'exists': {'field': F.detector_chain}})
    if sel.target == RegionStatus.PENDING_VERIFICATION:
        # A box to re-verify: any state, any source -- has_any_box_query
        # covers both accepted and rejected (VERIFY_REJECTED's own box).
        filt.append(has_any_box_query(F))
    return {'bool': {'filter': filt, 'must_not': must_not}}


def detection_fields(fields: RegionFields | None = None) -> tuple[str, ...]:
    """Every item-level field a fresh ``pending_detection`` item would not
    carry yet. The per-box detection, verdict and text state lives on the
    boxes themselves, which :func:`apply_requeue` drops with
    ``clear_detection``."""
    F = fields or get_region_fields()
    return (
        F.reason,
        F.verified,
        F.verified_at,
        F.verifier,
        F.verifier_version,
        F.auto_confirmed,
        F.visible,
        F.detector_chain,
        F.detected_at,
        F.skip_verify,
    )


async def requeue_breakdown(
    opensearch: AsyncOpenSearch,
    sel: RequeueSelection,
    *,
    config: CurationConfig | None = None,
    fields: RegionFields | None = None,
    max_buckets: int = 50,
) -> dict[str, Any]:
    """Count the selected cohort, grouped by detector then rejection reason.

    ``by_detector``/its nested ``reasons`` count BOXES, not items -- a
    nested aggregation can only bucket elements of the ``region_boxes``
    list, so an item with 2+ selected boxes (today only possible for a
    multi-box deployment) is counted once per matching box, and their sum
    can exceed ``total - no_box`` (item count). This is inherent to
    box-level bucketing, not a bug: exact per-bucket item counts would
    need a ``reverse_nested`` sub-aggregation, which isn't worth the
    complexity for a dry-run report -- if an exact item count for one
    detector/reason combination ever matters, query it directly instead
    of assuming these buckets reconcile with ``total``.

    ``total`` (item count, from the outer query) and ``no_box`` (also an
    item count: how many of ``total`` carry no ``region_boxes`` element at
    all) DO reconcile with each other -- ``total - no_box`` is exactly how
    many selected items have at least one box. A nested aggregation has no
    element to bucket a zero-box item under (not even
    :data:`NONE_BUCKET`), so before this field existed a cohort of
    entirely-boxless items (``no_region_box``/``no_region_visible``/
    unseeded) rendered an empty ``by_detector`` breakdown with no
    indication why (W8c nit fix; the pre-nested-query version showed an
    explicit ``(none)`` bucket for this case).
    """
    cfg = config or get_curation_config()
    F = fields or get_region_fields()
    body = {
        'size': 0,
        'track_total_hits': True,
        'query': requeue_query(sel, F),
        'aggs': {
            # W8c: detector/rejection_reason moved onto the per-box
            # `region_boxes` list -- a nested agg over the box path. See
            # this function's docstring for why these buckets count BOXES,
            # not items, and don't reconcile with `total`.
            'boxes': {
                'nested': {'path': F.boxes},
                'aggs': {
                    'by_detector': {
                        'terms': {
                            'field': f'{F.boxes}.detector',
                            'size': max_buckets,
                            'missing': NONE_BUCKET,
                        },
                        'aggs': {
                            'by_reason': {
                                'terms': {
                                    'field': f'{F.boxes}.rejection_reason',
                                    'size': max_buckets,
                                    'missing': NONE_BUCKET,
                                }
                            }
                        },
                    }
                },
            },
            # W8c nit fix: a sibling, non-nested filter agg -- counts
            # ITEMS (not boxes), so `total - no_box` is the exact item
            # count that has at least one box, restoring the pre-port
            # "no box at all" visibility a nested agg alone can't give.
            'has_box': {'filter': has_any_box_query(F)},
        },
    }
    resp = await opensearch.search(index=cfg.items_index, body=body)
    total = int(((resp.get('hits') or {}).get('total') or {}).get('value', 0))
    aggs = resp.get('aggregations') or {}
    detectors = []
    boxes_agg = aggs.get('boxes') or {}
    for det in (boxes_agg.get('by_detector') or {}).get('buckets') or []:
        reasons = [
            {'reason': str(r['key']), 'count': int(r['doc_count'])}
            for r in (det.get('by_reason') or {}).get('buckets') or []
        ]
        detectors.append(
            {'detector': str(det['key']), 'count': int(det['doc_count']), 'reasons': reasons}
        )
    has_box_count = int((aggs.get('has_box') or {}).get('doc_count', 0))
    return {
        'status': sel.status.value if sel.status is not None else NONE_BUCKET,
        'target': sel.target.value,
        'total': total,
        'no_box': max(total - has_box_count, 0),
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
        clear_detection: Null every item-level detection/verify field
            (:func:`detection_fields`) AND drop every non-human-owned box
            from ``region_boxes`` (see :func:`~src.services.curation.
            region_boxes.is_human_owned`) so the cascade starts from
            scratch. Not allowed with a ``pending_verification`` target,
            which needs the existing box.
        config: Supplies ``items_index``.
        fields: RegionFields naming (defaults to the process-wide one).
        page_size: Items per ``search_after`` page / OCC bulk batch.
        max_docs: Stop after requeueing this many (0 = no limit).

    Returns:
        ``{'updated', 'skipped', 'errors'}`` — ``skipped`` counts items a
        concurrent writer changed first (their write wins); an item this
        function decides has nothing to do (e.g. a ``pending_verification``
        target with no re-proposable box, W8c M2) is silently excluded
        from all three counts, same as any other no-op merge result.
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
        box_update: dict[str, Any] = {}
        if sel.target == RegionStatus.PENDING_VERIFICATION:
            # W8c M2 fix: every REQUEUEABLE_STATUSES item's box(es) are
            # always `rejected` (never `proposed` -- if one were,
            # `derive_status` would already report `pending_verification`,
            # not one of these terminal statuses), so the worker's Path 1
            # (re-verify a stored `proposed` box) has nothing to act on
            # unless THIS requeue re-proposes them. Only non-human boxes
            # are re-proposed -- a human's rejected verdict is a verdict,
            # not a failure to re-try (`is_human_owned`, same rule
            # `clear_detection` uses below).
            stored = read_boxes(current, F)
            reproposed = [
                dataclasses.replace(b, state='proposed', rejection_reason=None)
                if b.state == 'rejected' and not is_human_owned(b)
                else b
                for b in stored
            ]
            if not any(b.state == 'proposed' for b in reproposed):
                # Nothing to re-verify (the only box(es) are human-owned,
                # or there is no box at all) -- moving `region_status` to
                # `pending_verification` here would silently fall through
                # to a fresh detection pass instead (the worker's Path 1
                # guard needs a stored `proposed` box). Leave the item
                # untouched rather than write a mismatched state.
                return {}
            box_update = boxes_write_fields(reproposed, current_src=current, F=F)
        update[F.status] = sel.target.value
        if expected is not None:
            # Unseeded items have no prior verdict to stash or reason to clear.
            update[F.rejection_reason] = None
            if current.get(F.status_legacy) is None:
                update[F.status_legacy] = expected
        if clear_detection:
            # W8c (r1 fix, scope corrected by M3): drop every box the
            # human hasn't touched -- created OR explicitly
            # accepted/rejected/transcribed via the W8a per-box edit
            # routes (`is_human_owned`), never a human's -- so the
            # cascade starts fresh. The worker's own fresh-detection
            # write later replaces whatever machine-sourced `region_boxes`
            # this requeue leaves behind with its new candidates, keeping
            # any human-owned ones (`_ItemTask.pending_merge`,
            # `bulk_writer._merge`'s `is_human_owned` branch, W8c M1);
            # wiping the whole list here would re-introduce the exact bug
            # that fix closed, just one write earlier.
            stored = read_boxes(current, F)
            kept = [b for b in stored if is_human_owned(b)]
            if len(kept) != len(stored):
                update.update(boxes_write_fields(kept, current_src=current, F=F))
        update.update(box_update)
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
        if clear_detection:
            await prune_box_embeddings(
                opensearch, index=cfg.items_index, crop_ids=[h['_id'] for h in hits]
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
