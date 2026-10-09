"""The accuracy audit: draw a stratified sample of machine-labelled crops for a
human to label, then report how right the detector and the VLM were.

Nothing here writes a class: drawing a crop only marks it (``audit_*`` fields)
and queues it; the human's label goes through the normal label routes, whose
single class writer stamps the verdict (:func:`audit_outcome_fields`). The
audited crops a human labelled are ordinary human labels, so they are
holdout-eligible like any other.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from src.clients.occ import is_locked_class
from src.clients.occ_bulk import occ_update_bulk
from src.config.curation import items_index
from src.services.curation.audit_math import (
    AUDIT_BATCH,
    AUDIT_HUMAN,
    AUDIT_LABEL_NAME,
    AUDIT_LABEL_SOURCE,
    AUDIT_OUTCOME,
    AUDIT_SAMPLE,
    AUDIT_SAMPLED_AT,
    AuditRow,
    allocate_sample,
    build_report,
)
from src.services.curation.class_sources import VLM_CLASS_SOURCES
from src.services.curation.ingest_class_sources import LABEL_IMPORT_CLASS_SOURCE
from src.services.curation.reprocess_targets import scan_items


DEFAULT_MIN_PER_CLASS = 30
DEFAULT_SAMPLE_SIZE = 300
# Safety valve on the candidate scan, not a sampling cap.
MAX_SCANNED = 500_000


def candidate_query() -> dict[str, Any]:
    """Crops eligible to be drawn: carrying a machine class and the detector's
    answer, not validated, not in the frozen holdout, not excluded or dismissed,
    not an imported suggestion and not already drawn."""
    return {
        'bool': {
            'filter': [
                {'exists': {'field': 'detector_class_name'}},
                {'exists': {'field': 'class_name'}},
            ],
            'must_not': [
                {'term': {'class_validated': True}},
                {'term': {'test_holdout': True}},
                {'term': {'class_excluded': True}},
                {'exists': {'field': 'review_dismissed_at'}},
                {'term': {'class_source': LABEL_IMPORT_CLASS_SOURCE}},
                {'term': {AUDIT_SAMPLE: True}},
            ],
        }
    }


def _rank(crop_id: str) -> str:
    """Stable pseudo-random order of a crop (a shuffle key, not a security hash):
    the same population and parameters always draw the same crops."""
    return hashlib.sha1(f'audit:{crop_id}'.encode(), usedforsecurity=False).hexdigest()


def _marker(batch_id: str, now: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        AUDIT_SAMPLE: True,
        AUDIT_BATCH: batch_id,
        AUDIT_SAMPLED_AT: now,
        # What the crop's machine label was when drawn.
        AUDIT_LABEL_NAME: source.get('class_name'),
        AUDIT_LABEL_SOURCE: source.get('class_source'),
    }


async def start_audit(
    opensearch: Any,
    *,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Draw and mark a sample. ``sample_size`` is the total budget (see
    :func:`allocate_sample`); strata are detector classes. A crop a human
    validated, or that entered the holdout, between the scan and the write is not
    marked. Returns the batch id, what was drawn per stratum and the strata that
    fall short of ``min_per_class``."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.isoformat()
    # Microsecond timestamp keeps ids sortable by time; the random suffix keeps two
    # starts in the same instant apart.
    batch_id = f'audit-{moment:%Y%m%dT%H%M%S%f}-{uuid.uuid4().hex[:6]}'
    found = await scan_items(
        opensearch,
        candidate_query(),
        index=items_index(),
        includes=['crop_id', 'detector_class_name'],
        max_docs=MAX_SCANNED,
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for crop_id, source in found:
        strata[str(source['detector_class_name'])].append(crop_id)
    alloc = allocate_sample(
        {name: len(ids) for name, ids in strata.items()},
        min_per_class=min_per_class,
        total=sample_size,
    )
    chosen = [
        crop_id for name, ids in strata.items() for crop_id in sorted(ids, key=_rank)[: alloc[name]]
    ]

    marked: set[str] = set()

    def merge(doc_id: str, source: dict[str, Any]) -> dict[str, Any]:
        # Re-checked at write time on the freshest source: never mark a crop a
        # human or the holdout claimed since the scan.
        if (
            source.get('class_validated')
            or is_locked_class(source)
            or source.get(AUDIT_SAMPLE)
            or not source.get('class_name')
            or not source.get('detector_class_name')
        ):
            marked.discard(doc_id)
            return {}
        marked.add(doc_id)
        return _marker(batch_id, stamp, source)

    if chosen:
        await occ_update_bulk(opensearch, ids=chosen, merge_fn=merge)
    per_stratum = [
        {
            'detector_class': name,
            'available': len(ids),
            'sampled': len(marked.intersection(sorted(ids, key=_rank)[: alloc[name]])),
            'short_of_floor': alloc[name] < min(len(ids), min_per_class),
        }
        for name, ids in sorted(strata.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    return {
        'batch_id': batch_id,
        'sampled': len(marked),
        'requested': sample_size,
        'min_per_class': min_per_class,
        'strata': per_stratum,
    }


def verdict_query() -> dict[str, Any]:
    """Audited crops a human labelled and that still stand (an undone label is
    no longer validated)."""
    return {
        'bool': {
            'filter': [
                {'term': {AUDIT_SAMPLE: True}},
                {'term': {'class_validated': True}},
                {'exists': {'field': AUDIT_OUTCOME}},
            ]
        }
    }


def pending_query(batch_id: str | None = None) -> dict[str, Any]:
    """Drawn crops still waiting for a human, optionally of one batch."""
    filters: list[dict[str, Any]] = [{'term': {AUDIT_SAMPLE: True}}]
    if batch_id:
        filters.append({'term': {AUDIT_BATCH: batch_id}})
    return {'bool': {'filter': filters, 'must_not': [{'term': {'class_validated': True}}]}}


async def load_report(
    opensearch: Any, *, min_per_class: int = DEFAULT_MIN_PER_CLASS
) -> dict[str, Any]:
    """The audit report over every human verdict so far."""
    found = await scan_items(
        opensearch,
        verdict_query(),
        index=items_index(),
        includes=[
            'detector_class_name',
            AUDIT_LABEL_NAME,
            AUDIT_LABEL_SOURCE,
            AUDIT_HUMAN,
            AUDIT_OUTCOME,
        ],
        max_docs=MAX_SCANNED,
    )
    rows = [
        AuditRow(
            detector=str(src['detector_class_name']),
            label=str(src[AUDIT_LABEL_NAME]),
            label_is_vlm=src.get(AUDIT_LABEL_SOURCE) in VLM_CLASS_SOURCES,
            human=str(src[AUDIT_HUMAN]),
            outcome=src[AUDIT_OUTCOME],
        )
        for _, src in found
        if src.get('detector_class_name') and src.get(AUDIT_LABEL_NAME) and src.get(AUDIT_HUMAN)
    ]
    pending = int(
        (await opensearch.count(index=items_index(), body={'query': pending_query()})).get(
            'count', 0
        )
    )
    return build_report(rows, min_per_class=min_per_class, pending=pending)
