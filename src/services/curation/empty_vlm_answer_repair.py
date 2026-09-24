"""Undo ``vlm_unmatched`` stamped on items whose VLM class answer was empty.

Before :mod:`src.services.curation.vlm_class_attempt`, every VLM class writer
recorded a reply with no class (an empty / unparseable batch entry, ``null``,
``-1``, an out-of-range index) as ``class_source='vlm_unmatched'`` with an
empty ``vlm_raw_class`` -- replacing a proposal's ingest source or a
classifier label's source, stamping ``label_source='vlm'`` and a
``vlm_confidence``, and dropping the item out of every later VLM pass.

Those writes only touched ``class_source`` / ``label_source`` /
``class_validated`` / ``vlm_confidence`` / ``vlm_raw_*``: ``class_id``,
``class_name``, the class provenance (``class_detector`` /
``class_labeler`` / ...) and ``cluster_id`` are still the pre-write values.
So the pre-write ``class_source`` is recoverable from the doc itself:

1. ``vlm_provenance`` -- the class provenance was written by an earlier VLM
   class write (a ``class_id`` whose labeler is not ingest and equals its
   detector): the item was ``class_source='vlm'``.
2. ``ingest_provenance`` -- the provenance is still ingest's: the class
   source ingest stamps for that detector (the primary's ``_proposal`` /
   ``_model`` / ``_low_conf``, the secondary's ``_model``), with
   ``label_source`` equal to it as ingest writes it.
3. ``class_history`` -- neither applies but the item has a class history:
   the last entry's recorded state.
4. Otherwise the item is reported, never written.

The empty write's ``vlm_confidence`` (a confidence for an answer that did
not exist) and empty ``vlm_raw_class`` / ``vlm_raw_label`` are cleared, and
the attempt is recorded as :mod:`~src.services.curation.vlm_class_attempt`
now does (``vlm_class_attempted_at`` = the item's ``updated_at``,
``vlm_class_empty_reason='no_answer'``).

Applying re-checks each item under OCC (still ``vlm_unmatched`` with an empty
raw class, not human-owned / validated) and records a restorable
``class_id_history`` snapshot of the state it replaces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.services.curation.class_sources import VLM_UNMATCHED_CLASS_SOURCE
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.ingest_class_sources import VLM_CLASS_SOURCE
from src.services.curation.item_doc import INGEST_CLASS_LABELER
from src.services.curation.vlm_class_attempt import (
    EmptyClassReason,
    class_attempt_fields,
    with_class_snapshot,
)


REPAIR_WRITER = 'operator:repair_empty_vlm_answer'

SOURCE_VLM = 'vlm_provenance'
SOURCE_INGEST = 'ingest_provenance'
SOURCE_HISTORY = 'class_history'
SOURCE_NONE = 'unresolved'

_HISTORY_FIELDS = ('class_id', 'class_name', 'class_source', 'label_source', 'confidence')


@dataclass
class RepairPlan:
    crop_id: str
    source: str
    applicable: bool
    current: dict[str, Any]
    restore: dict[str, Any]
    note: str = ''


def _empty_raw(doc: dict[str, Any]) -> bool:
    return not str(doc.get('vlm_raw_class') or '').strip()


def is_candidate(doc: dict[str, Any]) -> bool:
    return (
        doc.get('class_source') == VLM_UNMATCHED_CLASS_SOURCE
        and _empty_raw(doc)
        and not class_write_locked(doc)
    )


def candidate_query(*, id_prefix: str | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {
        'bool': {
            'filter': [{'term': {'class_source': VLM_UNMATCHED_CLASS_SOURCE}}],
            'should': [
                {'term': {'vlm_raw_class': ''}},
                {'bool': {'must_not': [{'exists': {'field': 'vlm_raw_class'}}]}},
            ],
            'minimum_should_match': 1,
        }
    }
    if id_prefix:
        query['bool']['filter'].append({'wildcard': {'crop_id': f'{id_prefix}*'}})
    return query


def ingest_class_source(doc: dict[str, Any]) -> str | None:
    """The class source ingest stamps for this doc's detector, or ``None``
    when the detector isn't a configured ingest detector (or the doc's
    class state is inconsistent with what that detector writes)."""
    detector = doc.get('class_detector')
    has_class = doc.get('class_id') is not None
    primary = ingest_primary_profile()
    if detector in {primary.detector_model, primary.name}:
        if not primary.assigns_class:
            return None if has_class else f'{primary.name}_proposal'
        return f'{primary.name}_model' if has_class else f'{primary.name}_low_conf'
    secondary = ingest_secondary_profile()
    if secondary is not None and detector in {secondary.detector_model, secondary.name}:
        return f'{secondary.name}_model' if has_class else None
    return None


def _cleanup(doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {'vlm_confidence': None}
    for field in ('vlm_raw_class', 'vlm_raw_label'):
        if field in doc and not str(doc.get(field) or '').strip():
            out[field] = None
    at = doc.get('updated_at') or doc.get('class_labeled_at')
    if at:
        out.update(class_attempt_fields(str(at), EmptyClassReason.NO_ANSWER))
    return out


def plan_repair(crop_id: str, doc: dict[str, Any]) -> RepairPlan | None:
    """The repair plan for one doc, or ``None`` if it isn't a candidate."""
    if not is_candidate(doc):
        return None
    current = {
        f: doc.get(f)
        for f in ('class_id', 'class_name', 'class_source', 'label_source', 'class_detector')
    }
    labeler = doc.get('class_labeler')
    if (
        doc.get('class_id') is not None
        and labeler
        and labeler != INGEST_CLASS_LABELER
        and labeler == doc.get('class_detector')
    ):
        restore: dict[str, Any] = {
            'class_source': VLM_CLASS_SOURCE,
            'label_source': VLM_CLASS_SOURCE,
        }
        return RepairPlan(crop_id, SOURCE_VLM, True, current, {**restore, **_cleanup(doc)})
    if labeler == INGEST_CLASS_LABELER:
        ingest_source = ingest_class_source(doc)
        if ingest_source is not None:
            restore = {'class_source': ingest_source, 'label_source': ingest_source}
            return RepairPlan(crop_id, SOURCE_INGEST, True, current, {**restore, **_cleanup(doc)})
    history = [e for e in doc.get('class_id_history') or [] if isinstance(e, dict)]
    if history and history[-1].get('class_source'):
        entry = history[-1]
        restore = {f: entry.get(f) for f in _HISTORY_FIELDS if f in entry}
        return RepairPlan(
            crop_id,
            SOURCE_HISTORY,
            True,
            current,
            {**restore, **_cleanup(doc)},
            note=f'state before {entry.get("writer")} at {entry.get("at")}',
        )
    return RepairPlan(
        crop_id,
        SOURCE_NONE,
        False,
        current,
        {},
        note=f'no VLM/ingest provenance (detector={doc.get("class_detector")!r}, '
        f'labeler={labeler!r}) and no class history',
    )


async def plan_repairs(
    client: Any, *, index: str, id_prefix: str | None = None, page_size: int = 500
) -> list[RepairPlan]:
    """Scan ``index`` for candidates (paged by ``crop_id``) and plan each."""
    plans: list[RepairPlan] = []
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            'query': candidate_query(id_prefix=id_prefix),
            'sort': [{'crop_id': 'asc'}],
            '_source': {'excludes': ['*embedding*']},
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await client.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        cursor = hits[-1].get('sort')
        for h in hits:
            plan = plan_repair(h['_id'], h.get('_source') or {})
            if plan is not None:
                plans.append(plan)
        if len(hits) < page_size or cursor is None:
            break
    return plans


def summarize(plans: list[RepairPlan]) -> dict[str, Counter[str]]:
    """Counts by plan source and by restored ``class_source``."""
    by_source: Counter[str] = Counter(p.source for p in plans)
    by_target: Counter[str] = Counter(
        str(p.restore.get('class_source')) for p in plans if p.applicable
    )
    return {'by_source': by_source, 'by_restored_class_source': by_target}


class _StaleError(Exception):
    pass


async def apply_repairs(client: Any, plans: list[RepairPlan], *, index: str) -> dict[str, int]:
    """Write every applicable plan under OCC. Returns ``repaired`` /
    ``skipped_changed`` / ``not_applicable`` / ``errors`` counts."""
    counts = {'repaired': 0, 'skipped_changed': 0, 'not_applicable': 0, 'errors': 0}
    for plan in plans:
        if not plan.applicable:
            counts['not_applicable'] += 1
            continue

        def _merge(current: dict[str, Any], plan: RepairPlan = plan) -> dict[str, Any]:
            if not is_candidate(current) or current.get('class_id') != plan.current['class_id']:
                raise _StaleError
            return with_class_snapshot(dict(plan.restore), current, writer=REPAIR_WRITER)

        try:
            await occ_update_one(
                client, doc_id=plan.crop_id, merger=_merge, index=index, writer_id=REPAIR_WRITER
            )
        except (_StaleError, OCCFinalConflictError):
            counts['skipped_changed'] += 1
            continue
        except Exception:
            counts['errors'] += 1
            continue
        counts['repaired'] += 1
    return counts


__all__ = [
    'REPAIR_WRITER',
    'SOURCE_HISTORY',
    'SOURCE_INGEST',
    'SOURCE_NONE',
    'SOURCE_VLM',
    'RepairPlan',
    'apply_repairs',
    'candidate_query',
    'ingest_class_source',
    'is_candidate',
    'plan_repair',
    'plan_repairs',
    'summarize',
]
