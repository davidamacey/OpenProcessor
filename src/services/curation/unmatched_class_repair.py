"""Repair legacy ``vlm_unmatched`` writes that still carry a class_id (IT-2).

Before the write-time fix (:func:`src.services.curation.class_sources.unmatched_class_clear`,
wired into ``scripts/curation/worker/bulk_writer.py``'s combined detect+verify
path and ``src/routers/curation/vlm.py``'s label-batch endpoint), a
``class_source='vlm_unmatched'`` write left the item's prior
``class_id``/``class_name`` in place -- the write's own ``class_source``
says the VLM's answer wasn't matched to a registry class, yet the item still
showed one. This module finds every row still carrying that stale
combination and clears it the same way the fixed writers now do at write
time (restorable: a ``class_id_history`` snapshot precedes every write).

Dry run by default (read-only, see ``scripts/curation/repair_unmatched_class.py``);
``--apply`` writes under OCC, skipping any row that changed between plan and
apply (a human edit, a validation) or is now human-owned/validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.services.curation.class_sources import VLM_UNMATCHED_CLASS_SOURCE, unmatched_class_clear
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.history import record_class_snapshot


REPAIR_WRITER = 'operator:repair_unmatched_class'


def is_candidate(doc: dict[str, Any]) -> bool:
    """A row this repair may still touch: ``vlm_unmatched`` with a class_id,
    not human-owned or validated."""
    return (
        doc.get('class_source') == VLM_UNMATCHED_CLASS_SOURCE
        and doc.get('class_id') is not None
        and not class_write_locked(doc)
    )


def candidate_query(*, id_prefix: str | None = None) -> dict[str, Any]:
    """Every ``vlm_unmatched`` row that still carries a ``class_id``.

    Includes human-owned/validated rows too (so the operator sees exactly
    how many are locked and being skipped, rather than a query that
    silently under-counts); :func:`plan_repair` marks those not applicable.
    """
    query: dict[str, Any] = {
        'bool': {
            'filter': [
                {'term': {'class_source': VLM_UNMATCHED_CLASS_SOURCE}},
                {'exists': {'field': 'class_id'}},
            ]
        }
    }
    if id_prefix:
        query['bool']['filter'].append({'wildcard': {'crop_id': f'{id_prefix}*'}})
    return query


@dataclass
class RepairPlan:
    crop_id: str
    applicable: bool
    current: dict[str, Any]
    restore: dict[str, Any]
    note: str = ''


def plan_repair(crop_id: str, doc: dict[str, Any]) -> RepairPlan | None:
    """The repair plan for one doc, or ``None`` if it isn't a candidate at
    all (not ``vlm_unmatched``, or already has no ``class_id``)."""
    if doc.get('class_source') != VLM_UNMATCHED_CLASS_SOURCE or doc.get('class_id') is None:
        return None
    current = {f: doc.get(f) for f in ('class_id', 'class_name', 'class_source', 'cluster_id')}
    if class_write_locked(doc):
        return RepairPlan(
            crop_id, False, current, {}, note='human-owned or class-validated -- skipped'
        )
    return RepairPlan(crop_id, True, current, unmatched_class_clear(doc))


async def plan_repairs(
    client: Any,
    *,
    index: str,
    id_prefix: str | None = None,
    page_size: int = 500,
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


class _StaleError(Exception):
    pass


async def apply_repairs(client: Any, plans: list[RepairPlan], *, index: str) -> dict[str, int]:
    """Write every applicable plan under OCC. Returns ``repaired`` /
    ``skipped_changed`` / ``not_applicable`` / ``errors`` counts.

    Recomputes the restore fields from the write-time ``current`` doc
    (never the planned snapshot) -- the same freshness rule
    ``unmatched_class_clear``'s write-time callers follow.

    Always records a ``class_id_history`` snapshot directly (rather than
    ``with_class_snapshot``'s "only if ``class_source`` changed" gate,
    which never fires here -- clearing ``class_id``/``class_name`` while
    the row stays ``class_source='vlm_unmatched'`` doesn't touch
    ``class_source`` at all): every applicable plan is by definition a
    real class-state change worth making restorable.
    """
    counts = {'repaired': 0, 'skipped_changed': 0, 'not_applicable': 0, 'errors': 0}
    for plan in plans:
        if not plan.applicable:
            counts['not_applicable'] += 1
            continue

        def _merge(current: dict[str, Any], plan: RepairPlan = plan) -> dict[str, Any]:
            if not is_candidate(current) or current.get('class_id') != plan.current['class_id']:
                raise _StaleError
            return {
                **unmatched_class_clear(current),
                'class_id_history': record_class_snapshot(
                    current, writer=REPAIR_WRITER, restorable=True
                ),
            }

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
    'RepairPlan',
    'apply_repairs',
    'candidate_query',
    'is_candidate',
    'plan_repair',
    'plan_repairs',
]
