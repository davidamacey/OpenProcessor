"""Clear a stale ``label_source`` left on a class-less item.

``label_source`` means "who set this item's class". A class-less item
(``class_id is None``) has no class, so a non-null ``label_source`` on
one is always stale -- left over from a writer that stamped it anyway.

The current writers (:func:`src.services.curation.vlm_class_attempt.
prediction_class_update`, ``scripts/curation/worker/verify.py``'s
``_combined_class_update``) were audited and do NOT do this: an empty
VLM class answer only records the attempt
(:mod:`src.services.curation.vlm_class_attempt`) and leaves every class
field -- including ``label_source`` -- untouched. This module is a
repair for legacy data written before those write paths existed, or by
any other bypass this audit didn't find; it is narrower and more general
than :mod:`src.services.curation.empty_vlm_answer_repair` (which
restores full prior provenance for one specific pattern,
``class_source='vlm_unmatched'`` with an empty raw label) -- this one
only clears the one field that is never valid without a class, whatever
``class_source`` says.

A human-owned or validated item is never touched
(:func:`~src.services.curation.class_write_guard.class_write_locked`) --
if a class exists and is validated, ``label_source`` is legitimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.history import record_class_snapshot


REPAIR_WRITER = 'operator:repair_stale_label_source'


def is_candidate(doc: dict[str, Any]) -> bool:
    """A class-less item that still carries a ``label_source``."""
    return (
        doc.get('class_id') is None
        and bool(doc.get('label_source'))
        and not class_write_locked(doc)
    )


def candidate_query(*, id_prefix: str | None = None) -> dict[str, Any]:
    """OpenSearch query matching every candidate."""
    query: dict[str, Any] = {
        'bool': {
            'must_not': [{'exists': {'field': 'class_id'}}],
            'filter': [{'exists': {'field': 'label_source'}}],
        }
    }
    if id_prefix:
        query['bool']['filter'].append({'wildcard': {'crop_id': f'{id_prefix}*'}})
    return query


@dataclass
class RepairPlan:
    crop_id: str
    label_source: str
    prior_class_source: str | None


def plan_repair(crop_id: str, doc: dict[str, Any]) -> RepairPlan | None:
    if not is_candidate(doc):
        return None
    return RepairPlan(
        crop_id=crop_id,
        label_source=str(doc.get('label_source')),
        prior_class_source=doc.get('class_source'),
    )


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
    """Clear ``label_source`` on every planned item, under OCC. Returns
    ``repaired`` / ``skipped_changed`` / ``errors`` counts."""
    counts = {'repaired': 0, 'skipped_changed': 0, 'errors': 0}
    for plan in plans:
        # plan=plan default binds this loop iteration's plan by value (the
        # closure is invoked later, inside occ_update_one, after the loop
        # variable would otherwise have moved on).
        def _merge(current: dict[str, Any], plan: RepairPlan = plan) -> dict[str, Any]:  # noqa: ARG001
            if not is_candidate(current):
                raise _StaleError
            return {
                'label_source': None,
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
