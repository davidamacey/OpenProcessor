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
from typing import TYPE_CHECKING, Any

from src.domain.history import record_class_snapshot
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.repair_scan import StaleRepairError, apply_occ_repairs, planner


if TYPE_CHECKING:
    from collections.abc import Callable


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


plan_repairs = planner(candidate_query, plan_repair)


async def apply_repairs(client: Any, plans: list[RepairPlan], *, index: str) -> dict[str, int]:
    """Clear ``label_source`` on every planned item, under OCC. Returns
    ``repaired`` / ``skipped_changed`` / ``errors`` counts."""

    def merger_for(_plan: RepairPlan) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def _merge(current: dict[str, Any]) -> dict[str, Any]:
            if not is_candidate(current):
                raise StaleRepairError
            return {
                'label_source': None,
                'class_id_history': record_class_snapshot(
                    current, writer=REPAIR_WRITER, restorable=True
                ),
            }

        return _merge

    return await apply_occ_repairs(
        client,
        plans,
        index=index,
        writer_id=REPAIR_WRITER,
        merger_for=merger_for,
        counts={'repaired': 0, 'skipped_changed': 0, 'errors': 0},
    )


__all__ = [
    'REPAIR_WRITER',
    'RepairPlan',
    'apply_repairs',
    'candidate_query',
    'is_candidate',
    'plan_repair',
    'plan_repairs',
]
