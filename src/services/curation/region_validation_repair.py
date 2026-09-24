"""Re-label machine-set region validation as auto-confirm.

Before ``RegionFields.auto_confirmed`` existed, the detection worker wrote
its auto-confirm verdict (detector + verifier agreed strongly enough) into
``RegionFields.validated`` -- the field every reader treats as *human*
validation. Those regions were hidden from the human region-review queue
and counted as human ground truth.

This repair finds rows with ``validated=true`` that carry no human
verdict (:func:`is_human_region_verdict`) and rewrites them to
``validated=false, auto_confirmed=true``: still accepted regions, now
reviewable. Nothing else changes. The rewrite is its own record -- every
repaired row is exactly a row with ``auto_confirmed=true`` written by
:data:`REPAIR_WRITER` -- so it is not added to the human edit history
(a human region undo must never undo an operator repair).

Applying re-checks each row under OCC (still validated, still no human
verdict); a row a human touched since planning is left alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.config import get_region_fields
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind
from src.services.detection.profile_registry import region_profile_or_neutral


REPAIR_WRITER = 'operator:repair_region_validation'


def is_human_region_verdict(doc: dict[str, Any]) -> bool:
    """True when a human wrote this region's verdict.

    Every human region writer stamps a ``human*`` label source, or the
    human verifier / detector name, and snapshots a ``human:*`` region
    edit into the edit history; any one of them counts.
    """
    F = get_region_fields()
    human = region_profile_or_neutral().human_detector_name
    if str(doc.get(F.label_source) or '').lower().startswith('human'):
        return True
    if human and human in (doc.get(F.verifier), doc.get(F.detector)):
        return True
    return any(
        isinstance(e, dict)
        and e.get('kind') == EditKind.REGION.value
        and str(e.get('writer') or '').startswith('human')
        for e in doc.get(EDIT_HISTORY_FIELD) or []
    )


def _needs_repair(doc: dict[str, Any]) -> bool:
    return doc.get(get_region_fields().validated) is True and not is_human_region_verdict(doc)


@dataclass
class RegionValidationRepairPlan:
    """What a repair would change: the machine-validated ids, how many
    validated rows were kept as human verdicts, and the machine rows by
    region status and detector."""

    machine_ids: list[str] = field(default_factory=list)
    human_kept: int = 0
    by_status: Counter[str] = field(default_factory=Counter)
    by_detector: Counter[str] = field(default_factory=Counter)


async def plan_region_validation_repair(
    client: Any, *, index: str, page_size: int = 500
) -> RegionValidationRepairPlan:
    """Scan ``index`` (read-only) for validated regions and classify them."""
    F = get_region_fields()
    plan = RegionValidationRepairPlan()
    cursor: list[Any] | None = None
    while True:
        body: dict[str, Any] = {
            'size': page_size,
            'query': {'bool': {'filter': [{'term': {F.validated: True}}]}},
            'sort': [{'crop_id': 'asc'}],
            '_source': {
                'includes': [
                    'crop_id',
                    F.validated,
                    F.status,
                    F.label_source,
                    F.verifier,
                    F.detector,
                    EDIT_HISTORY_FIELD,
                ]
            },
        }
        if cursor is not None:
            body['search_after'] = cursor
        resp = await client.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            break
        cursor = hits[-1].get('sort')
        for h in hits:
            doc = h.get('_source') or {}
            if not _needs_repair(doc):
                plan.human_kept += 1
                continue
            plan.machine_ids.append(h['_id'])
            plan.by_status[str(doc.get(F.status))] += 1
            plan.by_detector[str(doc.get(F.detector))] += 1
        if len(hits) < page_size or cursor is None:
            break
    return plan


async def apply_region_validation_repair(
    client: Any, plan: RegionValidationRepairPlan, *, index: str
) -> dict[str, Any]:
    """Rewrite the planned rows under OCC; returns the bulk writer's
    ``updated`` / ``skipped_due_to_conflict`` / ``errors``."""
    F = get_region_fields()
    now = datetime.now(UTC).isoformat()

    def _merge(_doc_id: str, current: dict[str, Any]) -> dict[str, Any]:
        if not _needs_repair(current):
            return {}
        return {F.validated: False, F.auto_confirmed: True, 'updated_at': now}

    if not plan.machine_ids:
        return {'updated': 0, 'skipped_due_to_conflict': 0, 'errors': []}
    return await occ_skip_on_conflict_bulk(
        client,
        doc_ids=list(plan.machine_ids),
        merger=_merge,
        index=index,
        refresh='wait_for',
        writer_id=REPAIR_WRITER,
    )


__all__ = [
    'REPAIR_WRITER',
    'RegionValidationRepairPlan',
    'apply_region_validation_repair',
    'is_human_region_verdict',
    'plan_region_validation_repair',
]
