"""Restore region detector provenance lost to a same-box human confirm.

Before the fix in :func:`src.services.curation.region_writes.region_box_write`,
a human confirming a detector's region with ``PUT /crops/{id}/region``
carrying the *unchanged* box re-stamped the detector as the human and the
score as ``1.0``, destroying which model found the box and how confident
it was. This module finds those items and plans the restore of the
detector, its version, score and detection time.

Candidates: region detector is the human (profile ``human_detector_name``),
a box is stored, and the detector chain records a model ``hit`` — a model
found a box that a human then took over. Sources for the pre-confirm
provenance, in order:

1. ``edit_history`` — a region snapshot whose box equals the stored box
   and whose detector was a model (edits made after snapshots existed);
2. a backup index (``backup_index``) holding the same doc from before the
   confirm, again only when its box equals the stored box;
3. the detector chain alone. It names the detector (and, for older
   ``<actor>::<event>@<iso>`` entries, the detection time) but not the
   score or the pre-confirm box, so it can't tell a same-box confirm from
   a human redraw: such a plan is reported, never applied.

Applying re-checks each item under OCC (detector still human, box still
the planned one) and records a region ``edit_history`` snapshot, so
``POST /crops/{id}/region/undo`` reverts a restore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.config import get_region_fields
from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
from src.services.curation.history import normalize_region_chain_entry
from src.services.curation.region_writes import same_box
from src.services.detection.profile_registry import region_profile_or_neutral


RESTORE_WRITER = 'operator:restore_region_provenance'

SOURCE_HISTORY = 'edit_history'
SOURCE_BACKUP = 'backup_index'
SOURCE_CHAIN = 'detector_chain'


def provenance_fields() -> tuple[str, ...]:
    F = get_region_fields()
    return (F.detector, F.detector_version, F.score, F.detected_at)


@dataclass
class RestorePlan:
    crop_id: str
    source: str
    applicable: bool
    box: list[float] | None
    current: dict[str, Any]
    restore: dict[str, Any]
    note: str = ''
    fields: tuple[str, ...] = field(default_factory=provenance_fields)


def _human_name() -> str:
    return region_profile_or_neutral().human_detector_name


def _chain_hit(chain: Any) -> tuple[str, str | None] | None:
    """``(actor, detected_at_or_None)`` of the last non-human ``hit``."""
    human = _human_name()
    for raw in reversed(chain or []):
        text = str(raw)
        at = None
        at_pos = text.rfind('@')
        if at_pos > 0 and text[at_pos + 1 : at_pos + 2].isdigit():
            at = text[at_pos + 1 :]
        entry = normalize_region_chain_entry(text)
        actor, _, event = entry.partition(':')
        if event == 'hit' and actor and actor != human:
            return actor, at
    return None


def is_candidate(doc: dict[str, Any]) -> bool:
    F = get_region_fields()
    return (
        doc.get(F.detector) == _human_name()
        and bool(doc.get(F.bbox_norm))
        and _chain_hit(doc.get(F.detector_chain)) is not None
    )


def _from_state(state: dict[str, Any]) -> dict[str, Any]:
    return {f: state.get(f) for f in provenance_fields()}


def _model_state_with_box(state: dict[str, Any], box: Any) -> bool:
    F = get_region_fields()
    detector = state.get(F.detector)
    return bool(detector) and detector != _human_name() and same_box(state.get(F.bbox_norm), box)


def plan_restore(
    crop_id: str, doc: dict[str, Any], backup_doc: dict[str, Any] | None = None
) -> RestorePlan | None:
    """The restore plan for one candidate doc, or ``None`` if it isn't one."""
    if not is_candidate(doc):
        return None
    F = get_region_fields()
    box = list(doc[F.bbox_norm])
    current = _from_state(doc)
    for entry in reversed(doc.get(EDIT_HISTORY_FIELD) or []):
        if not isinstance(entry, dict) or entry.get('kind') != EditKind.REGION.value:
            continue
        state = entry.get('state') or {}
        if _model_state_with_box(state, box):
            return RestorePlan(
                crop_id,
                SOURCE_HISTORY,
                True,
                box,
                current,
                _from_state(state),
                note=f'snapshot before {entry.get("writer")} at {entry.get("at")}',
            )
    if backup_doc is not None and _model_state_with_box(backup_doc, box):
        return RestorePlan(crop_id, SOURCE_BACKUP, True, box, current, _from_state(backup_doc))
    actor, at = _chain_hit(doc.get(F.detector_chain)) or ('', None)
    restore = dict.fromkeys(provenance_fields())
    restore[F.detector] = actor
    restore[F.detected_at] = at
    return RestorePlan(
        crop_id,
        SOURCE_CHAIN,
        False,
        box,
        current,
        restore,
        note='chain names the detector only: score, version and the pre-confirm box '
        'are unrecorded, so a same-box confirm cannot be told from a human redraw',
    )


def candidate_query(*, id_prefix: str | None = None) -> dict[str, Any]:
    F = get_region_fields()
    must: list[dict[str, Any]] = [
        {'term': {F.detector: _human_name()}},
        {'exists': {'field': F.bbox_norm}},
        {'exists': {'field': F.detector_chain}},
    ]
    if id_prefix:
        must.append({'wildcard': {'crop_id': f'{id_prefix}*'}})
    return {'bool': {'must': must}}


async def plan_restores(
    client: Any,
    *,
    index: str,
    backup_index: str | None = None,
    id_prefix: str | None = None,
    page_size: int = 500,
) -> list[RestorePlan]:
    """Scan ``index`` for candidates (paged by ``crop_id``) and plan each."""
    plans: list[RestorePlan] = []
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
        backups: dict[str, dict[str, Any]] = {}
        if backup_index:
            got = await client.mget(
                body={'docs': [{'_index': backup_index, '_id': h['_id']} for h in hits]}
            )
            backups = {
                d['_id']: d.get('_source') or {} for d in got.get('docs') or [] if d.get('found')
            }
        for h in hits:
            plan = plan_restore(h['_id'], h.get('_source') or {}, backups.get(h['_id']))
            if plan is not None:
                plans.append(plan)
        if len(hits) < page_size or cursor is None:
            break
    return plans


class _StaleError(Exception):
    pass


async def apply_restores(client: Any, plans: list[RestorePlan], *, index: str) -> dict[str, int]:
    """Write every applicable plan under OCC. Returns ``restored`` /
    ``skipped_changed`` / ``not_applicable`` / ``errors`` counts."""
    F = get_region_fields()
    counts = {'restored': 0, 'skipped_changed': 0, 'not_applicable': 0, 'errors': 0}
    for plan in plans:
        if not plan.applicable:
            counts['not_applicable'] += 1
            continue

        def _merge(current: dict[str, Any], plan: RestorePlan = plan) -> dict[str, Any]:
            if current.get(F.detector) != _human_name() or not same_box(
                current.get(F.bbox_norm), plan.box
            ):
                raise _StaleError
            return {
                **plan.restore,
                EDIT_HISTORY_FIELD: record_edit(
                    current, kind=EditKind.REGION, writer=RESTORE_WRITER
                ),
            }

        try:
            await occ_update_one(
                client, doc_id=plan.crop_id, merger=_merge, index=index, writer_id=RESTORE_WRITER
            )
        except (_StaleError, OCCFinalConflictError):
            counts['skipped_changed'] += 1
            continue
        except Exception:
            counts['errors'] += 1
            continue
        counts['restored'] += 1
    return counts


__all__ = [
    'RESTORE_WRITER',
    'SOURCE_BACKUP',
    'SOURCE_CHAIN',
    'SOURCE_HISTORY',
    'RestorePlan',
    'apply_restores',
    'candidate_query',
    'is_candidate',
    'plan_restore',
    'plan_restores',
    'provenance_fields',
]
