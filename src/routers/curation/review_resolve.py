"""``POST /curation/review/new_class_proposals/resolve`` — bulk-resolve
every pending VLM new-class proposal for a term in one call.

Split out of ``review.py`` (which owns ``GET
/review/new_class_proposals/summary``) to keep both files under the
per-file LOC ceiling; see ``docs/design/curation_design_rationale.md``
for the module-split convention.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException, Query

from src.clients.curation_opensearch import ClassRegistryError
from src.clients.occ import OCCFinalConflictError, occ_update_one
from src.routers.curation._class_models import (
    ResolveConflict,
    ResolveNewClassRequest,
    ResolveNewClassResponse,
)
from src.routers.curation._common import (
    CURATION_ITEMS_INDEX,
    OpenSearchDep,
    _ensure_indexes,
    get_class_registry,
    logger,
    router,
)
from src.routers.curation.classes import create_registry_class
from src.services.curation.class_sources import VLM_NEW_CLASS_PENDING_CLASS_SOURCE
from src.services.curation.export_support import scroll_hits
from src.services.curation.human_label import human_label_update


_RESOLVE_WRITER = 'human:resolve_new_class'
_RESOLVE_CONCURRENCY = 16
# Above this many matching pending items, refuse rather than silently
# truncate the cohort a scroll would collect. Large enough that no real
# new-class term (bounded by how many crops a VLM pass proposed the same
# name for) should ever hit it in normal operation.
_RESOLVE_SAFETY_CAP = 5000


class _PendingStateChangedError(Exception):
    """The item is no longer pending this exact proposal at write time."""


def _pending_proposal_query(label: str) -> dict[str, Any]:
    return {
        'bool': {
            'must': [
                {'term': {'class_source': VLM_NEW_CLASS_PENDING_CLASS_SOURCE}},
                {'term': {'vlm_proposed_class': label}},
            ],
            'must_not': [{'term': {'class_validated': True}}],
        }
    }


def _resolve_merger(*, label: str, class_id: int, class_name: str, label_source: str) -> Any:
    def _merge(current: dict[str, Any]) -> dict[str, Any]:
        if (
            current.get('class_source') != VLM_NEW_CLASS_PENDING_CLASS_SOURCE
            or current.get('vlm_proposed_class') != label
            or current.get('class_validated')
        ):
            # Re-checked on every OCC attempt against the freshly fetched
            # doc: someone else already resolved (or otherwise changed)
            # this item's pending state between selection and this write.
            raise _PendingStateChangedError
        update = human_label_update(
            current,
            class_id=class_id,
            class_name=class_name,
            label_source=label_source,
            writer=_RESOLVE_WRITER,
        )
        update['needs_new_class'] = False
        return update

    return _merge


async def _resolve_one(
    opensearch: Any, crop_id: str, merger: Any
) -> tuple[str, dict[str, Any] | None]:
    """``(outcome, conflict)`` — outcome is ``updated`` / ``skipped`` / ``conflict``."""
    try:
        await occ_update_one(
            opensearch,
            doc_id=crop_id,
            merger=merger,
            max_retries=5,
            refresh=False,
            writer_id=_RESOLVE_WRITER,
        )
        return 'updated', None
    except _PendingStateChangedError:
        return 'skipped', None
    except OCCFinalConflictError:
        try:
            doc = await opensearch.get(index=CURATION_ITEMS_INDEX, id=crop_id)
            current_source = (doc.get('_source') or {}).get('class_source')
        except Exception:
            current_source = None
        return 'conflict', {'crop_id': crop_id, 'current_source': current_source}
    except Exception as exc:
        logger.warning('resolve_new_class_update_failed', crop_id=crop_id, error=str(exc))
        return 'conflict', {'crop_id': crop_id, 'current_source': None}


def _selected_crop_ids(hits: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for h in hits:
        cid = (h.get('_source') or {}).get('crop_id') or h.get('_id')
        if cid:
            out.append(str(cid))
    return out


@router.post('/review/new_class_proposals/resolve', response_model=ResolveNewClassResponse)
async def resolve_new_class_proposal(
    payload: ResolveNewClassRequest,
    opensearch: OpenSearchDep,
    dry_run: bool = Query(False, description='Report the match without writing or creating.'),
) -> ResolveNewClassResponse:
    """Bulk-resolve every ``vlm_new_class_pending`` item proposing ``label``.

    Unlike relabeling ``GET /review/new_class_proposals/summary``'s
    ``sample_crop_ids`` (capped at 20) one at a time, this selects and
    writes **every** matching, still-pending item for the term in one
    call — the backend owns the "resolve this term" decision, not the
    client re-issuing ``PUT /crops/batch_label`` per page of samples.

    Exactly one of ``class_id`` (map to an existing registry class) /
    ``create`` (register a new one first, through the same path as
    ``POST /classes``) must be set — ``422`` otherwise. An unknown
    ``class_id`` is ``400``; a duplicate ``create.class_name`` is ``409``
    and writes nothing. ``create`` happens before any item write — if
    nothing matches, the class is still created (the operator asked for
    it) and ``matched`` reports ``0``.

    Each matching item is written independently via ``occ_update_one``
    (bounded concurrency), re-checking at write time that it is still
    pending this exact proposal; an item whose state changed in the
    meantime is reported in ``skipped``, not written. Every write records
    a restorable ``class_id_history`` snapshot, so
    ``POST /crops/label/undo_batch`` on ``updated_ids`` puts the item
    back exactly as a pending proposal.

    ``?dry_run=true`` reports ``matched`` / ``matched_ids`` and writes or
    creates nothing (``class_id`` is ``null`` when ``create`` was given).
    """
    await _ensure_indexes(opensearch)
    if (payload.class_id is None) == (payload.create is None):
        raise HTTPException(status_code=422, detail='exactly one of class_id or create is required')

    query = _pending_proposal_query(payload.label)
    try:
        count_resp = await opensearch.count(index=CURATION_ITEMS_INDEX, body={'query': query})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    total = int(count_resp.get('count', 0))
    if total > _RESOLVE_SAFETY_CAP:
        raise HTTPException(
            status_code=422,
            detail=(
                f'{total} pending items propose {payload.label!r}, over the '
                f'{_RESOLVE_SAFETY_CAP}-item safety cap; resolve in smaller batches'
            ),
        )
    try:
        hits = await scroll_hits(
            opensearch, index=CURATION_ITEMS_INDEX, query=query, source=['crop_id']
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'opensearch unavailable: {exc}') from exc
    crop_ids = _selected_crop_ids(hits)
    matched = len(crop_ids)

    reg = get_class_registry()
    created = False
    class_id: int | None
    class_name: str

    if payload.create is not None:
        if dry_run:
            class_id, class_name = None, payload.create.class_name
        else:
            try:
                result = create_registry_class(
                    reg,
                    name=payload.create.class_name,
                    group=payload.create.group,
                    notes=payload.create.notes or '',
                )
            except ClassRegistryError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            class_id, class_name, created = result['class_id'], result['class_name'], True
    else:
        assert payload.class_id is not None  # exactly-one check above
        if not reg.validate_id(payload.class_id):
            raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
        entry = reg.get(payload.class_id)
        class_id, class_name = payload.class_id, (entry.class_name if entry is not None else '')

    if dry_run:
        return ResolveNewClassResponse(
            class_id=class_id,
            class_name=class_name,
            created=False,
            label=payload.label,
            matched=matched,
            matched_ids=crop_ids,
            updated=0,
            updated_ids=[],
            conflicts=[],
            skipped=[],
        )

    assert class_id is not None  # only None on the dry_run/create branch above
    merger = _resolve_merger(
        label=payload.label,
        class_id=class_id,
        class_name=class_name,
        label_source=payload.label_source,
    )
    sem = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

    async def _bounded(crop_id: str) -> tuple[str, dict[str, Any] | None]:
        async with sem:
            return await _resolve_one(opensearch, crop_id, merger)

    outcomes = await asyncio.gather(*(_bounded(cid) for cid in crop_ids))
    updated_ids: list[str] = []
    conflicts: list[ResolveConflict] = []
    skipped: list[str] = []
    for crop_id, (outcome, conflict) in zip(crop_ids, outcomes, strict=True):
        if outcome == 'updated':
            updated_ids.append(crop_id)
        elif outcome == 'skipped':
            skipped.append(crop_id)
        else:
            conflicts.append(ResolveConflict(**(conflict or {'crop_id': crop_id})))

    if updated_ids:
        try:
            await opensearch.indices.refresh(index=CURATION_ITEMS_INDEX)
        except Exception as exc:
            logger.warning('resolve_new_class_refresh_failed', error=str(exc))

    return ResolveNewClassResponse(
        class_id=class_id,
        class_name=class_name,
        created=created,
        label=payload.label,
        matched=matched,
        matched_ids=crop_ids,
        updated=len(updated_ids),
        updated_ids=updated_ids,
        conflicts=conflicts,
        skipped=skipped,
    )
