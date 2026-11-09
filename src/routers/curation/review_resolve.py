"""``POST /curation/review/new_class_proposals/resolve`` — bulk-resolve
every pending VLM new-class proposal for a term in one call.

Split out of ``review.py`` (which owns ``GET
/review/new_class_proposals/summary``) to keep both files under the
per-file LOC ceiling; see ``docs/design/curation_design_rationale.md``
for the module-split convention.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

from src.config.batch_sync import batch_sync_max
from src.config.project_context import current_project_id
from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers._errors import server_error
from src.routers.curation._batch_job_models import BatchJobAccepted
from src.routers.curation._class_models import (
    ResolveConflict,
    ResolveNewClassRequest,
    ResolveNewClassResponse,
)
from src.routers.curation._common import OpenSearchDep, get_class_registry, items_index, router
from src.routers.curation.classes_write import create_registry_class
from src.services.classes.registry import ClassRegistryError
from src.services.curation.export_support import scroll_hits
from src.services.curation.new_class_resolve import (
    queue_resolve_job,
    refresh_items,
    resolve_pending_proposals,
    response_body,
)
from src.services.curation.new_class_terms import proposal_term_query


# Above this many matching pending items, refuse rather than silently
# truncate the cohort a scroll would collect. Large enough that no real
# new-class term (bounded by how many crops a VLM pass proposed the same
# name for) should ever hit it in normal operation.
_RESOLVE_SAFETY_CAP = 5000


async def _pending_crop_ids(opensearch: Any, label: str) -> list[str]:
    """Every open proposal for ``label`` (the selection the queue and its summary use)."""
    query = proposal_term_query(label)
    try:
        count_resp = await opensearch.count(index=items_index(), body={'query': query})
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    total = int(count_resp.get('count', 0))
    if total > _RESOLVE_SAFETY_CAP:
        raise HTTPException(
            status_code=422,
            detail=(
                f'{total} pending items propose {label!r}, over the '
                f'{_RESOLVE_SAFETY_CAP}-item safety cap; resolve in smaller batches'
            ),
        )
    try:
        hits = await scroll_hits(opensearch, index=items_index(), query=query, source=['crop_id'])
    except Exception as exc:
        raise server_error(503, 'opensearch unavailable', exc) from exc
    out: list[str] = []
    for h in hits:
        cid = (h.get('_source') or {}).get('crop_id') or h.get('_id')
        if cid:
            out.append(str(cid))
    return out


async def _target_class(
    payload: ResolveNewClassRequest, *, dry_run: bool
) -> tuple[int | None, str, bool]:
    """``(class_id, class_name, created)``; ``create`` registers the class first, except in a
    dry run (``class_id`` is then ``None``)."""
    store = get_class_registry()
    if payload.create is not None:
        if dry_run:
            return None, payload.create.class_name, False
        try:
            result = await create_registry_class(
                store,
                name=payload.create.class_name,
                group=payload.create.group,
                notes=payload.create.notes or '',
            )
        except ClassRegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result['class_id'], result['class_name'], True
    assert payload.class_id is not None  # exactly-one check in the route
    reg = await store.snapshot()
    if not reg.validate_id(payload.class_id):
        raise HTTPException(status_code=400, detail=f'unknown class_id {payload.class_id}')
    entry = reg.get(payload.class_id)
    return payload.class_id, (entry.class_name if entry is not None else ''), False


async def _queue_resolve(session: Any, params: dict[str, Any], crop_ids: list[str]) -> JSONResponse:
    """Queue the job over ``crop_ids``: 202 with its id."""
    job = await queue_resolve_job(session, current_project_id(), params, crop_ids)
    accepted = BatchJobAccepted(job_id=str(job.id), status=job.status)
    return JSONResponse(status_code=202, content=accepted.model_dump())


@router.post(
    '/review/new_class_proposals/resolve',
    response_model=None,
    responses={200: {'model': ResolveNewClassResponse}, 202: {'model': BatchJobAccepted}},
)
async def resolve_new_class_proposal(
    payload: ResolveNewClassRequest,
    opensearch: OpenSearchDep,
    session: DbSessionDep,
    dry_run: bool = Query(False, description='Report the match without writing or creating.'),
) -> ResolveNewClassResponse | JSONResponse:
    """Bulk-resolve every new-class queue item proposing ``label`` — the
    same selection ``GET /review/new_class_proposals`` and its summary use
    (``vlm_new_class_pending`` rows and ``needs_new_class`` flags, never a
    validated, dismissed, excluded or test-holdout item), so ``matched``
    equals the summary's ``count`` for the term.

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

    Items are written through the paged bulk OCC loop (one read and one
    conditional write per 500 ids), re-checking at write time that each is
    still pending this exact proposal; an item whose state changed in the
    meantime is reported in ``skipped``, not written. Above
    ``OP_BATCH_SYNC_MAX`` matching items (default 500) the write is queued
    instead: ``202`` with ``job_id``, and the same body is the job's
    ``result`` at ``GET /jobs/{job_id}`` (without ``matched_ids``). Every write records
    a restorable ``class_id_history`` snapshot, so
    ``POST /crops/label/undo_batch`` on ``updated_ids`` puts the item
    back exactly as a pending proposal.

    ``?dry_run=true`` reports ``matched`` / ``matched_ids`` and writes or
    creates nothing (``class_id`` is ``null`` when ``create`` was given).
    """
    if (payload.class_id is None) == (payload.create is None):
        raise HTTPException(status_code=422, detail='exactly one of class_id or create is required')

    crop_ids = await _pending_crop_ids(opensearch, payload.label)
    class_id, class_name, created = await _target_class(payload, dry_run=dry_run)
    if dry_run:
        return ResolveNewClassResponse(
            class_id=class_id,
            class_name=class_name,
            created=False,
            label=payload.label,
            matched=len(crop_ids),
            matched_ids=crop_ids,
            updated=0,
            updated_ids=[],
            conflicts=[],
            skipped=[],
        )

    assert class_id is not None  # only None on the dry_run/create branch above
    params = {
        'label': payload.label,
        'class_id': class_id,
        'class_name': class_name,
        'label_source': payload.label_source,
        'created': created,
    }
    if len(crop_ids) > batch_sync_max():
        return await _queue_resolve(session, params, crop_ids)

    outcome = await resolve_pending_proposals(
        opensearch,
        crop_ids,
        label=payload.label,
        class_id=class_id,
        class_name=class_name,
        label_source=payload.label_source,
    )
    if outcome.updated_ids:
        await refresh_items(opensearch)
    body = response_body(params, len(crop_ids), outcome)
    return ResolveNewClassResponse(
        **{**body, 'conflicts': [ResolveConflict(**c) for c in outcome.conflicts]},
        matched_ids=crop_ids,
    )
