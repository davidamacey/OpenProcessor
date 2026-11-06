"""Curation router sub-module — ``POST /classes/merge``: fold one class into another."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

from src.config.project_context import current_project_id
from src.db.session import async_session_factory
from src.routers.curation._class_models import ClassMergeRequest  # noqa: TC001 - FastAPI
from src.routers.curation._common import OpenSearchDep, get_class_registry, items_index, router
from src.services.classes import projection_sync
from src.services.jobs import service as jobs
from src.services.jobs.status import get_job


async def _merge_dry_run(payload: ClassMergeRequest, opensearch: Any) -> dict[str, Any]:
    reg = await get_class_registry().snapshot()
    for cid in (payload.source_id, payload.target_id):
        if reg.get(cid) is None:
            raise HTTPException(status_code=400, detail=f'class_id {cid} not found')
    if payload.source_id == payload.target_id:
        raise HTTPException(status_code=400, detail='cannot merge a class into itself')
    index = items_index()
    holdout = await projection_sync.count_items(
        opensearch, index, projection_sync.holdout_of_class_query(payload.source_id)
    )
    relabel = await projection_sync.count_items(
        opensearch, index, projection_sync.merge_source_query([payload.source_id])
    )
    # A merge keeps class_validated: this counts the crops the carry-over applies to.
    validated = await projection_sync.count_items(
        opensearch,
        index,
        {
            'bool': {
                'filter': [
                    {'term': {'class_id': payload.source_id}},
                    {'term': {'class_validated': True}},
                ],
                'must_not': [projection_sync.HOLDOUT],
            }
        },
    )
    return {
        'dry_run': True,
        'source_id': payload.source_id,
        'target_id': payload.target_id,
        'would_relabel': relabel,
        'validations_carried_over': validated,
        'holdout_blocking': holdout,
        'blocked': holdout > 0,
    }


@router.post('/classes/merge', response_model=None)
async def merge_class(
    payload: ClassMergeRequest,
    opensearch: OpenSearchDep,
    dry_run: Annotated[bool, Query(description='Report counts; change nothing.')] = False,
) -> dict[str, Any] | JSONResponse:
    """Queue the merge of ``source_id`` into ``target_id``: 202 with the ``class_merge`` job.

    The job deprecates the source in the registry, then bulk-relabels its crops and labels; a
    merge whose source holds frozen ``test_holdout`` crops is refused with 409 here and again by
    the job. ``dry_run=true`` returns ``{dry_run, source_id, target_id, would_relabel,
    validations_carried_over, holdout_blocking, blocked}`` and writes nothing. A human-validated
    crop of the source class stays validated under the target.
    """
    counts = await _merge_dry_run(payload, opensearch)
    if dry_run:
        return counts
    if counts['holdout_blocking']:
        raise HTTPException(
            status_code=409,
            detail=(
                f'cannot merge class {payload.source_id}: {counts["holdout_blocking"]} '
                'frozen test_holdout crop(s) would be relabeled'
            ),
        )
    target = (await get_class_registry().snapshot()).get(payload.target_id)
    if target is not None and target.deprecated:
        raise HTTPException(
            status_code=400,
            detail=f'target class_id {payload.target_id} is deprecated; cannot merge into it',
        )
    async with async_session_factory()() as session:
        started = await jobs.start_job(
            session,
            'class_merge',
            current_project_id(),
            params={'source_id': payload.source_id, 'target_id': payload.target_id},
        )
        job = await get_job(session, started.job.id, project=current_project_id())
    return JSONResponse(job.model_dump(mode='json'), status_code=202)
