"""Curation router sub-module: unified reprocess (W10.13).

``POST /reprocess`` (batch, dry run by default), the thin per-image and
per-item forms (an explicit button: apply by default) and the job status /
cancel routes. All of them build one
:class:`~src.services.curation.reprocess_models.ReprocessRequest` and call
the same :func:`~src.services.curation.reprocess.apply_reprocess`; the lock
rule (human and imported labels and boxes are never written) is enforced
there, not here.
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import (
    OpenSearchDep,
    RegistryDep,
    images_index,
    is_not_found,
    items_index,
    router,
)
from src.routers.curation._config_common_models import api_error
from src.routers.curation._reprocess_models import (
    ReprocessJobInfo,
    ReprocessOneRequest,
    ReprocessRequest,
    ReprocessTargets,
    ReprocessWireResponse,
)
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_job import ReprocessBusyError, cancel_job, read_job
from src.services.curation.reprocess_targets import ReprocessTargetsError
from src.services.curation.wire import item_source_excludes, serialize_item


async def _service_factory(opensearch: Any, registry: Any) -> Any:
    from src.routers.curation.ingest import _get_ingest_service

    return await _get_ingest_service(opensearch, registry)


async def _run(request: ReprocessRequest, opensearch: Any, registry: Any) -> ReprocessWireResponse:
    async def factory() -> Any:
        return await _service_factory(opensearch, registry)

    try:
        result = await apply_reprocess(opensearch, request, service_factory=factory)
    except ReprocessTargetsError as exc:
        raise api_error(422, 'reprocess_targets_invalid', str(exc)) from exc
    except ReprocessBusyError as exc:
        raise api_error(409, 'reprocess_busy', str(exc)) from exc
    return ReprocessWireResponse(**result.model_dump())


async def _items_where(opensearch: Any, field: str, value: str) -> list[dict[str, Any]]:
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 1000,
            '_source': {'excludes': item_source_excludes()},
            'query': {'bool': {'filter': [{'term': {field: value}}]}},
            'sort': [{'crop_id': 'asc'}],
        },
    )
    hits = (resp.get('hits') or {}).get('hits') or []
    return [serialize_item(h.get('_source') or {}, h['_id']) for h in hits]


async def _with_items(
    result: ReprocessWireResponse, opensearch: Any, field: str, value: str, dry_run: bool
) -> ReprocessWireResponse:
    """``result`` plus the target's items as written (not on a dry run)."""
    if dry_run:
        return result
    items = await _items_where(opensearch, field, value)
    return ReprocessWireResponse.model_validate({**result.model_dump(), 'items': items})


@router.post('/reprocess', response_model=ReprocessWireResponse)
async def reprocess_batch(
    body: ReprocessRequest, opensearch: OpenSearchDep, registry: RegistryDep
) -> ReprocessWireResponse:
    """Re-run one or more scopes (``detect`` / ``region`` / ``vlm`` /
    ``embed``) over images, items or a filter. Dry run by default: the
    response counts, per scope, what is selected and what the lock rule
    skips (``locked_skipped``). Detect and embed over many images return a
    ``job`` to poll at ``GET /reprocess/jobs/{job_id}``."""
    return await _run(body, opensearch, registry)


@router.post('/images/{image_id}/reprocess', response_model=ReprocessWireResponse)
async def reprocess_image(
    image_id: str, body: ReprocessOneRequest, opensearch: OpenSearchDep, registry: RegistryDep
) -> ReprocessWireResponse:
    """The explicit per-image button: apply (``dry_run`` false) by default.
    The response carries the image's items as written."""
    try:
        resp = await opensearch.search(
            index=images_index(),
            body={'size': 1, '_source': ['image_id'], 'query': {'term': {'image_id': image_id}}},
        )
    except Exception as exc:
        if not is_not_found(exc):
            raise
        resp = {}
    if not ((resp.get('hits') or {}).get('hits') or []):
        raise api_error(404, 'image_not_found', f'no image {image_id}')
    result = await _run(
        ReprocessRequest(
            targets=ReprocessTargets(image_ids=[image_id]),
            scopes=body.scopes,
            region_mode=body.region_mode,
            dry_run=body.dry_run,
        ),
        opensearch,
        registry,
    )
    return await _with_items(result, opensearch, 'image_id', image_id, body.dry_run)


@router.post('/crops/{crop_id}/reprocess', response_model=ReprocessWireResponse)
async def reprocess_crop(
    crop_id: str, body: ReprocessOneRequest, opensearch: OpenSearchDep, registry: RegistryDep
) -> ReprocessWireResponse:
    """The explicit per-item button: apply by default. The response carries
    the item as written; a locked set with nothing to regenerate reports
    ``locked_skipped: 1`` and ``queued: 0``."""
    if not await _items_where(opensearch, 'crop_id', crop_id):
        raise api_error(404, 'not_found', f'no item {crop_id}')
    result = await _run(
        ReprocessRequest(
            targets=ReprocessTargets(crop_ids=[crop_id]),
            scopes=body.scopes,
            region_mode=body.region_mode,
            dry_run=body.dry_run,
        ),
        opensearch,
        registry,
    )
    return await _with_items(result, opensearch, 'crop_id', crop_id, body.dry_run)


@router.get('/reprocess/jobs/{job_id}', response_model=ReprocessJobInfo)
async def get_reprocess_job(job_id: str) -> ReprocessJobInfo:
    job = read_job(job_id)
    if job is None:
        raise api_error(404, 'not_found', f'no reprocess job {job_id}')
    return job


@router.post('/reprocess/jobs/{job_id}/cancel', response_model=ReprocessJobInfo)
async def cancel_reprocess_job(job_id: str) -> ReprocessJobInfo:
    """Ask a live job to stop after its current chunk."""
    if read_job(job_id) is None:
        raise api_error(404, 'not_found', f'no reprocess job {job_id}')
    cancel_job(job_id)
    job = read_job(job_id)
    assert job is not None
    return job
