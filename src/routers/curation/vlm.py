"""VLM-based labeling endpoints.

``POST /curation/vlm/label_batch`` and ``/verify_region_batch`` queue a job (202 with its id;
the work is :mod:`src.services.curation.vlm_batch`, run by a worker);
``/region_visible_batch`` judges a sub-region-of-interest (e.g. a printed label or sticker)
from caller-supplied crops, synchronously;
per-box verification of stored boxes is the ``verify`` action of an assist run.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._common import OpenSearchDep, RegionProfileDep, router
from src.routers.curation._error_models import REGION_PROFILE_RESPONSES
from src.routers.curation._vlm_route_models import (
    VlmBatchAccepted,
    VlmLabelBatchRequest,
    VlmRegionVisibleBatchRequest,
    VlmRegionVisibleBatchResponse,
    VlmVerifyRegionBatchRequest,
)
from src.routers.curation.pipeline_params import ACKNOWLEDGE_EXTERNAL_DESC, VLM_DESC
from src.services.curation import vlm_batch, vlm_batch_jobs
from src.services.labeling.run_labeler import get_vlm_labeler, resolve_pack


async def _resolve_request(
    opensearch: Any, vlm: Any, acknowledge_external: Any
) -> tuple[str | None, Any]:
    """``(pack_name, RunVlm)`` for one VLM route call, through the shared gate."""
    from src.services.curation.autolabel.run_vlm import resolve_run_vlm

    pack_name = await _default_pack_name(opensearch)
    run = await resolve_run_vlm(
        vlm,
        pack=resolve_pack(pack_name),
        acknowledge_external=acknowledge_external,
    )
    return pack_name, run


async def request_labeler(
    opensearch: Any, vlm: Any, acknowledge_external: Any, *, usage_source: str
) -> Any:
    """The labeler for one synchronous VLM route call: the settings-default pack, and the
    endpoint the request selected (``?vlm=``) or the project default, both through the shared
    gate. 409 ``vlm_not_configured`` when the project's VLM is off."""
    from src.services.curation.autolabel.run_vlm import labeler_unavailable
    from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
    from src.services.labeling.vlm_factory import assert_may_connect

    pack_name, run = await _resolve_request(opensearch, vlm, acknowledge_external)
    try:
        if run.endpoint is not None:
            await assert_may_connect(run.endpoint)
        return get_vlm_labeler(pack_name, endpoint=run.endpoint, usage_source=usage_source)
    except VlmEndpointUnavailableError as exc:
        raise labeler_unavailable(exc) from exc


async def request_pins(opensearch: Any, vlm: Any, acknowledge_external: Any) -> dict[str, Any]:
    """Resolve and gate the request's pack and endpoint now (a 422 or 409 before queueing) and
    return what the job runs against, as pinned ``params``."""
    from src.services.curation.autolabel.run_vlm import labeler_unavailable, require_run_vlm
    from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError
    from src.services.labeling.vlm_factory import assert_may_connect

    pack_name, run = await _resolve_request(opensearch, vlm, acknowledge_external)
    require_run_vlm(run)
    try:
        await assert_may_connect(run.endpoint)
    except VlmEndpointUnavailableError as exc:
        raise labeler_unavailable(exc) from exc
    return {
        'pack_name': pack_name,
        'vlm_endpoint': run.name,
        'vlm_endpoint_revision': run.revision,
    }


async def _default_pack_name(opensearch: Any) -> str | None:
    """The ``prompt_pack`` axis's effective default: the config store's
    active pack, else the process default pack."""
    from src.services.curation.strategy_defaults import resolve_effective_default

    return await resolve_effective_default('prompt_pack', opensearch)


@router.post('/vlm/label_batch', status_code=202, response_model=VlmBatchAccepted)
async def vlm_label_batch(
    payload: VlmLabelBatchRequest,
    session: DbSessionDep,
    opensearch: OpenSearchDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> dict[str, Any]:
    """Queue up to 64 crops for the VLM; answers 202 with the job id.

    Read the outcome (``predicted``, ``updated``, ``new_class_proposals``, or the
    ``no_classes`` / ``vlm_unreachable`` flags) from the job's ``result`` at
    ``GET /jobs/{job_id}``."""
    if len(payload.crop_ids) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 crop_ids per call')
    # Resolved (and gated) before queueing: an unknown or unacknowledged endpoint is a
    # 422 up front, never a failed job.
    pins = await request_pins(opensearch, vlm, acknowledge_external)
    return await vlm_batch_jobs.start(
        session, 'vlm_label_batch', {**pins, 'crop_ids': list(payload.crop_ids)}
    )


@router.post(
    '/vlm/verify_region_batch',
    status_code=202,
    response_model=VlmBatchAccepted,
    responses=REGION_PROFILE_RESPONSES,
)
async def vlm_verify_region_batch(
    payload: VlmVerifyRegionBatchRequest,
    session: DbSessionDep,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> dict[str, Any]:
    """Queue region crops for verification; answers 202 with the job id.

    The images are staged in blob storage and the job reads them from there. The job's
    ``result.results`` lists a verdict per ``crop_id`` the VLM answered; a ``crop_id`` missing
    from it got no verdict and must be retried, never read as a reject."""
    items = payload.items
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')
    decoded: list[bytes] = []
    seen: set[str] = set()
    for item in items:
        if item.crop_id in seen:
            raise HTTPException(status_code=400, detail=f'duplicate crop_id: {item.crop_id}')
        seen.add(item.crop_id)
        try:
            jpeg_bytes = base64.b64decode(item.region_image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f'invalid base64 for crop_id={item.crop_id}: {exc}'
            ) from exc
        if not jpeg_bytes:
            raise HTTPException(
                status_code=400, detail=f'empty region_image_b64 for crop_id={item.crop_id}'
            )
        decoded.append(jpeg_bytes)
    pins = await request_pins(opensearch, vlm, acknowledge_external)
    batch_id = vlm_batch.new_batch_id()
    keys = vlm_batch.stage_images(batch_id, decoded)
    staged = [
        {'crop_id': i.crop_id, 'key': k, 'candidate_text': i.candidate_text}
        for i, k in zip(items, keys, strict=True)
    ]
    try:
        return await vlm_batch_jobs.start(
            session, 'vlm_verify_region_batch', {**pins, 'batch_id': batch_id, 'items': staged}
        )
    except BaseException:
        vlm_batch.drop_staged(batch_id)
        raise


@router.post(
    '/vlm/region_visible_batch',
    response_model=VlmRegionVisibleBatchResponse,
    responses=REGION_PROFILE_RESPONSES,
)
async def vlm_region_visible_batch(
    payload: VlmRegionVisibleBatchRequest,
    opensearch: OpenSearchDep,
    _profile: RegionProfileDep,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
) -> VlmRegionVisibleBatchResponse:
    """Pre-filter item crops by asking the VLM whether a sub-region is visible.

    Why this endpoint exists
    ------------------------
    A full region-of-interest detector is often the throughput
    bottleneck. A non-trivial fraction of item crops contain no visible
    sub-region at all. Routing those crops through the detector +
    verify pipeline is pure waste. A yes/no VLM call packed several
    crops at a time costs roughly an order of magnitude less per crop
    than a verify call.

    The endpoint returns ``{crop_id: bool}``: ``True`` means the crop
    should continue to the detector; ``False`` means the caller should
    write a terminal "not visible" status directly and skip the
    detector entirely. On an RPC failure or a garbled entry the verdict
    defaults to ``True`` (fail-open) so we never silently drop a crop that
    might have a real sub-region. A crop missing from the map got no
    verdict (the VLM answered its chunk with nothing): retry it.
    """

    items = payload.items
    if not items:
        return VlmRegionVisibleBatchResponse(visible={})
    if len(items) > 64:
        raise HTTPException(status_code=400, detail='maximum 64 items per call')

    from src.services.labeling.vlm_models import RegionCrop

    crops: list[RegionCrop] = []
    seen: set[str] = set()
    for item in items:
        if item.crop_id in seen:
            raise HTTPException(status_code=400, detail=f'duplicate crop_id: {item.crop_id}')
        seen.add(item.crop_id)
        try:
            jpeg_bytes = base64.b64decode(item.image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f'invalid base64 for crop_id={item.crop_id}: {exc}',
            ) from exc
        if not jpeg_bytes:
            raise HTTPException(
                status_code=400,
                detail=f'empty image_b64 for crop_id={item.crop_id}',
            )
        crops.append(RegionCrop(crop_id=item.crop_id, jpeg_bytes=jpeg_bytes))

    labeler = await request_labeler(
        opensearch, vlm, acknowledge_external, usage_source='region_verify'
    )
    visible = await labeler.region_visible_batch(crops)
    return VlmRegionVisibleBatchResponse(visible=visible)


__all__ = ['get_vlm_labeler', 'request_labeler']
