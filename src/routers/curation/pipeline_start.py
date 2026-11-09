"""``POST /pipeline/auto_label/start``: resolve and pin the run's options, then queue
:func:`~src.services.curation.auto_label_run.run_auto_label` as a job."""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the query type at runtime
from typing import Annotated, Any

from fastapi import Query

from src.core.db_session import DbSessionDep  # noqa: TC001 - FastAPI resolves the dependency
from src.routers.curation._autolabel_models import AutoLabelJobState
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation.pipeline_params import (
    ACKNOWLEDGE_EXTERNAL_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_UID_DESC as _CLUSTER_UID_DESC,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    RUN_VLM_DESC as _RUN_VLM_DESC,
    VLM_ATTRIBUTES_DESC,
    VLM_DESC,
    VLM_SCOPE_DESC as _VLM_SCOPE_DESC,
)
from src.services.curation.autolabel.run_params import (
    labeler_resolution_args,
    omitted_pack_is_store_active,
    reject_detection_profile,
    resolve_run_prompt_pack,
)
from src.services.curation.autolabel.run_vlm import (
    RunVlm,
    require_item_attributes,
    require_run_vlm,
    resolve_run_vlm,
)
from src.services.curation.vlm_policy import VlmScope  # noqa: TC001 - FastAPI
from src.services.labeling.run_labeler import resolve_pack
from src.services.rejections import rejection


_EMBED_MISSING_DESC = (
    'Embed the in-scope items that were stored without a vector (not_selected, deferred or '
    'failed) before the other stages, so they are labeled too. Scoped by class_id, '
    'cluster_uid and the item filter.'
)


@router.post(
    '/pipeline/auto_label/start',
    status_code=202,
    response_model=None,
    responses={202: {'model': AutoLabelJobState}},
)
async def pipeline_auto_label_start(
    session: DbSessionDep,
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    vlm_batch_size: Annotated[int, Query(ge=4, le=64)] = 32,
    vlm_concurrency: Annotated[int, Query(ge=1, le=128)] = 16,
    max_vlm_crops: Annotated[int, Query(ge=0, le=100000)] = 0,
    classifier_confidence_skip_vlm: Annotated[float, Query(ge=0.0, le=1.0)] = 0.80,
    run_vlm: Annotated[bool, Query(description=_RUN_VLM_DESC)] = False,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_uid: Annotated[uuid.UUID | None, Query(description=_CLUSTER_UID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
    embed_missing: Annotated[bool, Query(description=_EMBED_MISSING_DESC)] = False,
    vlm_scope: Annotated[VlmScope | None, Query(description=_VLM_SCOPE_DESC)] = None,
    vlm_attributes: Annotated[bool, Query(description=VLM_ATTRIBUTES_DESC)] = False,
) -> dict[str, Any]:
    """Queue auto_label as a background job and answer 202 with its state.

    Poll ``GET /pipeline/auto_label/status/{job_id}`` (the returned
    ``job_id``) for this job's progress.
    Only one job per project is queued, running or paused at a time; a second start
    request returns 409 ``job_busy``.
    """
    from src.services.curation.autolabel import service

    # Resolved here (422 before queueing) so the job args echo what runs.
    reject_detection_profile(detection_profile)
    # captured BEFORE
    # `resolve_run_prompt_pack` overwrites `prompt_pack` below -- the raw
    # query param tells us whether this request omitted `prompt_pack`
    # entirely (the Cropwright default path), independent of what name it
    # resolves to. See ``run_auto_label``'s ``prompt_pack_omitted`` for
    # why the job needs this as its OWN signal, distinct from the echoed
    # `prompt_pack` name.
    prompt_pack_was_omitted = not isinstance(prompt_pack, str)
    prompt_pack, prompt_pack_revision = await resolve_run_prompt_pack(opensearch, prompt_pack)
    if prompt_pack_was_omitted:
        # only the store-active
        # case has TOCTOU draft risk -- a legacy-settings-doc/env/file
        # default has nothing to go stale against, so it must keep
        # running the exact pack it echoes.
        prompt_pack_was_omitted = omitted_pack_is_store_active(prompt_pack)
    # resolved, gated (SSRF, external-images ack, pack pairing) and PINNED
    # to a concrete `(name, revision)` here, before queueing -- the job then
    # resolves exactly that by id, in whichever process runs it, so an
    # activation or an endpoint edit made mid-job changes nothing for it.
    if vlm_attributes and not run_vlm:
        raise rejection(
            422, 'invalid_request', 'vlm_attributes needs run_vlm=true: it is a VLM option'
        )
    vlm_run = RunVlm(None, None, None)
    if run_vlm or isinstance(vlm, str):
        pack_name, pack_revision = labeler_resolution_args(
            prompt_pack, prompt_pack_revision, prompt_pack_omitted=prompt_pack_was_omitted
        )
        if vlm_attributes:
            require_item_attributes(resolve_pack(pack_name, pack_revision))
        vlm_run = await resolve_run_vlm(
            vlm,
            pack=resolve_pack(pack_name, pack_revision),
            acknowledge_external=acknowledge_external,
        )
        if run_vlm:
            require_run_vlm(vlm_run)
    params = {
        'vlm_batch_size': vlm_batch_size,
        'vlm_concurrency': vlm_concurrency,
        'max_vlm_crops': max_vlm_crops,
        'classifier_confidence_skip_vlm': classifier_confidence_skip_vlm,
        'run_vlm': run_vlm,
        'class_id': class_id,
        'cluster_uid': None if cluster_uid is None else str(cluster_uid),
        'item_filter': item_filter.model_dump(mode='json', exclude_defaults=True),
        'embed_missing': embed_missing,
        'vlm_scope': vlm_scope,
        'vlm_attributes': vlm_attributes,
        'prompt_pack': prompt_pack,
        'prompt_pack_revision': prompt_pack_revision,
        # always set, even for the omitted-pack default
        # case -- `/start` already ran `resolve_run_prompt_pack`
        # above for every shape of this request, so the job must
        # never re-run it.
        'prompt_pack_resolved': True,
        # Tells `run_auto_label` whether THIS request
        # omitted `prompt_pack` -- when it did, the job must
        # resolve the VLM labeler against `None` (always "whatever
        # is active right now"), never the echoed `prompt_pack`
        # name, which can go stale between `/start` and the VLM
        # stage actually running (see `run_auto_label`).
        'prompt_pack_omitted': prompt_pack_was_omitted,
        'vlm_endpoint': vlm_run.name,
        'vlm_endpoint_revision': vlm_run.revision,
        'vlm_resolved': True,
    }
    return await service.start(session, params)


__all__ = ['pipeline_auto_label_start']
