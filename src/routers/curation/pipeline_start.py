"""``POST /pipeline/auto_label/start`` -- split out of ``pipeline.py`` to
stay under the repo's 700-LOC pre-commit ratchet (same pattern as
``_region_profile_clone.py``/``region_profiles.py`` and this package's
own ``pipeline_public.py``).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation.pipeline_params import (
    AUTO_PROMOTE_DESC as _AUTO_PROMOTE_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_ID_DESC as _CLUSTER_ID_DESC,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    REASSIGN_ONLY_DESC as _REASSIGN_ONLY_DESC,
    RUN_VLM_DESC as _RUN_VLM_DESC,
    omitted_pack_is_store_active,
    reject_detection_profile,
    resolve_run_prompt_pack,
)
from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, PROMOTE_MIN_PURITY


@router.post('/pipeline/auto_label/start')
async def pipeline_auto_label_start(
    opensearch: OpenSearchDep,
    train_clusters: Annotated[bool, Query()] = True,
    promote_min_purity: Annotated[float, Query(ge=0.5, le=1.0)] = PROMOTE_MIN_PURITY,
    promote_min_members: Annotated[int, Query(ge=2, le=1000)] = PROMOTE_MIN_MEMBERS,
    vlm_batch_size: Annotated[int, Query(ge=4, le=64)] = 32,
    vlm_concurrency: Annotated[int, Query(ge=1, le=128)] = 16,
    max_vlm_crops: Annotated[int, Query(ge=0, le=100000)] = 0,
    classifier_confidence_skip_vlm: Annotated[float, Query(ge=0.0, le=1.0)] = 0.80,
    clustering_method: Annotated[str | None, Query()] = None,
    run_vlm: Annotated[bool, Query(description=_RUN_VLM_DESC)] = False,
    recluster_unvalidated: Annotated[bool, Query(description='Merge candidate clusters.')] = False,
    run_auto_promote: Annotated[bool, Query(description=_AUTO_PROMOTE_DESC)] = False,
    reassign_only: Annotated[bool, Query(description=_REASSIGN_ONLY_DESC)] = False,
    # Cluster scope (primary-subject gate).
    gate_max_rank: Annotated[int | None, Query(ge=1)] = None,
    gate_min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    n_clusters: Annotated[int | None, Query(ge=2, le=4096)] = None,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_id: Annotated[int | None, Query(description=_CLUSTER_ID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
) -> dict[str, Any]:
    """Kick off auto_label as a background job. Returns immediately.

    Poll ``GET /pipeline/auto_label/status/{job_id}`` (the returned
    ``job_id``) for this job's progress.
    Only one job runs at a time; a second start request returns HTTP 409
    while a job is in flight.
    """
    from src.routers.curation.pipeline import _run_auto_label
    from src.services.curation.autolabel import job as auto_label_job

    # Resolved here (422 before queueing) so the job args echo what runs.
    reject_detection_profile(detection_profile)
    # R6-1b fix (Blocker, W3/W4 round-6 review): captured BEFORE
    # `resolve_run_prompt_pack` overwrites `prompt_pack` below -- the raw
    # query param tells us whether this request omitted `prompt_pack`
    # entirely (the Cropwright default path), independent of what name it
    # resolves to. See ``_run_auto_label``'s ``prompt_pack_omitted`` for
    # why the job needs this as its OWN signal, distinct from the echoed
    # `prompt_pack` name.
    prompt_pack_was_omitted = not isinstance(prompt_pack, str)
    prompt_pack, prompt_pack_revision = await resolve_run_prompt_pack(opensearch, prompt_pack)
    if prompt_pack_was_omitted:
        # R7-2 fix (Major, W3/W4 round-7 review): only the store-active
        # case has TOCTOU draft risk -- a legacy-settings-doc/env/file
        # default has nothing to go stale against, so it must keep
        # running the exact pack it echoes.
        prompt_pack_was_omitted = omitted_pack_is_store_active(prompt_pack)
    try:
        return auto_label_job.start_job(
            # R6-m1 fix: the worker must invoke the internal implementation
            # directly, not the public route wrapper -- `pipeline_auto_label`
            # now always forces `prompt_pack_resolved=False`, which would
            # silently re-resolve (and potentially re-pin) every job.
            _run_auto_label,
            {
                'opensearch': opensearch,
                'train_clusters': train_clusters,
                'promote_min_purity': promote_min_purity,
                'promote_min_members': promote_min_members,
                'vlm_batch_size': vlm_batch_size,
                'vlm_concurrency': vlm_concurrency,
                'max_vlm_crops': max_vlm_crops,
                'classifier_confidence_skip_vlm': classifier_confidence_skip_vlm,
                'clustering_method': clustering_method,
                'run_vlm': run_vlm,
                'recluster_unvalidated': recluster_unvalidated,
                'run_auto_promote': run_auto_promote,
                'reassign_only': reassign_only,
                'gate_max_rank': gate_max_rank,
                'gate_min_blur_ratio': gate_min_blur_ratio,
                'n_clusters': n_clusters,
                'class_id': class_id,
                'cluster_id': cluster_id,
                'prompt_pack': prompt_pack,
                'prompt_pack_revision': prompt_pack_revision,
                # R5-2 fix: always set, even for the omitted-pack default
                # case -- `/start` already ran `resolve_run_prompt_pack`
                # above for every shape of this request, so the job must
                # never re-run it.
                'prompt_pack_resolved': True,
                # R6-1b fix: tells `_run_auto_label` whether THIS request
                # omitted `prompt_pack` -- when it did, the job must
                # resolve the VLM labeler against `None` (always "whatever
                # is active right now"), never the echoed `prompt_pack`
                # name, which can go stale between `/start` and the VLM
                # stage actually running (see `_run_auto_label`).
                'prompt_pack_omitted': prompt_pack_was_omitted,
            },
        )
    except RuntimeError as exc:
        # 409 makes it unambiguous in the UI that a run is already in flight.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


__all__ = ['pipeline_auto_label_start']
