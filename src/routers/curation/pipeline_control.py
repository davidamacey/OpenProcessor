"""Curation auto-label job control endpoints (status, cancel, and the
cluster-scoped VLM run).

Split out of :mod:`pipeline` to keep that module under the 700-LOC hook
ceiling. Side-effect import: registers ``@router`` handlers on the shared
``_common.router``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation.pipeline_params import PROMPT_PACK_DESC


@router.get('/pipeline/auto_label/status')
async def pipeline_auto_label_status() -> dict[str, Any]:
    """Poll the current/last auto_label job state."""
    from src.services.curation.autolabel import job as auto_label_job

    return auto_label_job.get_state()


@router.post('/pipeline/auto_label/cancel')
async def pipeline_auto_label_cancel() -> dict[str, Any]:
    """Request cancellation of the active auto_label job."""
    from src.services.curation.autolabel import job as auto_label_job

    cancelled = auto_label_job.cancel_job()
    return {'cancelled': cancelled, **auto_label_job.get_state()}


@router.post('/vlm/label_cluster/{cluster_id}')
async def vlm_label_cluster(
    cluster_id: int,
    opensearch: OpenSearchDep,
    prompt_pack: Annotated[str | None, Query(description=PROMPT_PACK_DESC)] = None,
) -> dict[str, Any]:
    """VLM-label every unvalidated member of one cluster, as a background job.

    Queues the auto-label job scoped to ``cluster_id`` with only the VLM
    stage (no re-clustering, no auto-promote, no cap): the server selects
    every unvalidated, non-holdout, non-excluded member and chunks them.
    Returns the job state; poll ``GET /pipeline/auto_label/status`` —
    ``total`` is the number selected, ``result.stages.vlm`` the outcome.
    ``409`` while another auto-label job runs.
    """
    from src.routers.curation.pipeline import pipeline_auto_label_start

    return await pipeline_auto_label_start(
        opensearch,
        train_clusters=False,
        run_vlm=True,
        run_auto_promote=False,
        max_vlm_crops=0,
        cluster_id=cluster_id,
        prompt_pack=prompt_pack,
    )
