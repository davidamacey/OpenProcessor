"""``POST /pipeline/auto_label`` -- split out of ``pipeline.py`` to stay
under the repo's 700-LOC pre-commit ratchet (same pattern as
``_models_sharing.py``/``models.py`` and ``_region_profile_clone.py``/
``region_profiles.py``).

R6-m1 fix (Minor, W3/W4 round-6 review): this is the thin, PUBLIC route
wrapper around :func:`~src.routers.curation.pipeline._run_auto_label`.
``prompt_pack_revision``/``prompt_pack_resolved`` used to be hidden
(``include_in_schema=False``) query params directly on the combined
route -- hidden from the OpenAPI docs but NOT off the wire
(``?prompt_pack=nope&prompt_pack_resolved=true`` skipped the
unknown-pack 422 ``_run_auto_label`` runs). Splitting the internal
implementation (called directly by the worker, and by ``/start``'s own
resolved pin) from this public route -- which always forces
``prompt_pack_resolved=False``, ``prompt_pack_revision=None`` -- makes
that no longer reachable from any request at all.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation.pipeline_params import (
    AUTO_PROMOTE_DESC as _AUTO_PROMOTE_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_ID_DESC as _CLUSTER_ID_DESC,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    REASSIGN_ONLY_DESC as _REASSIGN_ONLY_DESC,
)
from src.services.curation.cluster_purity import PROMOTE_MIN_MEMBERS, PROMOTE_MIN_PURITY


@router.post('/pipeline/auto_label')
async def pipeline_auto_label(
    opensearch: OpenSearchDep,
    train_clusters: Annotated[
        bool, Query(description='Re-train clusters on the items index before promote/VLM.')
    ] = True,
    promote_min_purity: Annotated[float, Query(ge=0.5, le=1.0)] = PROMOTE_MIN_PURITY,
    promote_min_members: Annotated[int, Query(ge=2, le=1000)] = PROMOTE_MIN_MEMBERS,
    vlm_batch_size: Annotated[int, Query(ge=4, le=64)] = 32,
    vlm_concurrency: Annotated[int, Query(ge=1, le=128)] = 8,
    max_vlm_crops: Annotated[int, Query(ge=0, le=100000, description='0 = all in scope')] = 0,
    classifier_confidence_skip_vlm: Annotated[
        float,
        Query(
            ge=0.0,
            le=1.0,
            description=(
                'Skip the VLM for classifier-labeled items at or above this confidence '
                '(global sweep only; a cluster-scoped run labels every unvalidated member).'
            ),
        ),
    ] = 0.80,
    clustering_method: Annotated[
        str | None,
        Query(description='Residual-pool clusterer id from GET /methods (axis=cluster).'),
    ] = None,
    run_vlm: Annotated[bool, Query(description='Run the VLM stage. See /start.')] = False,
    recluster_unvalidated: Annotated[bool, Query(description='Merge candidate clusters.')] = False,
    reassign_only: Annotated[bool, Query(description=_REASSIGN_ONLY_DESC)] = False,
    run_auto_promote: Annotated[bool, Query(description=_AUTO_PROMOTE_DESC)] = False,
    gate_max_rank: Annotated[int | None, Query(ge=1)] = None,
    gate_min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None,
    n_clusters: Annotated[int | None, Query(ge=2, le=4096)] = None,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_id: Annotated[int | None, Query(description=_CLUSTER_ID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
) -> dict[str, Any]:
    """Public ``POST /pipeline/auto_label`` route (synchronous, no job).

    A thin wrapper around ``_run_auto_label`` that always calls it with
    ``prompt_pack_resolved=False`` and ``prompt_pack_revision=None`` --
    those are real Python-only params now, set only by ``/start``'s own
    resolved pin and by the worker re-invoking ``_run_auto_label``
    directly -- never settable by an HTTP request.
    """
    from src.routers.curation.pipeline import _run_auto_label

    return await _run_auto_label(
        opensearch=opensearch,
        train_clusters=train_clusters,
        promote_min_purity=promote_min_purity,
        promote_min_members=promote_min_members,
        vlm_batch_size=vlm_batch_size,
        vlm_concurrency=vlm_concurrency,
        max_vlm_crops=max_vlm_crops,
        classifier_confidence_skip_vlm=classifier_confidence_skip_vlm,
        clustering_method=clustering_method,
        run_vlm=run_vlm,
        recluster_unvalidated=recluster_unvalidated,
        reassign_only=reassign_only,
        run_auto_promote=run_auto_promote,
        gate_max_rank=gate_max_rank,
        gate_min_blur_ratio=gate_min_blur_ratio,
        n_clusters=n_clusters,
        class_id=class_id,
        cluster_id=cluster_id,
        detection_profile=detection_profile,
        prompt_pack=prompt_pack,
        prompt_pack_revision=None,
        prompt_pack_resolved=False,
    )


__all__ = ['pipeline_auto_label']
