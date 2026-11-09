"""``POST /pipeline/auto_label``: the thin, PUBLIC route wrapper around
:func:`~src.services.curation.auto_label_run.run_auto_label`.
``prompt_pack_revision``/``prompt_pack_resolved`` used to be hidden
(``include_in_schema=False``) query params directly on the combined
route -- hidden from the OpenAPI docs but NOT off the wire
(``?prompt_pack=nope&prompt_pack_resolved=true`` skipped the
unknown-pack 422 ``run_auto_label`` runs). Splitting the internal
implementation (called directly by the worker, and by ``/start``'s own
resolved pin) from this public route -- which always forces
``prompt_pack_resolved=False``, ``prompt_pack_revision=None`` -- makes
that no longer reachable from any request at all.
"""

from __future__ import annotations

import uuid  # noqa: TC003 - FastAPI resolves the query type at runtime
from typing import Annotated, Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation.pipeline_params import (
    ACKNOWLEDGE_EXTERNAL_DESC,
    CLASS_ID_DESC as _CLASS_ID_DESC,
    CLUSTER_UID_DESC as _CLUSTER_UID_DESC,
    PROMPT_PACK_DESC as _PROMPT_PACK_DESC,
    VLM_ATTRIBUTES_DESC,
    VLM_DESC,
    VLM_SCOPE_DESC as _VLM_SCOPE_DESC,
)
from src.services.curation.auto_label_run import run_auto_label
from src.services.curation.vlm_policy import VlmScope  # noqa: TC001 - FastAPI


# Internal: a thin wrapper around `run_auto_label` -- see that function
# and the module docstring above for the Python-only params
# (`prompt_pack_resolved`/`prompt_pack_revision`) this always forces,
# never settable by an HTTP request. Kept as a comment (not the
# docstring below) since the docstring is the public OpenAPI
# `description` -- it must describe only the public contract.
@router.post('/pipeline/auto_label')
async def pipeline_auto_label(
    opensearch: OpenSearchDep,
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
    run_vlm: Annotated[bool, Query(description='Run the VLM stage. See /start.')] = False,
    class_id: Annotated[int | None, Query(description=_CLASS_ID_DESC)] = None,
    cluster_uid: Annotated[uuid.UUID | None, Query(description=_CLUSTER_UID_DESC)] = None,
    detection_profile: Annotated[str | None, Query(include_in_schema=False)] = None,
    prompt_pack: Annotated[str | None, Query(description=_PROMPT_PACK_DESC)] = None,
    vlm: Annotated[str | None, Query(description=VLM_DESC)] = None,
    acknowledge_external: Annotated[bool, Query(description=ACKNOWLEDGE_EXTERNAL_DESC)] = False,
    vlm_scope: Annotated[VlmScope | None, Query(description=_VLM_SCOPE_DESC)] = None,
    vlm_attributes: Annotated[bool, Query(description=VLM_ATTRIBUTES_DESC)] = False,
) -> dict[str, Any]:
    """Run the auto-labeling chain synchronously (no job): the optional embed-missing stage,
    then the VLM over the unvalidated items with the open-vocabulary prompt (high-confidence
    labels are auto-validated, ``__new__`` proposals are flagged for the curator queue).

    Output enumerates each stage's counts so the labeler dashboard can show
    "this many items still need a human." Idempotent: safe to re-run.

    For a background job with progress polling, see ``POST
    /pipeline/auto_label/start`` instead.
    """
    return await run_auto_label(
        opensearch=opensearch,
        vlm_batch_size=vlm_batch_size,
        vlm_concurrency=vlm_concurrency,
        max_vlm_crops=max_vlm_crops,
        classifier_confidence_skip_vlm=classifier_confidence_skip_vlm,
        run_vlm=run_vlm,
        class_id=class_id,
        cluster_uid=cluster_uid,
        detection_profile=detection_profile,
        prompt_pack=prompt_pack,
        prompt_pack_revision=None,
        prompt_pack_resolved=False,
        vlm=vlm,
        acknowledge_external=acknowledge_external,
        vlm_scope=vlm_scope,
        vlm_attributes=vlm_attributes,
        vlm_resolved=False,
    )


__all__ = ['pipeline_auto_label']
