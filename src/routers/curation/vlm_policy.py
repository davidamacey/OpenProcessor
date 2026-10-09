"""Curation router sub-module: the per-project VLM scope policy.

``GET/PUT /vlm/policy`` read and replace which crops the automated VLM class
writers (the continuous worker and the ``auto_label`` VLM stage) may label, and
their daily budget (see :mod:`src.services.curation.vlm_policy`). Explicit
requests (``/vlm/label_cluster/{id}``, a ``cluster_id``-scoped run) are not limited.
"""

from __future__ import annotations

from src.routers.curation._common import OpenSearchDep, _ensure_indexes, router
from src.routers.curation._config_common_models import ApiErrorResponse, api_error
from src.routers.curation._vlm_policy_models import VlmPolicyUpdate  # noqa: TC001 - FastAPI
from src.services.curation.policy_doc_store import PolicyConflictError
from src.services.curation.vlm_policy import VlmPolicy, VlmPolicyBody
from src.services.curation.vlm_policy_store import get_vlm_policy, put_vlm_policy


@router.get('/vlm/policy', response_model=VlmPolicy)
async def get_policy(opensearch: OpenSearchDep) -> VlmPolicy:
    """The project's VLM scope policy; scope ``all`` (every eligible crop) when
    none was ever written."""
    await _ensure_indexes(opensearch)
    return await get_vlm_policy(opensearch)


@router.put('/vlm/policy', response_model=VlmPolicy, responses={409: {'model': ApiErrorResponse}})
async def put_policy(body: VlmPolicyUpdate, opensearch: OpenSearchDep) -> VlmPolicy:
    """Replace the policy. ``409`` when ``expected_revision`` is not the stored
    revision (re-read and retry). Takes effect on the worker's next poll (within
    about 30 seconds) and on the next ``auto_label`` run; stored labels are untouched."""
    await _ensure_indexes(opensearch)
    policy_body = VlmPolicyBody(**body.model_dump(exclude={'expected_revision'}))
    try:
        return await put_vlm_policy(
            opensearch, policy_body, expected_revision=body.expected_revision
        )
    except PolicyConflictError as exc:
        raise api_error(409, 'revision_conflict', f'vlm policy changed: {exc}') from exc
