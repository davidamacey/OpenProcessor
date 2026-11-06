"""``/vlm/endpoints/active``, ``/{name}/activate``, ``/active/rollback`` and
``/deactivate``: which endpoint the BOUND project's VLM runs.

The registry is global (:mod:`vlm_endpoints`); activation is per project.
Every route here writes through
:mod:`src.services.config_store.vlm_activation`, which runs the one shared
gate (:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate`) first.
"""

from __future__ import annotations

from typing import Any, Literal

from src.routers.curation._api_error import active_conflict_error, api_error
from src.routers.curation._common import router
from src.routers.curation._vlm_endpoint_models import (
    VlmActivateRequest,
    VlmActiveResponse,
    VlmDeactivateRequest,
    VlmRollbackRequest,
)
from src.schemas.curation.config_common import ActiveRef
from src.services.config_store import documents, get_config_store
from src.services.config_store.activation_view import applied_runtime
from src.services.config_store.documents import ActiveConflictError
from src.services.config_store.vlm_activation import activate_vlm, deactivate_vlm, rollback_vlm
from src.services.labeling.vlm_endpoints import refresh_vlm_state


_PREFIX = '/vlm/endpoints'


def _conflict(exc: ActiveConflictError) -> Any:
    return active_conflict_error('the active VLM', exc.current)


def _dump(ref: ActiveRef | None) -> dict[str, Any] | None:
    return None if ref is None else {'name': ref.name, 'revision': ref.revision}


async def active_response(*, validation: Any = None) -> VlmActiveResponse:
    """The bound project's VLM activation, with what each detection worker
    actually applied."""
    await refresh_vlm_state()
    snapshot = get_config_store().current
    ref = snapshot.active_vlm
    if ref is None or ref == 'off':
        active = ActiveRef()
        source = 'off'
    else:
        active = ActiveRef(name=ref[0], revision=ref[1])
        source = 'env' if ref[0] == 'env' else 'stored'
    detail = await documents.get_activation_detail('vlm') if source != 'env' else None
    previous = detail.previous if detail is not None else None
    try:
        reports = await documents.live_runtime_reports(process='detection_worker')
    except Exception:  # pragma: no cover - applied[] degrades to empty
        reports = []
    return VlmActiveResponse(
        axis='vlm',
        active=active,
        source=source,
        activated_at=detail.activated_at if detail is not None else None,
        previous=ActiveRef(name=previous[0], revision=previous[1]) if previous else None,
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        applied=[applied_runtime(r, snapshot) for r in reports],
        validation=validation,
    )


@router.get(f'{_PREFIX}/active', response_model=VlmActiveResponse, tags=['VLM'])
async def get_active_vlm() -> VlmActiveResponse:
    """This project's active VLM endpoint (``active.name: null`` = off)."""
    return await active_response()


@router.post(f'{_PREFIX}/active/rollback', response_model=VlmActiveResponse, tags=['VLM'])
async def rollback_active_vlm(
    payload: VlmRollbackRequest,
) -> VlmActiveResponse:
    """Re-activate the previous endpoint through the same gate."""
    try:
        _result, gate = await rollback_vlm(expected_active=_dump(payload.expected_active))
    except LookupError as exc:
        code: Literal['previous_deleted', 'no_previous'] = (
            'previous_deleted' if str(exc) == 'previous_deleted' else 'no_previous'
        )
        raise api_error(409, code, 'there is no previous VLM to roll back to') from exc
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response(validation=gate.report if gate else None)


@router.post(f'{_PREFIX}/deactivate', response_model=VlmActiveResponse, tags=['VLM'])
async def deactivate_active_vlm(
    payload: VlmDeactivateRequest,
) -> VlmActiveResponse:
    """VLM off for this project."""
    try:
        await deactivate_vlm(expected_active=_dump(payload.expected_active))
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response()


@router.post(f'{_PREFIX}/{{name}}/activate', response_model=VlmActiveResponse, tags=['VLM'])
async def activate_vlm_endpoint(
    name: str,
    payload: VlmActivateRequest,
) -> VlmActiveResponse:
    """Activate ``name`` (at ``revision``, else its latest) for this project.
    An endpoint outside this deployment needs ``acknowledge_external``."""
    try:
        _result, gate = await activate_vlm(
            name=name,
            revision=payload.revision,
            expected_active=_dump(payload.expected_active),
            mode='activate',
            force=payload.force,
            acknowledge_external=payload.acknowledge_external,
        )
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response(validation=gate.report)
