"""``/vlm/endpoints/active``, ``/{name}/activate``, ``/active/rollback`` and
``/deactivate``: which endpoint the BOUND project's VLM runs (W9.3).

The registry is global (:mod:`vlm_endpoints`); activation is per project.
Every route here writes through
:mod:`src.services.config_store.vlm_activation`, which runs the one shared
gate (:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate`) first.
"""

from __future__ import annotations

from typing import Any, Literal

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._config_common_models import ActiveRef, AppliedRuntime, api_error
from src.routers.curation._vlm_endpoint_models import (
    VlmActivateRequest,
    VlmActiveResponse,
    VlmDeactivateRequest,
    VlmRollbackRequest,
)
from src.services.config_store import get_config_store
from src.services.config_store.index import ActiveConflictError, get_activation, get_runtime_docs
from src.services.config_store.vlm_activation import activate_vlm, deactivate_vlm, rollback_vlm
from src.services.labeling.vlm_endpoints import env_builtin, refresh_vlm_state


_PREFIX = '/vlm/endpoints'


def _conflict(exc: ActiveConflictError) -> Any:
    return api_error(
        409,
        'active_conflict',
        'the VLM was activated by another writer since this request started',
        current=exc.current,
    )


def _dump(ref: ActiveRef | None) -> dict[str, Any] | None:
    return None if ref is None else {'name': ref.name, 'revision': ref.revision}


async def active_response(client: Any, *, validation: Any = None) -> VlmActiveResponse:
    """The bound project's VLM activation, with what each detection worker
    actually applied."""
    await refresh_vlm_state(client)
    store = get_config_store()
    snapshot = store.current
    ref = snapshot.active_vlm
    if ref is None:
        env = env_builtin()
        active = ActiveRef(name=env.name if env else None)
        source = 'env'
    elif ref == 'off':
        active = ActiveRef()
        source = 'off'
    else:
        active = ActiveRef(name=ref[0], revision=ref[1])
        source = 'env' if ref[0] == 'env' else 'stored'
    doc = await get_activation(client, store.index, 'vlm') if source != 'env' else None
    prev = (doc or {}).get('previous')
    applied: list[AppliedRuntime] = []
    try:
        runtime_docs = await get_runtime_docs(client, store.index, process='detection_worker')
    except Exception:  # pragma: no cover - applied[] degrades to empty
        runtime_docs = []
    for rt in runtime_docs:
        rev = int(rt.get('applied_config_revision') or 0)
        applied.append(
            AppliedRuntime(
                process=rt.get('process', 'detection_worker'),
                host=rt.get('host', ''),
                applied_config_revision=rev,
                profile=ActiveRef(name=rt.get('profile'), revision=rt.get('profile_revision')),
                pack=ActiveRef(name=rt.get('pack'), revision=rt.get('pack_revision')),
                vlm=ActiveRef(name=rt.get('vlm'), revision=rt.get('vlm_revision')),
                applied_at=rt.get('applied_at'),
                lagging=rev < snapshot.config_revision,
            )
        )
    return VlmActiveResponse(
        axis='vlm',
        active=active,
        source=source,
        activated_at=(doc or {}).get('activated_at'),
        previous=ActiveRef(name=prev.get('name'), revision=prev.get('revision')) if prev else None,
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        applied=applied,
        validation=validation,
    )


@router.get(f'{_PREFIX}/active', response_model=VlmActiveResponse, tags=['VLM'])
async def get_active_vlm(client: OpenSearchDep) -> VlmActiveResponse:
    """This project's active VLM endpoint (``active.name: null`` = off)."""
    return await active_response(client)


@router.post(f'{_PREFIX}/active/rollback', response_model=VlmActiveResponse, tags=['VLM'])
async def rollback_active_vlm(
    payload: VlmRollbackRequest, client: OpenSearchDep
) -> VlmActiveResponse:
    """Re-activate the previous endpoint through the same gate."""
    try:
        _result, gate = await rollback_vlm(client, expected_active=_dump(payload.expected_active))
    except LookupError as exc:
        code: Literal['previous_deleted', 'no_previous'] = (
            'previous_deleted' if str(exc) == 'previous_deleted' else 'no_previous'
        )
        raise api_error(409, code, 'there is no previous VLM to roll back to') from exc
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response(client, validation=gate.report if gate else None)


@router.post(f'{_PREFIX}/deactivate', response_model=VlmActiveResponse, tags=['VLM'])
async def deactivate_active_vlm(
    payload: VlmDeactivateRequest, client: OpenSearchDep
) -> VlmActiveResponse:
    """VLM off for this project."""
    try:
        await deactivate_vlm(client, expected_active=_dump(payload.expected_active))
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response(client)


@router.post(f'{_PREFIX}/{{name}}/activate', response_model=VlmActiveResponse, tags=['VLM'])
async def activate_vlm_endpoint(
    name: str, payload: VlmActivateRequest, client: OpenSearchDep
) -> VlmActiveResponse:
    """Activate ``name`` (at ``revision``, else its latest) for this project.
    An endpoint outside this deployment needs ``acknowledge_external``."""
    try:
        _result, gate = await activate_vlm(
            client,
            name=name,
            revision=payload.revision,
            expected_active=_dump(payload.expected_active),
            mode='activate',
            force=payload.force,
            acknowledge_external=payload.acknowledge_external,
        )
    except ActiveConflictError as exc:
        raise _conflict(exc) from exc
    return await active_response(client, validation=gate.report)
