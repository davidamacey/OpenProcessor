"""Routes of the research config kinds: serving profiles (deployment-wide), cluster profiles and
assist-loop presets (per project).

Registered on the projects ``global_router`` and the curation ``router``; every call goes through
:mod:`src.services.config_store.research_kinds`. A save names the revision it was made against
(``expected_revision``, ``null`` to create), an activation names what it expects to replace; a stale
one is a 409.
"""

from __future__ import annotations

from typing import Any

from fastapi import Query, Response

from src.routers.curation._api_error import active_conflict_error, api_error
from src.routers.curation._common import router
from src.routers.curation.projects import global_router
from src.schemas.curation.research_config import (
    ResearchActivateRequest,
    ResearchActive,
    ResearchConfigDoc,
    ResearchConfigList,
    ResearchConfigPut,
    ServingActivateRequest,
    ServingActive,
)
from src.services.config_store.research_kinds import (
    KIND_AXES,
    ActiveConflictError,
    InvalidConfigBodyError,
    RevisionConflictError,
    UnknownConfigError,
    activate_project_axis,
    activate_serving_profile,
    active_serving_profile,
    delete_document,
    get_active,
    get_document,
    list_documents,
    save_document,
    wire_doc,
)


def _conflict(name: str, exc: RevisionConflictError) -> Any:
    return api_error(
        409,
        'revision_conflict',
        f'{name!r} is at revision {exc.current_revision}',
        current_revision=exc.current_revision,
    )


def _invalid(exc: InvalidConfigBodyError) -> Any:
    reasons = [f'{".".join(str(part) for part in e["loc"])}: {e["msg"]}' for e in exc.errors]
    return api_error(422, 'validation_failed', str(exc), reasons=reasons)


def _register_documents(kind: str, path: str, *, tag: str, group: Any) -> None:
    """List, get, save and delete of one kind on ``group`` (a router)."""

    @group.get(f'/{path}', response_model=ResearchConfigList, tags=[tag], name=f'list_{kind}')
    async def list_all() -> ResearchConfigList:
        return ResearchConfigList(items=[wire_doc(d) for d in await list_documents(kind)])

    @group.get(
        f'/{path}/{{name}}', response_model=ResearchConfigDoc, tags=[tag], name=f'get_{kind}'
    )
    async def get_one(
        name: str, revision: int | None = Query(default=None, ge=1)
    ) -> ResearchConfigDoc:
        doc = await get_document(kind, name, revision)
        if doc is None:
            raise api_error(404, 'not_found', f'no {kind} {name!r}')
        return wire_doc(doc)

    @group.put(
        f'/{path}/{{name}}', response_model=ResearchConfigDoc, tags=[tag], name=f'put_{kind}'
    )
    async def put_one(name: str, body: ResearchConfigPut) -> ResearchConfigDoc:
        try:
            doc = await save_document(
                kind,
                name,
                body.body,
                expected_revision=body.expected_revision,
                description=body.description,
            )
        except RevisionConflictError as exc:
            raise _conflict(name, exc) from exc
        except InvalidConfigBodyError as exc:
            raise _invalid(exc) from exc
        return wire_doc(doc)

    @group.delete(f'/{path}/{{name}}', status_code=204, tags=[tag], name=f'delete_{kind}')
    async def delete_one(name: str, expected_revision: int = Query(ge=1)) -> Response:
        try:
            await delete_document(kind, name, expected_revision=expected_revision)
        except KeyError as exc:
            raise api_error(404, 'not_found', f'no {kind} {name!r}') from exc
        except RevisionConflictError as exc:
            raise _conflict(name, exc) from exc
        return Response(status_code=204)


def _register_project_activation(kind: str, path: str, axes: tuple[str, ...], *, tag: str) -> None:
    @router.get(f'/{path}/active', response_model=ResearchActive, tags=[tag], name=f'active_{kind}')
    async def active(axis: str = Query(default=axes[0])) -> ResearchActive:
        if axis not in axes:
            raise api_error(422, 'validation_failed', f'axis must be one of {list(axes)}')
        ref = await get_active(axis)
        if ref is None:
            return ResearchActive(axis=axis, state='none')
        return ResearchActive(
            axis=axis,
            state=ref.state,
            name=ref.name,
            revision=ref.revision,
            activated_at=None if ref.activated_at is None else ref.activated_at.isoformat(),
        )

    @router.post(
        f'/{path}/{{name}}/activate',
        response_model=ResearchActive,
        tags=[tag],
        name=f'activate_{kind}',
    )
    async def activate(name: str, body: ResearchActivateRequest) -> ResearchActive:
        axis = body.axis or axes[0]
        if axis not in axes:
            raise api_error(422, 'validation_failed', f'axis must be one of {list(axes)}')
        expected = None if body.expected_active is None else body.expected_active.model_dump()
        try:
            ref = await activate_project_axis(
                axis, name, revision=body.revision, expected_active=expected
            )
        except UnknownConfigError as exc:
            raise api_error(404, 'not_found', str(exc)) from exc
        except ActiveConflictError as exc:
            raise active_conflict_error(f'the active {kind}', exc.current) from exc
        return ResearchActive(
            axis=axis,
            state=ref.state,
            name=ref.name,
            revision=ref.revision,
            activated_at=None if ref.activated_at is None else ref.activated_at.isoformat(),
        )


@global_router.get('/serving_profiles/active', response_model=ServingActive, tags=['Serving'])
async def get_active_serving_profile() -> ServingActive:
    found = await active_serving_profile()
    if found is None:
        raise api_error(404, 'not_found', 'no serving profile is active')
    doc, setting_revision = found
    return ServingActive(
        name=doc.name, revision=doc.revision, setting_revision=setting_revision, body=doc.body
    )


@global_router.post(
    '/serving_profiles/{name}/activate', response_model=ServingActive, tags=['Serving']
)
async def activate_serving_profile_route(name: str, body: ServingActivateRequest) -> ServingActive:
    try:
        setting = await activate_serving_profile(
            name, revision=body.revision, expected_revision=body.expected_revision
        )
    except UnknownConfigError as exc:
        raise api_error(404, 'not_found', str(exc)) from exc
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'the active serving profile is at revision {exc.current_revision}',
            current_revision=exc.current_revision,
        ) from exc
    found = await active_serving_profile()
    if found is None:  # written by this call
        msg = 'the serving activation was not stored'
        raise RuntimeError(msg)
    doc, _ = found
    return ServingActive(
        name=doc.name, revision=doc.revision, setting_revision=setting.revision, body=doc.body
    )


# `/{kind}/active` is registered before `/{kind}/{name}`: the literal path must win.
_register_project_activation(
    'cluster_profile', 'cluster_profiles', KIND_AXES['cluster_profile'], tag='Clustering'
)
_register_project_activation(
    'assist_loop_preset', 'assist_loop_presets', KIND_AXES['assist_loop_preset'], tag='Auto-label'
)
_register_project_activation(
    'near_dup_profile', 'near_dup_profiles', KIND_AXES['near_dup_profile'], tag='Near-duplicates'
)
_register_documents('serving_profile', 'serving_profiles', tag='Serving', group=global_router)
_register_documents('cluster_profile', 'cluster_profiles', tag='Clustering', group=router)
_register_documents('assist_loop_preset', 'assist_loop_presets', tag='Auto-label', group=router)
_register_documents('near_dup_profile', 'near_dup_profiles', tag='Near-duplicates', group=router)
