"""``/ingest_profiles*``: named, revisioned ingest detector profiles (config axis
``ingest_profile``). The active profile is the project's ingest detector; with
none active the deployment's ``OP_INGEST_*`` profile applies.

Route-order note: ``schema``, ``validate``, ``active`` (+ its children) and
``deactivate`` are declared before ``/{name}`` and its children.
``POST /ingest_profiles/{name}/clone`` lives in ``_ingest_profile_clone.py``
(imported at the bottom for its registration side effect).
"""

from __future__ import annotations

from typing import Any

from fastapi import Query, Response

from src.core.wire_json import WireJSONResponse
from src.routers.curation._api_error import active_conflict_error, api_error
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._ingest_profile_models import (
    IngestProfileActivateRequest,
    IngestProfileActivateResponse,
    IngestProfileCreateRequest,
    IngestProfileDeactivateRequest,
    IngestProfileDoc,
    IngestProfileFieldSchema,
    IngestProfileList,
    IngestProfileRevisionsResponse,
    IngestProfileRevisionSummary,
    IngestProfileRollbackRequest,
    IngestProfileSaveRequest,
    IngestProfileSchema,
    IngestProfileSummary,
    IngestProfileTemplateSummary,
    IngestProfileValidateRequest,
)
from src.schemas.curation.config_common import ActiveConfigResponse, ActiveRef, ValidationReport
from src.schemas.curation.ingest_profile import IngestProfileBody
from src.services.config_store import ActiveConflictError, RevisionConflictError, get_config_store
from src.services.config_store.activation_view import build_active_config_response
from src.services.config_store.ingest_profile_validation import (
    taken_message,
    validate_ingest_profile,
)
from src.services.config_store.ingest_profiles import (
    IngestProfileRecord,
    activate_profile,
    all_known_names,
    build_record,
    delete_profile,
    get_revision_record,
    list_revisions,
    rollback_profile,
    save_profile,
    template_names,
)


def _to_doc(record: IngestProfileRecord, *, validation: Any = None) -> IngestProfileDoc:
    return IngestProfileDoc(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        description=record.description,
        body=IngestProfileBody.model_validate(record.body),
        created_at=record.created_at,
        updated_at=record.updated_at,
        cloned_from=record.cloned_from,
        active=record.active,
        active_revision=record.active_revision,
        validation=validation,
    )


def _to_summary(record: IngestProfileRecord) -> IngestProfileSummary:
    body = IngestProfileBody.model_validate(record.body)
    return IngestProfileSummary(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        primary_model=body.primary.model,
        secondary_model=body.secondary.model if body.secondary else None,
        active=record.active,
        active_revision=record.active_revision,
        updated_at=record.updated_at,
    )


def _active_conflict(exc: ActiveConflictError) -> Any:
    return active_conflict_error('the active ingest profile', exc.current)


def _revision_conflict(name: str, exc: RevisionConflictError) -> Any:
    return api_error(
        409,
        'revision_conflict',
        f'{name!r} changed since you loaded it',
        current_revision=exc.current_revision,
    )


def _schema_rows() -> list[IngestProfileFieldSchema]:
    rows: list[IngestProfileFieldSchema] = []
    for scope in ('primary', 'secondary'):
        rows.extend(
            [
                IngestProfileFieldSchema(
                    scope=scope,
                    field='model',
                    label='Triton model',
                    type='string',
                    help='A model loaded on Triton. The primary serves the end2end outputs; '
                    'the secondary the raw output0 tensor.',
                ),
                IngestProfileFieldSchema(
                    scope=scope, field='version', label='Model version', type='string', default='1'
                ),
                IngestProfileFieldSchema(
                    scope=scope,
                    field='input_size',
                    label='Input size',
                    type='int',
                    min=32,
                    max=4096,
                    help='Empty inherits the deployment value.',
                ),
                IngestProfileFieldSchema(
                    scope=scope,
                    field='labels_path',
                    label='Labels file',
                    type='string',
                    default='',
                    help='A labels.txt inside the model repository; empty reads the one in '
                    "the model's own directory.",
                ),
                IngestProfileFieldSchema(
                    scope=scope,
                    field='confidence_floor',
                    label='Confidence floor',
                    type='float',
                    min=0.0,
                    max=1.0,
                    help='Empty inherits the deployment value.',
                ),
                IngestProfileFieldSchema(
                    scope=scope,
                    field='batch_limit',
                    label='Batch limit',
                    type='int',
                    min=1,
                    max=256,
                    help='Empty inherits the deployment value.',
                ),
            ]
        )
    return rows


@router.get('/ingest_profiles/schema', response_model=IngestProfileSchema)
async def get_ingest_profile_schema() -> IngestProfileSchema:
    return IngestProfileSchema(fields=_schema_rows())


@router.post('/ingest_profiles/validate', response_model=ValidationReport)
async def validate_ingest_profile_route(
    body: IngestProfileValidateRequest, opensearch: OpenSearchDep, for_activation: bool = False
) -> Any:
    from src.app.resources import get_async_triton_pool

    await get_config_store().refresh(opensearch)
    return await validate_ingest_profile(
        body.name,
        body.body.model_dump(),
        existing_names=all_known_names(),
        for_activation=for_activation,
        pool_provider=get_async_triton_pool,
    )


@router.get('/ingest_profiles/active', response_model=ActiveConfigResponse)
async def get_active_ingest_profile(opensearch: OpenSearchDep) -> ActiveConfigResponse:
    return await build_active_config_response(opensearch, axis='ingest_profile')


@router.post('/ingest_profiles/active/rollback', response_model=ActiveConfigResponse)
async def rollback_active_ingest_profile(
    body: IngestProfileRollbackRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await rollback_profile(opensearch, expected_active=expected)
    except LookupError as exc:
        if exc.args and exc.args[0] == 'previous_deleted':
            raise api_error(
                409, 'previous_deleted', 'the previous profile was deleted; cannot roll back to it'
            ) from exc
        raise api_error(
            409, 'no_previous', 'there is no previous activation to roll back to'
        ) from exc
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    return await build_active_config_response(opensearch, axis='ingest_profile')


@router.post('/ingest_profiles/deactivate', response_model=ActiveConfigResponse)
async def deactivate_ingest_profile(
    body: IngestProfileDeactivateRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await activate_profile(opensearch, name=None, revision=None, expected_active=expected)
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    return await build_active_config_response(opensearch, axis='ingest_profile')


@router.get('/ingest_profiles', response_model=IngestProfileList)
async def list_ingest_profiles(
    opensearch: OpenSearchDep,
    include_templates: bool = Query(
        default=False,
        description='Also return the shipped example templates in `templates`; they are omitted by default.',
    ),
) -> IngestProfileList:
    store = get_config_store()
    await store.refresh(opensearch)
    templates = set(template_names())
    records = [build_record(n) for n in sorted(all_known_names() - templates)]
    template_rows: list[IngestProfileTemplateSummary] = []
    if include_templates:
        for template_name in sorted(templates):
            rec = build_record(template_name)
            body = IngestProfileBody.model_validate(rec.body if rec else {})
            template_rows.append(
                IngestProfileTemplateSummary(
                    name=template_name,
                    path=f'examples/ingest_profiles/{template_name}.json',
                    primary_model=body.primary.model,
                )
            )
    ref = store.current.active_ingest_profile
    active_name, active_revision = ref if isinstance(ref, tuple) else (None, None)
    return IngestProfileList(
        profiles=[_to_summary(r) for r in records if r is not None],
        templates=template_rows,
        active=ActiveRef(name=active_name, revision=active_revision),
        config_revision=store.current.config_revision,
        stale=store.current.stale,
    )


@router.post('/ingest_profiles', response_model=IngestProfileDoc, status_code=201)
async def create_ingest_profile(
    body: IngestProfileCreateRequest, opensearch: OpenSearchDep
) -> IngestProfileDoc:
    await get_config_store().refresh(opensearch)
    existing = all_known_names()
    report = await validate_ingest_profile(
        body.name, body.body.model_dump(), existing_names=existing - {body.name}
    )
    if any(
        e.code in ('ingest_profile_name_invalid', 'ingest_profile_name_reserved')
        for e in report.errors
    ):
        raise api_error(422, 'validation_failed', 'the profile name is not usable', report=report)
    if body.name in existing:
        raise api_error(409, 'name_conflict', taken_message(body.name))
    if not report.ok:
        raise api_error(
            422,
            'validation_failed',
            f'the profile has {len(report.errors)} error(s)',
            report=report,
        )
    record = await save_profile(
        opensearch,
        name=body.name,
        body=body.body.model_dump(),
        expected_revision=None,
        description=body.description,
    )
    return _to_doc(record, validation=report)


@router.get('/ingest_profiles/{name}', response_model=IngestProfileDoc)
async def get_ingest_profile(name: str, opensearch: OpenSearchDep) -> Any:
    await get_config_store().refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known ingest profile')
    return WireJSONResponse(
        content=_to_doc(record).model_dump(), headers={'ETag': f'"{record.etag}"'}
    )


@router.get('/ingest_profiles/{name}/revisions', response_model=IngestProfileRevisionsResponse)
async def list_ingest_profile_revisions(
    name: str, opensearch: OpenSearchDep
) -> IngestProfileRevisionsResponse:
    revisions = await list_revisions(opensearch, name)
    if revisions is None:
        raise api_error(404, 'not_found', f'{name!r} is not a stored ingest profile')
    return IngestProfileRevisionsResponse(
        name=name,
        revisions=[
            IngestProfileRevisionSummary(
                revision=r.revision,
                saved_at=r.saved_at,
                cloned_from=r.cloned_from,
                description=r.description,
            )
            for r in revisions
        ],
    )


@router.get('/ingest_profiles/{name}/revisions/{revision}', response_model=IngestProfileDoc)
async def get_ingest_profile_revision(
    name: str, revision: int, opensearch: OpenSearchDep
) -> IngestProfileDoc:
    record = await get_revision_record(opensearch, name, revision)
    if record is None:
        raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    return _to_doc(record)


@router.put('/ingest_profiles/{name}', response_model=IngestProfileDoc)
async def save_ingest_profile(
    name: str, body: IngestProfileSaveRequest, opensearch: OpenSearchDep
) -> IngestProfileDoc:
    await get_config_store().refresh(opensearch)
    existing = build_record(name)
    if existing is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known ingest profile')
    if existing.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    report = await validate_ingest_profile(name, body.body.model_dump(), existing_names=frozenset())
    if not report.ok:
        raise api_error(
            422,
            'validation_failed',
            f'the profile has {len(report.errors)} error(s)',
            report=report,
        )
    try:
        record = await save_profile(
            opensearch,
            name=name,
            body=body.body.model_dump(),
            expected_revision=body.expected_revision,
            description=body.description if body.description is not None else existing.description,
        )
    except RevisionConflictError as exc:
        raise _revision_conflict(name, exc) from exc
    return _to_doc(record, validation=report)


@router.delete('/ingest_profiles/{name}', status_code=204)
async def delete_ingest_profile(
    name: str, expected_revision: int, opensearch: OpenSearchDep
) -> None:
    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known ingest profile')
    if record.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    ref = store.current.active_ingest_profile
    if isinstance(ref, tuple) and ref[0] == name:
        raise api_error(409, 'in_use', f'{name!r} is the active ingest profile')
    try:
        await delete_profile(opensearch, name=name, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise _revision_conflict(name, exc) from exc
    return Response(status_code=204)  # type: ignore[return-value]


@router.post('/ingest_profiles/{name}/activate', response_model=IngestProfileActivateResponse)
async def activate_ingest_profile(
    name: str, body: IngestProfileActivateRequest, opensearch: OpenSearchDep
) -> IngestProfileActivateResponse:
    from src.services.config_store.activation_gate import run_activation_gate

    await get_config_store().refresh(opensearch)
    record = build_record(name, revision=body.revision)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known ingest profile')
    report = await run_activation_gate(
        'ingest_profile', name, record.revision, force=body.force, client=opensearch
    )
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await activate_profile(
            opensearch, name=name, revision=record.revision, expected_active=expected
        )
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    response = await build_active_config_response(opensearch, axis='ingest_profile')
    return IngestProfileActivateResponse(**response.model_dump(), validation=report)


# POST /ingest_profiles/{name}/clone lives in _ingest_profile_clone.py (imported
# for its route-registration side effect).
from src.routers.curation import _ingest_profile_clone  # noqa: E402,F401  # bottom-of-module import breaks a circular import; imported for its side effect
