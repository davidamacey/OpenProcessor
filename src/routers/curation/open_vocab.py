"""``/open_vocab*``: open-vocabulary prompt sets (config axis ``open_vocab``).

Route-order note: ``schema``, ``validate``, ``active`` (+ its children) and
``deactivate`` are declared before ``/{name}`` and its children, as in
``region_profiles.py``. ``POST /open_vocab/{name}/clone`` lives in
``_open_vocab_clone.py`` (imported at the bottom for its registration side
effect), the per-image test route in ``open_vocab_test.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Query

from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.routers.curation._config_common_models import (
    ActiveConfigResponse,
    ActiveRef,
    ValidationReport,
    active_conflict_error,
    api_error,
)
from src.routers.curation._models_segmenter import segmenter_availability
from src.routers.curation._open_vocab_models import (
    OpenVocabActivateRequest,
    OpenVocabActivateResponse,
    OpenVocabBody,
    OpenVocabCreateRequest,
    OpenVocabDeactivateRequest,
    OpenVocabDoc,
    OpenVocabList,
    OpenVocabRevisionsResponse,
    OpenVocabRevisionSummary,
    OpenVocabRollbackRequest,
    OpenVocabSaveRequest,
    OpenVocabSchema,
    OpenVocabSummary,
    OpenVocabTemplateSummary,
    OpenVocabValidateRequest,
    SegmenterAvailability,
)
from src.routers.curation._open_vocab_schema import build_open_vocab_schema
from src.services.config_store import ActiveConflictError, RevisionConflictError, get_config_store
from src.services.config_store.activation_view import build_active_config_response
from src.services.config_store.open_vocab import (
    OpenVocabRecord,
    activate_set,
    all_known_names,
    build_record,
    delete_set,
    get_revision_record,
    list_revisions,
    rollback_set,
    save_set,
    template_names,
)
from src.services.config_store.open_vocab_validation import validate_open_vocab


def validation_inputs() -> dict[str, Any]:
    """The project-bound inputs every open-vocabulary validation needs (the
    registry's class names, the primary detector's class names, the
    segmenter probe): the one place they are assembled, so create, save,
    validate, clone and the activation gate cannot disagree."""
    from src.config.ingest_profiles import ingest_primary_profile
    from src.routers.curation._models_segmenter import configured_segmenter_health
    from src.services.labeling.vlm_endpoints import vlm_configured
    from src.utils.class_names import get_class_names

    registry = get_class_registry().load()
    detector = ingest_primary_profile().detector_model
    return {
        'class_names': frozenset(c.class_name for c in registry.classes if not c.deprecated),
        'detector_class_names': frozenset(get_class_names(detector).values())
        if detector
        else frozenset(),
        'segmenter_health': configured_segmenter_health,
        'vlm_configured': vlm_configured(),
    }


def _to_doc(record: OpenVocabRecord, *, validation: Any = None) -> OpenVocabDoc:
    return OpenVocabDoc(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        description=record.description,
        body=OpenVocabBody(**record.body),
        created_at=record.created_at,
        updated_at=record.updated_at,
        cloned_from=record.cloned_from,
        active=record.active,
        active_revision=record.active_revision,
        validation=validation,
    )


def _to_summary(record: OpenVocabRecord) -> OpenVocabSummary:
    body = OpenVocabBody(**record.body)
    return OpenVocabSummary(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        display_name=body.display_name,
        n_targets=len(body.targets),
        n_enabled_targets=sum(1 for t in body.targets if t.enabled),
        run_on_ingest=body.run_on_ingest,
        active=record.active,
        active_revision=record.active_revision,
        updated_at=record.updated_at,
    )


def _active_conflict(exc: ActiveConflictError) -> Any:
    return active_conflict_error('the active open-vocabulary set', exc.current)


def _revision_conflict(name: str, exc: RevisionConflictError) -> Any:
    return api_error(
        409,
        'revision_conflict',
        f'{name!r} changed since you loaded it',
        current_revision=exc.current_revision,
    )


@router.get('/open_vocab/schema', response_model=OpenVocabSchema)
async def get_open_vocab_schema() -> OpenVocabSchema:
    return build_open_vocab_schema()


@router.post('/open_vocab/validate', response_model=ValidationReport)
async def validate_open_vocab_route(
    body: OpenVocabValidateRequest, opensearch: OpenSearchDep, for_activation: bool = False
) -> Any:
    await get_config_store().refresh(opensearch)
    return await validate_open_vocab(
        body.name,
        body.body.model_dump(),
        existing_names=all_known_names(),
        for_activation=for_activation,
        **validation_inputs(),
    )


@router.get('/open_vocab/active', response_model=ActiveConfigResponse)
async def get_active_open_vocab(opensearch: OpenSearchDep) -> ActiveConfigResponse:
    return await build_active_config_response(opensearch, axis='open_vocab')


@router.post('/open_vocab/active/rollback', response_model=ActiveConfigResponse)
async def rollback_active_open_vocab(
    body: OpenVocabRollbackRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await rollback_set(opensearch, expected_active=expected)
    except LookupError as exc:
        if exc.args and exc.args[0] == 'previous_deleted':
            raise api_error(
                409, 'previous_deleted', 'the previous set was deleted; cannot roll back to it'
            ) from exc
        raise api_error(
            409, 'no_previous', 'there is no previous activation to roll back to'
        ) from exc
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    return await build_active_config_response(opensearch, axis='open_vocab')


@router.post('/open_vocab/deactivate', response_model=ActiveConfigResponse)
async def deactivate_open_vocab(
    body: OpenVocabDeactivateRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await activate_set(opensearch, name=None, revision=None, expected_active=expected)
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    return await build_active_config_response(opensearch, axis='open_vocab')


@router.get('/open_vocab', response_model=OpenVocabList)
async def list_open_vocab(
    opensearch: OpenSearchDep,
    include_templates: bool = Query(
        default=False,
        description='Also return the shipped example templates in `templates`; they are omitted by default.',
    ),
) -> OpenVocabList:
    store = get_config_store()
    await store.refresh(opensearch)
    templates = set(template_names())
    records = [build_record(n) for n in sorted(all_known_names() - templates)]
    template_rows: list[OpenVocabTemplateSummary] = []
    if include_templates:
        for template_name in sorted(templates):
            rec = build_record(template_name)
            body = OpenVocabBody(**(rec.body if rec else {}))
            template_rows.append(
                OpenVocabTemplateSummary(
                    name=template_name,
                    path=f'examples/open_vocab/{template_name}.json',
                    display_name=body.display_name or None,
                    n_targets=len(body.targets),
                )
            )
    ref = store.current.active_open_vocab
    active_name, active_revision = ref if isinstance(ref, tuple) else (None, None)
    configured, reachable = await segmenter_availability()
    return OpenVocabList(
        sets=[_to_summary(r) for r in records if r is not None],
        templates=template_rows,
        active=ActiveRef(name=active_name, revision=active_revision),
        config_revision=store.current.config_revision,
        stale=store.current.stale,
        segmenter=SegmenterAvailability(configured=configured, reachable=reachable),
    )


@router.post('/open_vocab', response_model=OpenVocabDoc, status_code=201)
async def create_open_vocab(
    body: OpenVocabCreateRequest, opensearch: OpenSearchDep
) -> OpenVocabDoc:
    await get_config_store().refresh(opensearch)
    existing = all_known_names()
    report = await validate_open_vocab(
        body.name,
        body.body.model_dump(),
        existing_names=existing - {body.name},
        **validation_inputs(),
    )
    if any(
        e.code in ('open_vocab_name_invalid', 'open_vocab_name_reserved') for e in report.errors
    ):
        raise api_error(422, 'validation_failed', 'the set name is not usable', report=report)
    if body.name in existing:
        raise api_error(409, 'name_conflict', f'{body.name!r} is already taken')
    if not report.ok:
        raise api_error(
            422, 'validation_failed', f'the set has {len(report.errors)} error(s)', report=report
        )
    record = await save_set(
        opensearch,
        name=body.name,
        body=body.body.model_dump(),
        expected_revision=None,
        description=body.description,
    )
    return _to_doc(record, validation=report)


@router.get('/open_vocab/{name}', response_model=OpenVocabDoc)
async def get_open_vocab(name: str, opensearch: OpenSearchDep) -> Any:
    from fastapi.responses import ORJSONResponse

    await get_config_store().refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known open-vocabulary set')
    return ORJSONResponse(
        content=_to_doc(record).model_dump(), headers={'ETag': f'"{record.etag}"'}
    )


@router.get('/open_vocab/{name}/revisions', response_model=OpenVocabRevisionsResponse)
async def list_open_vocab_revisions(
    name: str, opensearch: OpenSearchDep
) -> OpenVocabRevisionsResponse:
    revisions = await list_revisions(opensearch, name)
    if revisions is None:
        raise api_error(404, 'not_found', f'{name!r} is not a stored open-vocabulary set')
    return OpenVocabRevisionsResponse(
        name=name,
        revisions=[
            OpenVocabRevisionSummary(
                revision=r.revision,
                saved_at=r.saved_at,
                cloned_from=r.cloned_from,
                description=r.description,
            )
            for r in revisions
        ],
    )


@router.get('/open_vocab/{name}/revisions/{revision}', response_model=OpenVocabDoc)
async def get_open_vocab_revision(
    name: str, revision: int, opensearch: OpenSearchDep
) -> OpenVocabDoc:
    record = await get_revision_record(opensearch, name, revision)
    if record is None:
        raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    return _to_doc(record)


@router.put('/open_vocab/{name}', response_model=OpenVocabDoc)
async def save_open_vocab(
    name: str, body: OpenVocabSaveRequest, opensearch: OpenSearchDep
) -> OpenVocabDoc:
    await get_config_store().refresh(opensearch)
    existing = build_record(name)
    if existing is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known open-vocabulary set')
    if existing.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    report = await validate_open_vocab(name, body.body.model_dump(), **validation_inputs())
    if not report.ok:
        raise api_error(
            422, 'validation_failed', f'the set has {len(report.errors)} error(s)', report=report
        )
    try:
        record = await save_set(
            opensearch,
            name=name,
            body=body.body.model_dump(),
            expected_revision=body.expected_revision,
            description=body.description if body.description is not None else existing.description,
        )
    except RevisionConflictError as exc:
        raise _revision_conflict(name, exc) from exc
    return _to_doc(record, validation=report)


@router.delete('/open_vocab/{name}', status_code=204)
async def delete_open_vocab(name: str, expected_revision: int, opensearch: OpenSearchDep) -> None:
    from fastapi import Response

    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known open-vocabulary set')
    if record.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    ref = store.current.active_open_vocab
    if isinstance(ref, tuple) and ref[0] == name:
        raise api_error(409, 'in_use', f'{name!r} is the active open-vocabulary set')
    try:
        await delete_set(opensearch, name=name, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise _revision_conflict(name, exc) from exc
    return Response(status_code=204)  # type: ignore[return-value]


@router.post('/open_vocab/{name}/activate', response_model=OpenVocabActivateResponse)
async def activate_open_vocab(
    name: str, body: OpenVocabActivateRequest, opensearch: OpenSearchDep
) -> OpenVocabActivateResponse:
    from src.services.config_store.activation_gate import run_activation_gate

    await get_config_store().refresh(opensearch)
    record = build_record(name, revision=body.revision)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known open-vocabulary set')
    report = await run_activation_gate(
        'open_vocab', name, record.revision, force=body.force, client=opensearch
    )
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await activate_set(
            opensearch, name=name, revision=record.revision, expected_active=expected
        )
    except ActiveConflictError as exc:
        raise _active_conflict(exc) from exc
    response = await build_active_config_response(opensearch, axis='open_vocab')
    return OpenVocabActivateResponse(**response.model_dump(), validation=report)


# POST /open_vocab/{name}/clone lives in _open_vocab_clone.py (imported for
# its route-registration side effect).
from src.routers.curation import _open_vocab_clone  # noqa: E402,F401
