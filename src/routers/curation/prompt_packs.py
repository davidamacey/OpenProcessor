"""``/prompt_packs*`` -- prompt-pack CRUD (W3, any_domain_plan.md §3/§7.2).

Route-order note (§3.2): ``schema``, ``validate``, ``test``, ``active``
and ``active/rollback`` are declared before ``/{name}`` and its children,
or FastAPI would match ``{name}`` first.
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.routers.curation._config_common_models import (
    ActivateResponse,
    ActiveConfigResponse,
    ActiveRef,
    active_conflict_error,
    api_error,
)
from src.routers.curation._prompt_pack_models import (
    PromptPackActivateRequest,
    PromptPackBody,
    PromptPackCallSchema,
    PromptPackCloneRequest,
    PromptPackCreateRequest,
    PromptPackDoc,
    PromptPackFieldSchema,
    PromptPackList,
    PromptPackPlaceholderHelp,
    PromptPackRevisionsResponse,
    PromptPackRevisionSummary,
    PromptPackRollbackRequest,
    PromptPackSaveRequest,
    PromptPackSchema,
    PromptPackSummary,
    PromptPackTemplateSummary,
    PromptPackValidateRequest,
)
from src.services.config_store import ActiveConflictError, RevisionConflictError, get_config_store
from src.services.config_store.activation_view import build_active_config_response
from src.services.config_store.pack_validation import validate_pack
from src.services.config_store.packs import (
    PackRecord,
    activate_pack,
    all_known_names,
    build_record,
    delete_pack,
    get_revision_record,
    list_revisions,
    rollback_pack,
    save_pack,
)
from src.services.labeling.vlm_prompts import FORMATTED_PLACEHOLDERS, REPLY_KEY_CONTRACT


_PLACEHOLDER_HELP: dict[str, str] = {
    'class_names_csv': 'comma-separated class names from the registry',
    'class_block': "the classify instruction and numbered catalog, or 'no candidate'",
    'region_block': (
        'the proposed boxes (1..N), as a numbered-overlay description or numbered '
        "normalized coordinates, or 'no candidate'"
    ),
}

_FIELD_GROUP: dict[str, str] = {
    'class_system': 'classify',
    'class_user_template': 'classify',
    'open_class_system': 'open_classify',
    'open_class_user_template': 'open_classify',
    'combined_system': 'combined',
    'combined_user_template': 'combined',
    'combined_batch_system': 'combined_batch',
    'combined_batch_rules': 'combined_batch',
    'region_system': 'region_verify',
    'region_user': 'region_verify',
    'region_batch_system': 'region_verify_batch',
    'region_batch_user': 'region_verify_batch',
    'region_visible_system': 'region_visible',
    'region_visible_user': 'region_visible',
    'class_descriptions': 'vocabulary',
    'synonyms': 'vocabulary',
}

_CALL_LABELS: dict[str, str] = {
    'classify': 'Classify',
    'open_classify': 'Classify (open vocabulary)',
    'combined': 'Classify + verify region',
    'combined_batch': 'Classify + verify region (batch)',
    'region_verify': 'Verify region',
    'region_verify_batch': 'Verify region (batch)',
    'region_visible': 'Region visible pre-filter',
}

_FIELD_USED_BY: dict[str, list[str]] = {
    'classify': ['auto_label_vlm_stage', 'vlm_label_batch'],
    'open_classify': ['auto_label_vlm_stage', 'vlm_label_batch'],
    'combined': ['detection_worker_combined_verify'],
    'combined_batch': ['detection_worker_combined_verify'],
    'region_verify': ['vlm_verify_regions'],
    'region_verify_batch': ['vlm_verify_region_batch'],
    'region_visible': ['vlm_region_visible_batch'],
}


def _registry_class_names() -> frozenset[str]:
    reg = get_class_registry().load()
    return frozenset(c.class_name for c in reg.classes if not c.deprecated)


def _resolve_profile(profile_name: str | None) -> Any:
    from src.services.detection.profile_registry import get_active_region_profile, get_profiles

    if profile_name is None:
        return get_active_region_profile()
    return get_profiles().get(profile_name)


def _to_doc(record: PackRecord, *, validation: Any = None) -> PromptPackDoc:
    return PromptPackDoc(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        description=record.description,
        body=PromptPackBody(**record.body),
        created_at=record.created_at,
        updated_at=record.updated_at,
        cloned_from=record.cloned_from,
        active=record.active,
        active_revision=record.active_revision,
        validation=validation,
    )


def _to_summary(record: PackRecord) -> PromptPackSummary:
    return PromptPackSummary(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        description=record.description,
        asks_region_text=record.asks_region_text,
        active=record.active,
        active_revision=record.active_revision,
        updated_at=record.updated_at,
    )


# =============================================================================
# GET /prompt_packs/schema
# =============================================================================


@router.get('/prompt_packs/schema', response_model=PromptPackSchema)
async def get_prompt_pack_schema() -> PromptPackSchema:
    class_names = sorted(_registry_class_names())[:5] or ['car', 'truck', 'bus']
    fields: list[PromptPackFieldSchema] = []
    for f in PromptPackBody.model_fields:
        kind = 'map' if f in ('class_descriptions', 'synonyms') else 'text'
        formatted = f in FORMATTED_PLACEHOLDERS
        call_id = _FIELD_GROUP.get(f, 'vocabulary')
        contract = REPLY_KEY_CONTRACT.get(call_id)
        fields.append(
            PromptPackFieldSchema(
                field=f,
                label=f.replace('_', ' ').capitalize(),
                group=call_id,
                kind=kind,
                formatted=formatted,
                required_placeholders=list(FORMATTED_PLACEHOLDERS.get(f, ())),
                allowed_placeholders=list(FORMATTED_PLACEHOLDERS.get(f, ())),
                expected_reply_keys=list(contract['required']) if contract else [],
                optional_reply_keys=list(contract['optional']) if contract else [],
                used_by=_FIELD_USED_BY.get(call_id, []),
                help=f'Example (first classes): {", ".join(class_names)}' if formatted else '',
            )
        )
    placeholders = [
        PromptPackPlaceholderHelp(name=name, meaning=meaning, example=', '.join(class_names))
        for name, meaning in _PLACEHOLDER_HELP.items()
    ]
    calls = [
        PromptPackCallSchema(
            id=call_id, label=_CALL_LABELS.get(call_id, call_id), fields=contract['fields']
        )
        for call_id, contract in REPLY_KEY_CONTRACT.items()
    ]
    return PromptPackSchema(fields=fields, placeholders=placeholders, calls=calls)


# =============================================================================
# POST /prompt_packs/validate
# =============================================================================


@router.post('/prompt_packs/validate')
async def validate_prompt_pack_route(
    body: PromptPackValidateRequest, opensearch: OpenSearchDep, profile: str | None = None
) -> Any:
    store = get_config_store()
    await store.refresh(opensearch)
    existing = all_known_names()
    report = validate_pack(
        body.name,
        body.body.model_dump(),
        existing_names=existing,
        profile=_resolve_profile(profile),
        class_names=_registry_class_names(),
    )
    return report.model_dump()


# =============================================================================
# GET /prompt_packs/active, POST /prompt_packs/active/rollback
# =============================================================================


@router.get('/prompt_packs/active', response_model=ActiveConfigResponse)
async def get_active_prompt_pack_route(opensearch: OpenSearchDep) -> ActiveConfigResponse:
    return await build_active_config_response(opensearch, axis='prompt_pack')


@router.post('/prompt_packs/active/rollback', response_model=ActiveConfigResponse)
async def rollback_active_prompt_pack(
    body: PromptPackRollbackRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await rollback_pack(opensearch, expected_active=expected)
    except LookupError as exc:
        # m-b fix (W3/W4 round-5 review): `previous_deleted` is distinct
        # from `no_previous` -- the pack was deleted while it was the
        # rollback target, not "there is no target."
        if exc.args and exc.args[0] == 'previous_deleted':
            raise api_error(
                409, 'previous_deleted', 'the previous pack was deleted; cannot roll back to it'
            ) from exc
        raise api_error(
            409, 'no_previous', 'there is no previous activation to roll back to'
        ) from exc
    except ActiveConflictError as exc:
        raise active_conflict_error('the active pack', exc.current) from exc
    return await build_active_config_response(opensearch, axis='prompt_pack')


# =============================================================================
# GET/POST /prompt_packs
# =============================================================================


@router.get('/prompt_packs', response_model=PromptPackList)
async def list_prompt_packs(opensearch: OpenSearchDep) -> PromptPackList:
    store = get_config_store()
    await store.refresh(opensearch)
    names = all_known_names() - set(_template_names_only())
    packs = [build_record(name) for name in sorted(names)]
    templates = [
        PromptPackTemplateSummary(name=n, path=f'examples/prompt_packs/{n}.json')
        for n in sorted(_template_names_only())
    ]
    active_name, active_revision = None, None
    ref = store.current.active_pack
    if isinstance(ref, tuple):
        active_name, active_revision = ref
    return PromptPackList(
        packs=[_to_summary(r) for r in packs if r is not None],
        templates=templates,
        active=ActiveRef(name=active_name, revision=active_revision),
        config_revision=store.current.config_revision,
        stale=store.current.stale,
    )


def _template_names_only() -> list[str]:
    from src.services.config_store.packs import _template_names

    return list(_template_names())


@router.post('/prompt_packs', response_model=PromptPackDoc, status_code=201)
async def create_prompt_pack(
    body: PromptPackCreateRequest, opensearch: OpenSearchDep
) -> PromptPackDoc:
    store = get_config_store()
    await store.refresh(opensearch)
    existing = all_known_names()
    report = validate_pack(
        body.name,
        body.body.model_dump(),
        existing_names=existing - {body.name},
        profile=_resolve_profile(None),
        class_names=_registry_class_names(),
    )
    name_issues = [
        e for e in report.errors if e.code in ('pack_name_invalid', 'pack_name_reserved')
    ]
    if name_issues:
        raise api_error(422, 'validation_failed', 'the pack name is not usable', report=report)
    if body.name in existing:
        raise api_error(409, 'name_conflict', f'{body.name!r} is already taken')
    other_errors = [e for e in report.errors if e.code != 'name_conflict']
    if other_errors:
        raise api_error(
            422, 'validation_failed', f'the pack has {len(other_errors)} error(s)', report=report
        )

    record = await save_pack(
        opensearch,
        name=body.name,
        body=body.body.model_dump(),
        expected_revision=None,
        description=body.description,
    )
    return _to_doc(record, validation=report)


# =============================================================================
# GET /prompt_packs/{name}[/revisions[/{revision}]]
# =============================================================================


@router.get('/prompt_packs/{name}', response_model=PromptPackDoc)
async def get_prompt_pack_route(name: str, opensearch: OpenSearchDep) -> Any:
    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known pack')
    from fastapi.responses import ORJSONResponse

    payload = _to_doc(record).model_dump()
    return ORJSONResponse(content=payload, headers={'ETag': f'"{record.etag}"'})


@router.get('/prompt_packs/{name}/revisions', response_model=PromptPackRevisionsResponse)
async def list_prompt_pack_revisions(
    name: str, opensearch: OpenSearchDep
) -> PromptPackRevisionsResponse:
    revisions = await list_revisions(opensearch, name)
    if revisions is None:
        raise api_error(404, 'not_found', f'{name!r} is not a stored pack')
    return PromptPackRevisionsResponse(
        name=name,
        revisions=[
            PromptPackRevisionSummary(
                revision=r.revision,
                saved_at=r.saved_at,
                cloned_from=r.cloned_from,
                description=r.description,
            )
            for r in revisions
        ],
    )


@router.get('/prompt_packs/{name}/revisions/{revision}', response_model=PromptPackDoc)
async def get_prompt_pack_revision(
    name: str, revision: int, opensearch: OpenSearchDep
) -> PromptPackDoc:
    record = await get_revision_record(opensearch, name, revision)
    if record is None:
        raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    return _to_doc(record)


# =============================================================================
# Clone a pack into a new stored pack
# =============================================================================


async def _resolve_clone_source(
    client: Any, *, name: str, revision: int | None, source: str | None
) -> PackRecord:
    if source == 'stored' and revision is not None:
        record = await get_revision_record(client, name, revision)
        if record is None:
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
        return record
    if source is not None:
        record = build_record(name, revision=revision)
        if record is None or record.source != source:
            raise api_error(404, 'not_found', f'{name!r} has no {source} source')
        return record
    if revision is not None:
        record = await get_revision_record(client, name, revision)
        if record is not None:
            return record
    record = build_record(name, revision=revision)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known pack')
    return record


@router.post('/prompt_packs/{name}/clone', response_model=PromptPackDoc, status_code=201)
async def clone_prompt_pack(
    name: str, body: PromptPackCloneRequest, opensearch: OpenSearchDep
) -> PromptPackDoc:
    from src.config import get_curation_config
    from src.services.config_store.clone_shared import (
        cloned_description,
        cloned_from_tag,
        read_source_record,
        reject_invalid_clone_name,
    )

    target_slug = get_curation_config().project_slug

    async def _resolve(client: Any) -> PackRecord:
        return await _resolve_clone_source(
            client, name=name, revision=body.revision, source=body.source
        )

    source = await read_source_record(
        from_project=body.from_project,
        target_slug=target_slug,
        opensearch=opensearch,
        resolve=_resolve,
    )

    existing = all_known_names()
    # M-2 fix: new_name must obey the same slug/reserved-word rules as create.
    name_report = validate_pack(
        body.new_name, source.body, existing_names=existing, class_names=_registry_class_names()
    )
    reject_invalid_clone_name(
        name_report,
        new_name=body.new_name,
        existing=existing,
        name_codes=('pack_name_invalid', 'pack_name_reserved'),
        what='pack',
    )

    new_body = dict(source.body)
    report = validate_pack(None, new_body, class_names=_registry_class_names())
    if not report.ok:
        raise api_error(
            422, 'validation_failed', 'the cloned pack has content errors', report=report
        )

    cloned_from = cloned_from_tag(
        source_project=body.from_project or target_slug, name=name, revision=source.revision
    )
    try:
        record = await save_pack(
            opensearch,
            name=body.new_name,
            body=new_body,
            expected_revision=None,
            description=cloned_description(body.description, source.description),
            cloned_from=cloned_from,
        )
    except RevisionConflictError as exc:
        # M-5 fix: a name that exists in OpenSearch but wasn't yet visible
        # to this process's `all_known_names()` snapshot (a stale/cold
        # target store) must still 409, not bubble up as a bare 500.
        raise api_error(409, 'name_conflict', f'{body.new_name!r} is already taken') from exc
    return _to_doc(record, validation=report)


# =============================================================================
# Save a new revision, or delete a stored pack
# =============================================================================


@router.put('/prompt_packs/{name}', response_model=PromptPackDoc)
async def save_prompt_pack(
    name: str, body: PromptPackSaveRequest, opensearch: OpenSearchDep
) -> PromptPackDoc:
    store = get_config_store()
    await store.refresh(opensearch)
    existing = build_record(name)
    if existing is not None and existing.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    # M-2 fix: PUT must not be a back door to create a reserved/invalid
    # name -- creation goes through POST, which already runs this check.
    # An unknown name (whether or not it would otherwise be a legal slug)
    # 404s here; only an already-stored pack is ever a valid PUT target.
    if existing is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known pack')

    report = validate_pack(name, body.body.model_dump(), class_names=_registry_class_names())
    if not report.ok:
        raise api_error(
            422, 'validation_failed', f'the pack has {len(report.errors)} error(s)', report=report
        )

    try:
        record = await save_pack(
            opensearch,
            name=name,
            body=body.body.model_dump(),
            expected_revision=body.expected_revision,
            description=body.description
            if body.description is not None
            else (existing.description if existing else ''),
        )
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{name!r} changed since you loaded it',
            current_revision=exc.current_revision,
        ) from exc
    return _to_doc(record, validation=report)


@router.delete('/prompt_packs/{name}', status_code=204)
async def delete_prompt_pack_route(
    name: str, expected_revision: int, opensearch: OpenSearchDep
) -> None:
    from fastapi import Response

    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known pack')
    if record.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    ref = store.current.active_pack
    if isinstance(ref, tuple) and ref[0] == name:
        raise api_error(409, 'in_use', f'{name!r} is the active pack')
    try:
        await delete_pack(opensearch, name=name, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{name!r} changed since you loaded it',
            current_revision=exc.current_revision,
        ) from exc
    # A 204 must carry no body and no JSON content-type -- ORJSONResponse
    # (this router's default_response_class) would otherwise still stamp
    # 'application/json' on an empty body, and a client calling .json()
    # on it (as tests/curation/test_cross_project_leak.py's isolation
    # sweep does on every JSON-labeled response) gets a JSONDecodeError.
    return Response(status_code=204)


# =============================================================================
# Activate a pack as the default
# =============================================================================


@router.post('/prompt_packs/{name}/activate', response_model=ActivateResponse)
async def activate_prompt_pack_route(
    name: str, body: PromptPackActivateRequest, opensearch: OpenSearchDep
) -> ActivateResponse:
    from src.services.config_store.activation_gate import run_activation_gate

    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name, revision=body.revision)
    if record is None and body.revision is None:
        # Written through another API worker: this process may hold a stale snapshot.
        await store.refresh(opensearch, force=True)
        record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known pack')

    # R4-1 fix (W3/W4 round-4 review): the ONE shared `for_activation` gate
    # every activation writer calls -- activate, settings-bridge, rollback.
    report = await run_activation_gate(
        'prompt_pack', name, record.revision, force=body.force, client=opensearch
    )

    expected_active = (
        body.expected_active.model_dump() if body.expected_active is not None else None
    )
    try:
        await activate_pack(
            opensearch, name=name, revision=record.revision, expected_active=expected_active
        )
    except ActiveConflictError as exc:
        raise active_conflict_error('the active pack', exc.current) from exc

    response = await build_active_config_response(opensearch, axis='prompt_pack')
    return ActivateResponse(**response.model_dump(), validation=report)
