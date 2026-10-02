"""``/region_profiles*`` + ``/config/vocabulary`` -- region-profile CRUD
(W4, any_domain_plan.md §4/§7.3).

Route-order note (§4.2, mirrors W3's §3.2): ``schema``, ``validate``,
``validate_segmenter_prompt``, ``test``, ``active`` (+ its children) and
``deactivate`` are declared before ``/{name}`` and its children.
``config_vocabulary.py`` registers ``GET /config/vocabulary`` separately
(its own side-effect import in ``__init__.py``).
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.routers.curation._config_common_models import ActiveConfigResponse, ActiveRef, api_error
from src.routers.curation._region_profile_models import (
    RegionProfileActivateRequest,
    RegionProfileBody,
    RegionProfileCreateRequest,
    RegionProfileDeactivateRequest,
    RegionProfileDoc,
    RegionProfileEffective,
    RegionProfileList,
    RegionProfileRevisionsResponse,
    RegionProfileRevisionSummary,
    RegionProfileRollbackRequest,
    RegionProfileSaveRequest,
    RegionProfileSchema,
    RegionProfileSummary,
    RegionProfileTemplateSummary,
    RegionProfileValidateRequest,
    SegmenterPromptValidateRequest,
)
from src.routers.curation._region_profile_schema import build_region_profile_schema
from src.services.config_store import ActiveConflictError, RevisionConflictError, get_config_store
from src.services.config_store.activation_view import build_active_config_response
from src.services.config_store.profile_validation import validate_profile
from src.services.config_store.profiles import (
    ProfileRecord,
    activate_profile,
    all_known_names,
    build_record,
    delete_profile,
    get_revision_record,
    list_revisions,
    rollback_profile,
    save_profile,
)
from src.services.curation.region_impact import compute_activation_impact


def _registry_class_names() -> frozenset[str]:
    reg = get_class_registry().load()
    return frozenset(c.class_name for c in reg.classes if not c.deprecated)


async def _segmenter_health_fn() -> tuple[str, str | None]:
    from src.routers.curation._models_segmenter import _segmenter_health
    from src.services.detection.segmenter_http import first_segmenter_url

    url = first_segmenter_url()
    if url is None:
        return 'unavailable', 'OP_SEGMENTER_URL is not configured'
    return await _segmenter_health(url)


def _effective(profile: Any) -> RegionProfileEffective:
    legs = []
    if profile.detector_model:
        legs.append('detector')
    if profile.segmenter_text_prompt:
        legs.append('segmenter')
    return RegionProfileEffective(
        reads_text=profile.reads_text,
        text_hint_active=profile.text_hint_active(
            segmenter_enabled=bool(profile.segmenter_text_prompt)
        ),
        legs=legs,
        segmenter_enabled=bool(profile.segmenter_text_prompt),
    )


def _decode_or_none(name: str, body: dict[str, Any]) -> Any:
    from src.services.detection.profile_registry import region_profile_from_dict

    try:
        return region_profile_from_dict({**body, 'name': name}, source=name)
    except ValueError:
        return None


def _to_doc(record: ProfileRecord, *, validation: Any = None) -> RegionProfileDoc:
    profile = _decode_or_none(record.name, record.body)
    return RegionProfileDoc(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        description=record.description,
        body=RegionProfileBody(**record.body),
        effective=_effective(profile)
        if profile is not None
        else RegionProfileEffective(
            reads_text=False, text_hint_active=False, legs=[], segmenter_enabled=False
        ),
        created_at=record.created_at,
        updated_at=record.updated_at,
        cloned_from=record.cloned_from,
        active=record.active,
        active_revision=record.active_revision,
        validation=validation,
    )


def _to_summary(record: ProfileRecord) -> RegionProfileSummary:
    profile = _decode_or_none(record.name, record.body)
    return RegionProfileSummary(
        name=record.name,
        source=record.source,
        read_only=record.read_only,
        revision=record.revision,
        etag=record.etag,
        display_name=profile.display_name if profile else '',
        display_name_singular=profile.display_name_singular if profile else '',
        region_class_name=profile.region_class_name if profile else '',
        text_reader=profile.text_reader if profile else 'none',
        reads_text=profile.reads_text if profile else False,
        detector_model=profile.detector_model if profile else '',
        segmenter_text_prompt=profile.segmenter_text_prompt if profile else '',
        parent_classes=sorted(profile.parent_classes) if profile else [],
        max_regions_per_item=profile.max_regions_per_item if profile else 1,
        active=record.active,
        active_revision=record.active_revision,
        updated_at=record.updated_at,
    )


def _template_names_only() -> list[str]:
    from src.services.config_store.profiles import _template_names

    return list(_template_names())


# =============================================================================
# GET /region_profiles/schema
# =============================================================================


@router.get('/region_profiles/schema', response_model=RegionProfileSchema)
async def get_region_profile_schema() -> RegionProfileSchema:
    return build_region_profile_schema()


# =============================================================================
# POST /region_profiles/validate, /validate_segmenter_prompt
# =============================================================================


@router.post('/region_profiles/validate')
async def validate_region_profile_route(
    body: RegionProfileValidateRequest, opensearch: OpenSearchDep, for_activation: bool = False
) -> Any:
    store = get_config_store()
    await store.refresh(opensearch)
    report = await validate_profile(
        body.name,
        body.body.model_dump(),
        existing_names=all_known_names(),
        for_activation=for_activation,
        segmenter_health=_segmenter_health_fn,
        class_names=_registry_class_names(),
        project_slug=_project_slug(),
    )
    return report.model_dump()


@router.post('/region_profiles/validate_segmenter_prompt')
async def validate_segmenter_prompt_route(body: SegmenterPromptValidateRequest) -> Any:
    from src.routers.curation._config_common_models import ValidationReport
    from src.services.config_store.profile_validation import _check_segmenter_prompt_text

    issues = _check_segmenter_prompt_text(body.text_prompt, sole_leg=body.sole_leg)
    errors = [i for i in issues if i.severity == 'error']
    warnings = [i for i in issues if i.severity != 'error']
    return ValidationReport(
        ok=not errors, errors=errors, warnings=warnings, force_allowed=False
    ).model_dump()


def _project_slug() -> str | None:
    from src.config import get_curation_config

    try:
        return get_curation_config().project_slug
    except Exception:  # pragma: no cover - defensive; always bound in routes
        return None


# =============================================================================
# GET /region_profiles/active[/impact], rollback, deactivate
# =============================================================================


@router.get('/region_profiles/active', response_model=ActiveConfigResponse)
async def get_active_region_profile_route(opensearch: OpenSearchDep) -> ActiveConfigResponse:
    return await build_active_config_response(opensearch, axis='detection_profile')


@router.get('/region_profiles/active/impact')
async def get_active_region_profile_impact(opensearch: OpenSearchDep) -> Any:
    from src.services.detection.profile_registry import get_active_region_profile

    return (
        await compute_activation_impact(opensearch, profile=get_active_region_profile())
    ).model_dump()


@router.post('/region_profiles/active/rollback', response_model=ActiveConfigResponse)
async def rollback_active_region_profile(
    body: RegionProfileRollbackRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await rollback_profile(opensearch, expected_active=expected)
    except LookupError as exc:
        # m-b fix (W3/W4 round-5 review): `previous_deleted` is distinct
        # from `no_previous` -- the profile was deleted while it was the
        # rollback target, not "there is no target."
        if exc.args and exc.args[0] == 'previous_deleted':
            raise api_error(
                409,
                'previous_deleted',
                'the previous region profile was deleted; cannot roll back to it',
            ) from exc
        raise api_error(
            409, 'no_previous', 'there is no previous activation to roll back to'
        ) from exc
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            'the active profile changed since you loaded it',
            current=ActiveRef(**exc.current) if exc.current else None,
        ) from exc
    return await build_active_config_response(opensearch, axis='detection_profile')


@router.post('/region_profiles/deactivate', response_model=ActiveConfigResponse)
async def deactivate_region_profile(
    body: RegionProfileDeactivateRequest, opensearch: OpenSearchDep
) -> ActiveConfigResponse:
    expected = body.expected_active.model_dump() if body.expected_active is not None else None
    try:
        await activate_profile(opensearch, name=None, revision=None, expected_active=expected)
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            'the active profile changed since you loaded it',
            current=ActiveRef(**exc.current) if exc.current else None,
        ) from exc
    return await build_active_config_response(opensearch, axis='detection_profile')


# =============================================================================
# GET/POST /region_profiles
# =============================================================================


@router.get('/region_profiles', response_model=RegionProfileList)
async def list_region_profiles(
    opensearch: OpenSearchDep, include_templates: bool = False
) -> RegionProfileList:
    store = get_config_store()
    await store.refresh(opensearch)
    names = all_known_names() - set(_template_names_only())
    records = [build_record(name) for name in sorted(names)]
    templates: list[RegionProfileTemplateSummary] = []
    if include_templates:
        for template_name in sorted(_template_names_only()):
            rec = build_record(template_name)
            body = rec.body if rec else {}
            templates.append(
                RegionProfileTemplateSummary(
                    name=template_name,
                    path=f'examples/region_profiles/{template_name}.json',
                    display_name=body.get('display_name'),
                    reads_text=body.get('text_reader', 'none') != 'none',
                )
            )
    active_name, active_revision = None, None
    ref = store.current.active_profile
    if isinstance(ref, tuple):
        active_name, active_revision = ref
    return RegionProfileList(
        profiles=[_to_summary(r) for r in records if r is not None],
        templates=templates,
        active=ActiveRef(name=active_name, revision=active_revision),
        config_revision=store.current.config_revision,
        stale=store.current.stale,
    )


@router.post('/region_profiles', response_model=RegionProfileDoc, status_code=201)
async def create_region_profile(
    body: RegionProfileCreateRequest, opensearch: OpenSearchDep
) -> RegionProfileDoc:
    store = get_config_store()
    await store.refresh(opensearch)
    existing = all_known_names()
    report = await validate_profile(
        body.name,
        body.body.model_dump(),
        existing_names=existing - {body.name},
        segmenter_health=_segmenter_health_fn,
        class_names=_registry_class_names(),
        project_slug=_project_slug(),
    )
    name_issues = [
        e for e in report.errors if e.code in ('profile_name_invalid', 'profile_name_reserved')
    ]
    if name_issues:
        raise api_error(422, 'validation_failed', 'the profile name is not usable', report=report)
    if body.name in existing:
        raise api_error(409, 'name_conflict', f'{body.name!r} is already taken')
    other_errors = [e for e in report.errors if e.code != 'name_conflict']
    if other_errors:
        raise api_error(
            422, 'validation_failed', f'the profile has {len(other_errors)} error(s)', report=report
        )
    record = await save_profile(
        opensearch,
        name=body.name,
        body=body.body.model_dump(),
        expected_revision=None,
        description=body.description,
    )
    return _to_doc(record, validation=report)


# =============================================================================
# GET /region_profiles/{name}[/revisions[/{revision}]]
# =============================================================================


@router.get('/region_profiles/{name}', response_model=RegionProfileDoc)
async def get_region_profile_route(name: str, opensearch: OpenSearchDep) -> Any:
    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known region profile')
    from fastapi.responses import ORJSONResponse

    payload = _to_doc(record).model_dump()
    return ORJSONResponse(content=payload, headers={'ETag': f'"{record.etag}"'})


@router.get('/region_profiles/{name}/revisions', response_model=RegionProfileRevisionsResponse)
async def list_region_profile_revisions(
    name: str, opensearch: OpenSearchDep
) -> RegionProfileRevisionsResponse:
    revisions = await list_revisions(opensearch, name)
    if revisions is None:
        raise api_error(404, 'not_found', f'{name!r} is not a stored profile')
    return RegionProfileRevisionsResponse(
        name=name,
        revisions=[
            RegionProfileRevisionSummary(
                revision=r.revision,
                saved_at=r.saved_at,
                cloned_from=r.cloned_from,
                description=r.description,
            )
            for r in revisions
        ],
    )


@router.get('/region_profiles/{name}/revisions/{revision}', response_model=RegionProfileDoc)
async def get_region_profile_revision(
    name: str, revision: int, opensearch: OpenSearchDep
) -> RegionProfileDoc:
    record = await get_revision_record(opensearch, name, revision)
    if record is None:
        raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    return _to_doc(record)


# POST /region_profiles/{name}/clone lives in _region_profile_clone.py
# (kept under the 700-LOC ratchet); imported at the bottom of this module
# for its route-registration side effect, mirroring models.py/
# _models_sharing.py.


# =============================================================================
# Save a new revision, or delete a stored profile
# =============================================================================


@router.put('/region_profiles/{name}', response_model=RegionProfileDoc)
async def save_region_profile(
    name: str, body: RegionProfileSaveRequest, opensearch: OpenSearchDep
) -> RegionProfileDoc:
    store = get_config_store()
    await store.refresh(opensearch)
    existing = build_record(name)
    if existing is not None and existing.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    # M-2 fix: PUT is not a back door to create -- POST does that check.
    if existing is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known region profile')

    report = await validate_profile(
        name,
        body.body.model_dump(),
        segmenter_health=_segmenter_health_fn,
        class_names=_registry_class_names(),
        project_slug=_project_slug(),
    )
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


@router.delete('/region_profiles/{name}', status_code=204)
async def delete_region_profile_route(
    name: str, expected_revision: int, opensearch: OpenSearchDep
) -> None:
    from fastapi import Response

    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known region profile')
    if record.read_only:
        raise api_error(403, 'read_only', f'{name!r} is read-only')
    ref = store.current.active_profile
    if isinstance(ref, tuple) and ref[0] == name:
        raise api_error(409, 'in_use', f'{name!r} is the active profile')
    try:
        await delete_profile(opensearch, name=name, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{name!r} changed since you loaded it',
            current_revision=exc.current_revision,
        ) from exc
    return Response(status_code=204)  # type: ignore[return-value]


# =============================================================================
# Activate a profile as the default
# =============================================================================


@router.post('/region_profiles/{name}/activate')
async def activate_region_profile_route(
    name: str, body: RegionProfileActivateRequest, opensearch: OpenSearchDep
) -> Any:
    from src.services.config_store.activation_gate import run_activation_gate

    store = get_config_store()
    await store.refresh(opensearch)
    record = build_record(name, revision=body.revision)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known region profile')

    # R4-1 fix (W3/W4 round-4 review): the ONE shared `for_activation` gate
    # every activation writer calls -- activate, settings-bridge, rollback.
    report = await run_activation_gate(
        'detection_profile', name, record.revision, force=body.force, client=opensearch
    )

    expected_active = (
        body.expected_active.model_dump() if body.expected_active is not None else None
    )
    try:
        await activate_profile(
            opensearch, name=name, revision=record.revision, expected_active=expected_active
        )
    except ActiveConflictError as exc:
        raise api_error(
            409,
            'active_conflict',
            'the active profile changed since you loaded it',
            current=ActiveRef(**exc.current) if exc.current else None,
        ) from exc

    from src.services.detection.profile_registry import get_active_region_profile

    response = await build_active_config_response(opensearch, axis='detection_profile')
    impact = await compute_activation_impact(opensearch, profile=get_active_region_profile())
    return {
        **response.model_dump(),
        'impact': impact.model_dump(),
        'validation': report.model_dump(),
    }


# POST /region_profiles/{name}/clone lives in _region_profile_clone.py
# (kept under the 700-LOC ratchet); imported for its route-registration
# side effect.
from src.routers.curation import _region_profile_clone  # noqa: E402,F401
