"""``/curation/vlm/endpoints*``: the deployment-wide VLM endpoint registry
(W9.6). Registered on the projects ``global_router``: the registry belongs to
no project, so these routes run with nothing bound. Which endpoint a
project *uses* is that project's activation (:mod:`vlm_activation`, scoped).

Declared static-first (``/schema``, ``/validate``) so they are never read as
an endpoint ``{name}``; the reserved names in
:data:`~src.services.labeling.vlm_endpoints.RESERVED_NAMES` keep them that
way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Query, Response

from src.core.logging import get_logger
from src.routers.curation._common import OpenSearchDep  # noqa: TC001 - FastAPI resolves it
from src.routers.curation._config_common_models import api_error
from src.routers.curation._vlm_endpoint_models import (
    VlmEndpointCloneRequest,
    VlmEndpointCreate,
    VlmEndpointDoc,
    VlmEndpointList,
    VlmEndpointSaveRequest,
    VlmEndpointSchema,
    VlmProbeResult,
    VlmRevisionsResponse,
    VlmRevisionSummary,
    VlmValidateRequest,
    VlmValidateResponse,
)
from src.routers.curation._vlm_endpoint_views import (
    doc_of,
    endpoint_schema,
    labels_wire,
    probe_wire,
    secret_refs_wire,
    summary_of,
)
from src.routers.curation.projects import global_router
from src.services.config_store import get_global_config_store
from src.services.config_store.index import RevisionConflictError
from src.services.config_store.vlm_endpoints import (
    delete_endpoint,
    list_revisions,
    record_probe,
    save_endpoint,
)
from src.services.config_store.vlm_snapshot import fetch_revision
from src.services.config_store.vlm_usage import activations_by_project, slugs_running
from src.services.config_store.vlm_validation import external_policy, issue, validate_vlm_endpoint
from src.services.labeling.vlm_endpoint_probe import ProbeBusyError, probe_endpoint
from src.services.labeling.vlm_endpoints import (
    ENV_ENDPOINT_NAME,
    ENV_KEY_REF,
    VlmEndpoint,
    available_vlm_endpoints,
    get_vlm_endpoint,
    resolve_api_key,
    stored_endpoint,
)


if TYPE_CHECKING:
    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody

logger = get_logger(__name__)

#: Errors after which probing would be wrong (the URL must never be
#: contacted) or pointless.
_NO_PROBE_CODES = frozenset(
    {
        'vlm_url_invalid',
        'vlm_url_denied_internal_service',
        'vlm_url_denied_address',
        'vlm_api_key_ref_invalid',
        'vlm_external_denied',
    }
)
_PREFIX = '/vlm/endpoints'


async def _registry(client: Any) -> Any:
    store = get_global_config_store()
    await store.ensure_fresh(client)
    return store.current


async def _active_in(client: Any) -> dict[str, list[str]]:
    """``endpoint name -> projects running it`` for display; a project whose
    activation cannot be read is left out (display only: a delete re-reads
    live and refuses)."""
    try:
        activations = await activations_by_project(client)
    except Exception as exc:
        logger.warning('vlm_active_in_unavailable', error=str(exc))
        return {}
    names = {(doc or {}).get('name') for doc in activations.values() if doc} | {ENV_ENDPOINT_NAME}
    return {n: slugs_running(activations, n) for n in names if n}


def _require_existing(name: str) -> VlmEndpoint:
    endpoint = get_vlm_endpoint(name)
    if endpoint is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known VLM endpoint')
    return endpoint


def _require_stored(name: str) -> VlmEndpoint:
    endpoint = _require_existing(name)
    if endpoint.source == 'env':
        raise api_error(403, 'read_only', 'the built-in env endpoint is read-only; clone it')
    return endpoint


def _all_names() -> set[str]:
    return {e.name for e in available_vlm_endpoints()}


async def _validated(body: VlmEndpointBody, *, name: str | None, existing: set[str] | None) -> Any:
    report, _locality = await validate_vlm_endpoint(
        body, name=name, probe=None, for_activation=False, existing_names=existing
    )
    if not report.ok:
        raise api_error(
            422, 'validation_failed', f'{len(report.errors)} blocking error(s)', report=report
        )
    return report


async def _run_probe(
    client: Any,
    body: VlmEndpointBody,
    *,
    name: str | None,
    is_env: bool,
) -> VlmProbeResult:
    """Probe ``body`` and, when ``name`` is an existing endpoint, record the
    result. 429 ``probe_busy`` over the per-process cap."""
    body = body.normalized()
    try:
        record = await probe_endpoint(
            body, api_key=resolve_api_key(body.api_key_ref, is_env_builtin=is_env)
        )
    except ProbeBusyError as exc:
        raise api_error(429, 'probe_busy', 'too many probes are running; retry shortly') from exc
    if name is not None:
        await record_probe(client, name=name, body=body, record=record)
    return probe_wire(record)  # type: ignore[return-value]


@global_router.get(_PREFIX, response_model=VlmEndpointList, tags=['VLM'])
async def list_vlm_endpoints(client: OpenSearchDep) -> VlmEndpointList:
    """Every endpoint (the ``env`` built-in first), where each is in use,
    the external-images policy and the secret references present."""
    snapshot = await _registry(client)
    usage = await _active_in(client)
    return VlmEndpointList(
        endpoints=[
            await summary_of(e, active_in=usage.get(e.name, [])) for e in available_vlm_endpoints()
        ],
        config_revision=snapshot.config_revision,
        stale=snapshot.stale,
        external_policy=external_policy(),  # type: ignore[arg-type]
        secret_refs=secret_refs_wire(),
        labels=labels_wire(),
    )


@global_router.get(f'{_PREFIX}/schema', response_model=VlmEndpointSchema, tags=['VLM'])
async def vlm_endpoint_form_schema() -> VlmEndpointSchema:
    """Field specs for the endpoint form."""
    return endpoint_schema()


@global_router.post(f'{_PREFIX}/validate', response_model=VlmValidateResponse, tags=['VLM'])
async def validate_vlm_draft(
    payload: VlmValidateRequest,
    client: OpenSearchDep,
    probe: Annotated[bool, Query(description='Also test the endpoint (synthetic images).')] = False,
) -> VlmValidateResponse:
    """Validate a draft; never writes (except the probe record of an
    existing ``name``, when ``probe=true``). Always 200: the report says what
    is wrong."""
    await _registry(client)
    existing = _all_names() if payload.name else None
    report, locality = await validate_vlm_endpoint(
        payload.body,
        name=payload.name,
        probe=None,
        for_activation=False,
        existing_names=existing,
        is_env=payload.name == ENV_ENDPOINT_NAME,
    )
    result: VlmProbeResult | None = None
    if probe and not any(e.code in _NO_PROBE_CODES for e in report.errors):
        target = payload.name if payload.name and get_vlm_endpoint(payload.name) else None
        # Only a stored endpoint's probe is recorded; the built-in is probed
        # through POST /vlm/endpoints/env/probe.
        result = await _run_probe(
            client,
            payload.body,
            name=target if target != ENV_ENDPOINT_NAME else None,
            is_env=False,
        )
    from src.services.labeling.vlm_url_policy import sends_images_externally

    return VlmValidateResponse(
        validation=report,
        locality=locality,
        sends_images_externally=locality is not None and sends_images_externally(locality),
        probe=result,
    )


@global_router.get(f'{_PREFIX}/{{name}}', response_model=VlmEndpointDoc, tags=['VLM'])
async def get_vlm_endpoint_doc(
    name: str, client: OpenSearchDep, response: Response
) -> VlmEndpointDoc:
    await _registry(client)
    endpoint = _require_existing(name)
    usage = await _active_in(client)
    response.headers['ETag'] = f'"{endpoint.etag}"'
    return await doc_of(endpoint, active_in=usage.get(name, []))


@global_router.get(
    f'{_PREFIX}/{{name}}/revisions', response_model=VlmRevisionsResponse, tags=['VLM']
)
async def list_vlm_endpoint_revisions(name: str, client: OpenSearchDep) -> VlmRevisionsResponse:
    await _registry(client)
    endpoint = _require_existing(name)
    if endpoint.source == 'env':
        return VlmRevisionsResponse(name=name, revisions=[])
    return VlmRevisionsResponse(
        name=name,
        revisions=[
            VlmRevisionSummary(
                revision=int(doc['revision']),
                saved_at=doc.get('updated_at'),
                cloned_from=doc.get('cloned_from'),
                description=doc.get('description') or '',
            )
            for doc in await list_revisions(client, name)
        ],
    )


@global_router.get(
    f'{_PREFIX}/{{name}}/revisions/{{revision}}', response_model=VlmEndpointDoc, tags=['VLM']
)
async def get_vlm_endpoint_revision(
    name: str, revision: int, client: OpenSearchDep
) -> VlmEndpointDoc:
    registry = await _registry(client)
    _require_stored(name)
    item = await fetch_revision(client, name, revision)
    if item is None:
        raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    endpoint = stored_endpoint(item, registry.vlm_probes.get(name))
    return await doc_of(endpoint, active_in=[])


@global_router.post(_PREFIX, response_model=VlmEndpointDoc, status_code=201, tags=['VLM'])
async def create_vlm_endpoint(payload: VlmEndpointCreate, client: OpenSearchDep) -> VlmEndpointDoc:
    await _registry(client)
    report = await _validated(payload.body, name=payload.name, existing=_all_names())
    try:
        endpoint = await save_endpoint(
            client,
            name=payload.name,
            body=payload.body,
            expected_revision=None,
            description=payload.description,
        )
    except RevisionConflictError as exc:
        raise api_error(409, 'name_conflict', f'{payload.name!r} already exists') from exc
    return await doc_of(endpoint, active_in=[], validation=report)


@global_router.post(
    f'{_PREFIX}/{{name}}/clone', response_model=VlmEndpointDoc, status_code=201, tags=['VLM']
)
async def clone_vlm_endpoint(
    name: str, payload: VlmEndpointCloneRequest, client: OpenSearchDep
) -> VlmEndpointDoc:
    registry = await _registry(client)
    source = _require_existing(name)
    if payload.revision is not None and source.source == 'stored':
        item = await fetch_revision(client, name, payload.revision)
        if item is None:
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {payload.revision}')
        source = stored_endpoint(item, registry.vlm_probes.get(name))
    body = source.body
    dropped = source.source == 'env' and body.api_key_ref == ENV_KEY_REF
    if dropped:
        # The built-in's own env reference is valid only on the built-in
        # (it would let a clone send the deployment's key elsewhere).
        body = body.model_copy(update={'api_key_ref': None})
    report = await _validated(body, name=payload.new_name, existing=_all_names())
    if dropped:
        report = report.model_copy(
            update={
                'warnings': [
                    *report.warnings,
                    issue(
                        'vlm_api_key_ref_dropped',
                        'warning',
                        'The built-in endpoint reads its key from the environment; a clone '
                        'cannot. Pick a key reference for the clone.',
                        field='api_key_ref',
                        detail={'dropped': ENV_KEY_REF, 'suggest': 'secret:<new_name>'},
                    ),
                ]
            }
        )
    try:
        endpoint = await save_endpoint(
            client,
            name=payload.new_name,
            body=body,
            expected_revision=None,
            description=payload.description
            if payload.description is not None
            else source.description,
            cloned_from=name if source.source == 'env' else f'{name}@{source.revision}',
        )
    except RevisionConflictError as exc:
        raise api_error(409, 'name_conflict', f'{payload.new_name!r} already exists') from exc
    return await doc_of(endpoint, active_in=[], validation=report)


@global_router.put(f'{_PREFIX}/{{name}}', response_model=VlmEndpointDoc, tags=['VLM'])
async def update_vlm_endpoint(
    name: str, payload: VlmEndpointSaveRequest, client: OpenSearchDep
) -> VlmEndpointDoc:
    """A new revision. Does not change what any project runs until it is
    activated there."""
    await _registry(client)
    current = _require_stored(name)
    report = await _validated(payload.body, name=None, existing=None)
    try:
        endpoint = await save_endpoint(
            client,
            name=name,
            body=payload.body,
            expected_revision=payload.expected_revision,
            description=payload.description
            if payload.description is not None
            else current.description,
        )
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{name!r} changed since revision {payload.expected_revision}',
            current_revision=exc.current_revision,
        ) from exc
    return await doc_of(endpoint, active_in=[], validation=report)


@global_router.delete(f'{_PREFIX}/{{name}}', status_code=204, tags=['VLM'])
async def delete_vlm_endpoint(
    name: str, client: OpenSearchDep, expected_revision: Annotated[int, Query()]
) -> Response:
    """409 ``in_use`` while any project runs it (read live, so an unreadable
    project blocks the delete rather than being skipped)."""
    await _registry(client)
    _require_stored(name)
    projects = slugs_running(await activations_by_project(client), name)
    if projects:
        raise api_error(
            409,
            'in_use',
            f'{name!r} is the active VLM of {len(projects)} project(s)',
            projects=projects,
        )
    try:
        await delete_endpoint(client, name=name, expected_revision=expected_revision)
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            f'{name!r} changed since revision {expected_revision}',
            current_revision=exc.current_revision,
        ) from exc
    return Response(status_code=204)


@global_router.post(f'{_PREFIX}/{{name}}/probe', response_model=VlmProbeResult, tags=['VLM'])
async def probe_vlm_endpoint(name: str, client: OpenSearchDep) -> VlmProbeResult:
    """Test the saved endpoint with synthetic images and record the result
    (it also picks up a rotated key file). Never bumps the endpoint's
    revision; bumps the config revision so every process and worker sees
    the new ``json_mode`` / model facts."""
    await _registry(client)
    endpoint = _require_existing(name)
    report, locality = await validate_vlm_endpoint(
        endpoint.body,
        name=None,
        probe=None,
        for_activation=False,
        is_env=endpoint.source == 'env',
    )
    if any(e.code in _NO_PROBE_CODES for e in report.errors):
        raise api_error(
            422, 'validation_failed', 'this endpoint may not be contacted', report=report
        )
    del locality
    return await _run_probe(client, endpoint.body, name=name, is_env=endpoint.source == 'env')
