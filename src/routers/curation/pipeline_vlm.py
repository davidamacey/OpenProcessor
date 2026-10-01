"""Per-run VLM selection (``?vlm=``) for the auto-label and VLM routes (W9.3).

:func:`resolve_run_vlm` is the request-time resolver, the sibling of
``resolve_run_prompt_pack``: it resolves the endpoint, runs the ONE shared
gate (:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate`, mode
``run``: validity / SSRF, the external-images acknowledgement rule and the
pack pairing) and returns the PINNED ``(name, revision)`` the job carries.
Pinning to a concrete revision at request time is what keeps a job on the
endpoint it was accepted against: an activation (or an edit) made mid-job
changes nothing for it, and the job can resolve the same immutable revision
by id from any process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.routers.curation._config_common_models import api_error
from src.services.config_store.vlm_gate import current_pack_and_profile, enforce_vlm_gate
from src.services.labeling.vlm_endpoints import (
    VlmEndpoint,
    VlmEndpointUnavailableError,
    active_vlm_endpoint,
    available_vlm_endpoints,
    refresh_vlm_state,
    resolve_vlm_source,
)


if TYPE_CHECKING:
    from src.services.labeling.vlm_endpoint_body import VlmEndpointBody

VLM_DESC = (
    'Per-run VLM endpoint (a name, or name@revision, from GET /methods axis=vlm) used by the '
    "labeling stage. Overrides this project's default for this run only; never written to "
    'settings. Unset = the project default. Unknown -> 422. An endpoint outside this deployment '
    "also needs acknowledge_external=true, unless it is already this project's acknowledged "
    'default.'
)
ACKNOWLEDGE_EXTERNAL_DESC = (
    'Acknowledge that crops will be sent to a VLM endpoint outside this deployment for this run.'
)


#: The one message meaning "this project's VLM is off"; ``_get_vlm_labeler``
#: raises it and ``labeler_unavailable`` maps it to 409 ``vlm_not_configured``.
NO_VLM_MESSAGE = 'no VLM endpoint is active for this project'


@dataclass(frozen=True)
class RunVlm:
    """A run's pinned VLM. ``endpoint is None``: this project's VLM is off
    (or unconfigured) and none was asked for."""

    name: str | None
    revision: int | None
    endpoint: VlmEndpoint | None


def _unknown(requested: str) -> Any:
    return api_error(
        422,
        'unknown_vlm',
        f'{requested!r} is not a known VLM endpoint',
        axis='vlm',
        requested=requested,
        valid_ids=[e.name for e in available_vlm_endpoints()],
    )


def _parse(requested: str) -> tuple[str, int | None]:
    name, sep, rev = requested.rpartition('@')
    if not sep:
        return requested, None
    try:
        return name, int(rev)
    except ValueError:
        raise _unknown(requested) from None


async def resolve_run_vlm(
    opensearch: Any,
    vlm: Any,
    *,
    pack: Any,
    acknowledge_external: Any = False,
) -> RunVlm:
    """Resolve, gate and pin ``vlm`` for one run. Non-``str`` (``None``, or
    an unfilled FastAPI ``Query`` default on a direct call) means omitted =
    the bound project's active endpoint."""
    requested = vlm if isinstance(vlm, str) else None
    ack = acknowledge_external is True
    await refresh_vlm_state(opensearch)
    if requested is None:
        try:
            endpoint = active_vlm_endpoint()
        except VlmEndpointUnavailableError as exc:
            raise api_error(409, 'vlm_endpoint_unavailable', str(exc)) from exc
        if endpoint is None:
            return RunVlm(None, None, None)
    else:
        name, revision = _parse(requested)
        endpoint = await resolve_vlm_source(opensearch, name, revision)
        if endpoint is None:
            raise _unknown(requested)
    _pack, profile = current_pack_and_profile()
    await enforce_vlm_gate(
        endpoint, mode='run', acknowledge_external=ack, pack=pack or _pack, profile=profile
    )
    return RunVlm(endpoint.name, endpoint.revision, endpoint)


async def resolve_test_vlm(
    opensearch: Any,
    *,
    name: str | None,
    revision: int | None,
    draft: VlmEndpointBody | None,
    acknowledge_external: bool,
    pack: Any,
    profile: Any,
) -> VlmEndpoint:
    """The endpoint a test-on-crop run (W5) sends a real crop to: a draft
    body, ``name`` (+ ``revision``), or the project's active endpoint, through
    the ONE gate (mode ``test``: validity / SSRF, the external-images
    acknowledgement, secrets by reference, the pack pairing). 409
    ``vlm_not_configured`` when none is given and the project's VLM is off."""
    if name is not None and draft is not None:
        raise api_error(422, 'validation_failed', 'give vlm_name or vlm_draft, not both')
    await refresh_vlm_state(opensearch)
    try:
        endpoint = await resolve_vlm_source(opensearch, name, revision, draft)
    except VlmEndpointUnavailableError as exc:
        raise api_error(409, 'vlm_endpoint_unavailable', str(exc)) from exc
    if endpoint is None:
        if name is not None:
            raise _unknown(name if revision is None else f'{name}@{revision}')
        raise labeler_unavailable(VlmEndpointUnavailableError(NO_VLM_MESSAGE))
    await enforce_vlm_gate(
        endpoint, mode='test', acknowledge_external=acknowledge_external, pack=pack, profile=profile
    )
    return endpoint


def labeler_unavailable(exc: Exception) -> Any:
    """The ``api_error`` for a labeler that could not be built: 409
    ``vlm_not_configured`` when the project's VLM is off, else 409
    ``vlm_endpoint_unavailable`` (an unresolvable or refused endpoint)."""
    if str(exc) == NO_VLM_MESSAGE:
        return api_error(
            409,
            'vlm_not_configured',
            "this project's VLM is off or not configured; activate an endpoint first",
        )
    return api_error(409, 'vlm_endpoint_unavailable', str(exc))


def require_run_vlm(run: RunVlm) -> None:
    """409 when a run needs the VLM and the project's is off."""
    if run.endpoint is None:
        raise labeler_unavailable(VlmEndpointUnavailableError(NO_VLM_MESSAGE))


async def job_endpoint(
    opensearch: Any,
    *,
    vlm: Any,
    acknowledge_external: Any,
    pack: Any,
    pinned_name: str | None,
    pinned_revision: int | None,
    resolved: bool,
) -> VlmEndpoint | None:
    """The endpoint an auto-label run's VLM stage uses.

    ``resolved`` (a job started by ``/pipeline/auto_label/start``): the
    request already resolved and gated the selection and pinned
    ``(pinned_name, pinned_revision)``; this resolves exactly that by id, in
    THIS process (the job may run in the auto-label worker, whose registry
    snapshot is its own and starts cold: ``resolve_vlm_source`` refreshes it
    and fetches an immutable revision copy directly).

    Otherwise (the synchronous public route) the raw ``vlm`` is resolved and
    gated now. ``None`` = the project's VLM is off; the caller's labeler
    build then fails with a clear error.
    """
    if resolved:
        if pinned_name is None:
            return None
        endpoint = await resolve_vlm_source(opensearch, pinned_name, pinned_revision)
        if endpoint is None:
            msg = f'the pinned VLM endpoint {pinned_name}@{pinned_revision} no longer exists'
            raise VlmEndpointUnavailableError(msg)
        return endpoint
    return (
        await resolve_run_vlm(opensearch, vlm, pack=pack, acknowledge_external=acknowledge_external)
    ).endpoint


__all__ = [
    'ACKNOWLEDGE_EXTERNAL_DESC',
    'NO_VLM_MESSAGE',
    'VLM_DESC',
    'RunVlm',
    'job_endpoint',
    'labeler_unavailable',
    'require_run_vlm',
    'resolve_run_vlm',
    'resolve_test_vlm',
]
