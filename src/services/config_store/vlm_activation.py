"""Per-project VLM activation.

Which endpoint a project's VLM calls is that project's ``vlm`` activation;
the endpoints themselves are global
(:mod:`src.services.config_store.vlm_endpoints`). Every activation writer --
``POST /vlm/endpoints/{name}/activate``, rollback, deactivate, ``PUT
/settings`` and a project clone's activation copy -- goes through
:func:`activate_vlm` (or :func:`deactivate_vlm`), which runs
:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate` first and then
writes, applies the result to this process's snapshot and publishes
``config.changed``. The project's ``vlm_external_ack`` state records every
external ``name@revision`` it has acknowledged, with the time (written in the
activation's own transaction): a later PUT to an endpoint is a new revision
and needs a new ack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from src.core.logging import get_logger
from src.services.config_store import documents, get_config_store
from src.services.config_store.documents import Registered
from src.services.config_store.store import publish_config_changed
from src.services.config_store.vlm_endpoints import ENV_DESCRIPTION
from src.services.config_store.vlm_gate import (
    GateMode,
    GateResult,
    ProjectVlmState,
    current_pack_and_profile,
    enforce_vlm_gate,
    project_vlm_state,
)
from src.services.labeling.vlm_endpoints import (
    ENV_ENDPOINT_NAME,
    VlmEndpoint,
    VlmRegistryUnavailableError,
    get_vlm_endpoint,
    refresh_vlm_state,
    resolve_vlm_source,
)


if TYPE_CHECKING:
    from src.services.config_store.documents import Activated
    from src.services.config_store.snapshot import AxisRef

logger = get_logger(__name__)


def expected_active_for(ref: AxisRef) -> dict[str, Any] | None:
    """The ``expected_active`` an OCC write must present for the current
    ``ref``: ``None`` = never activated; an explicit off is the stored
    ``{name: None, revision: None}`` (NOT Python ``None``)."""
    if isinstance(ref, tuple):
        return {'name': ref[0], 'revision': ref[1]}
    if ref == 'off':
        return {'name': None, 'revision': None}
    return None


async def write_vlm_activation(
    endpoint: VlmEndpoint | None,
    *,
    expected_active: dict[str, Any] | None,
    ack_now: bool,
    reason: Literal['activate', 'rollback', 'clone'] = 'activate',
) -> Activated:
    store = get_config_store()
    name = endpoint.name if endpoint is not None else None
    revision = endpoint.revision if endpoint is not None else None
    body = None
    registered = None
    if endpoint is not None and endpoint.source == 'stored' and revision is not None:
        body = await documents.get_revision(
            kind='vlm_endpoint', name=endpoint.name, revision=revision
        )
    elif endpoint is not None and endpoint.source == 'env':
        registered = Registered(endpoint.body.model_dump(), ENV_DESCRIPTION)
    result = await documents.activate(
        axis='vlm',
        name=name,
        revision=revision,
        expected_active=expected_active,
        registered=registered,
        ack_ref=endpoint.ref if endpoint is not None and ack_now else None,
        reason=reason,
    )
    acked = await documents.get_acked_refs()
    store.apply_local(
        config_revision=result.epoch,
        active_vlm=result.ref,
        active_vlm_body=body,
        active_vlm_ack_at=acked.get(endpoint.ref) if endpoint is not None else None,
        acked_refs=frozenset(acked),
    )
    publish_config_changed('vlm', name, result.epoch)
    return result


async def _target(name: str, revision: int | None) -> VlmEndpoint:
    from src.services.rejections import rejection

    try:
        endpoint = await resolve_vlm_source(name, revision)
    except VlmRegistryUnavailableError as exc:
        raise rejection(503, 'config_store_unavailable', str(exc)) from exc
    if endpoint is None:
        if revision is not None and get_vlm_endpoint(name) is not None:
            raise rejection(404, 'unknown_revision', f'{name!r} has no revision {revision}')
        raise rejection(404, 'not_found', f'{name!r} is not a known VLM endpoint')
    return endpoint


#: "No override supplied: pair against the project's currently STORED
#: pack / profile". A real ``None`` override means "that side is being
#: cleared in this same request" (never fall back to the
#: stored value for a side a combined request is changing).
UNSET: Any = object()


@dataclass(frozen=True)
class PreparedVlmActivation:
    """A target that passed the gate and is ready to be written."""

    endpoint: VlmEndpoint
    gate: GateResult


async def prepare_vlm_activation(
    *,
    name: str,
    revision: int | None,
    mode: GateMode = 'activate',
    force: bool = False,
    acknowledge_external: bool = False,
    state: ProjectVlmState | None = None,
    pack: Any = UNSET,
    profile: Any = UNSET,
) -> PreparedVlmActivation:
    """Resolve ``name@revision`` (``revision=None`` = its latest, pinned to a
    concrete number) and run the gate; write nothing. ``pack`` / ``profile``
    are the PENDING values of a combined settings request (see :data:`UNSET`).
    Raises a rejection (404 / 422)."""
    await refresh_vlm_state()
    endpoint = await _target(name, revision)
    current_pack, current_profile = current_pack_and_profile()
    gate = await enforce_vlm_gate(
        endpoint,
        mode=mode,
        force=force,
        acknowledge_external=acknowledge_external,
        pack=current_pack if pack is UNSET else pack,
        profile=current_profile if profile is UNSET else profile,
        state=state,
    )
    return PreparedVlmActivation(endpoint, gate)


async def apply_vlm_activation(
    prepared: PreparedVlmActivation,
    *,
    expected_active: dict[str, Any] | None,
    reason: Literal['activate', 'rollback', 'clone'] = 'activate',
) -> Activated:
    """Write a prepared activation (OCC on ``expected_active``), apply it to
    this process's snapshot and publish ``config.changed``."""
    endpoint, gate = prepared.endpoint, prepared.gate
    result = await write_vlm_activation(
        endpoint, expected_active=expected_active, ack_now=gate.ack_now, reason=reason
    )
    if gate.ack.sends_images_externally and endpoint.source != 'env':
        # Once per activation, the host only -- never the key.
        logger.info(
            'vlm_external_endpoint_activated',
            endpoint=endpoint.name,
            host=urlsplit(endpoint.body.base_url).hostname,
        )
    return result


async def activate_vlm(
    *,
    name: str,
    revision: int | None,
    expected_active: dict[str, Any] | None,
    mode: GateMode = 'activate',
    force: bool = False,
    acknowledge_external: bool = False,
    state: ProjectVlmState | None = None,
) -> tuple[Activated, GateResult]:
    """Gate, then activate ``name@revision`` in the bound project. Raises
    a rejection (404 / 422) or :class:`ActiveConflictError`."""
    prepared = await prepare_vlm_activation(
        name=name,
        revision=revision,
        mode=mode,
        force=force,
        acknowledge_external=acknowledge_external,
        state=state,
    )
    result = await apply_vlm_activation(
        prepared,
        expected_active=expected_active,
        reason='rollback' if mode == 'rollback' else 'activate',
    )
    return result, prepared.gate


async def deactivate_vlm(*, expected_active: dict[str, Any] | None) -> Activated:
    """VLM off for the bound project (nothing to gate)."""
    await refresh_vlm_state()
    return await write_vlm_activation(None, expected_active=expected_active, ack_now=False)


async def rollback_vlm(
    *, expected_active: dict[str, Any] | None
) -> tuple[Activated, GateResult | None]:
    """Re-activate what the VLM axis moved away from through the same gate (an endpoint that was
    validly active once is not necessarily valid now). ``LookupError`` ``no_previous`` /
    ``previous_deleted`` with nothing written."""
    await refresh_vlm_state()
    target = await documents.previous_target('vlm')
    if target.name is None:
        off = await write_vlm_activation(
            None, expected_active=expected_active, ack_now=False, reason='rollback'
        )
        return off, None
    if target.name != ENV_ENDPOINT_NAME and get_vlm_endpoint(target.name) is None:
        msg = 'previous_deleted'
        raise LookupError(msg)
    return await activate_vlm(
        name=target.name,
        revision=target.revision,
        expected_active=expected_active,
        mode='rollback',
    )


@dataclass(frozen=True)
class DefaultVlmPlan:
    """A resolved ``PUT /settings`` VLM change: ``prepared`` is the gated
    target (``None`` = VLM off), ``expected_active`` the OCC the write must
    present."""

    prepared: PreparedVlmActivation | None
    expected_active: dict[str, Any] | None

    @property
    def endpoint(self) -> VlmEndpoint | None:
        return self.prepared.endpoint if self.prepared else None


async def plan_default_vlm(
    value: str | None, *, pack: Any = UNSET, profile: Any = UNSET
) -> DefaultVlmPlan:
    """Resolve and gate ``PUT /settings {"defaults": {"vlm": value}}`` without
    writing: a name = that endpoint's latest revision; ``'off'`` or ``None`` =
    VLM off (a project never inherits the ``env`` built-in). ``pack`` / ``profile`` are the pending sides
    of a combined request. Goes through the one gate like every other
    activation."""
    await refresh_vlm_state()
    expected = expected_active_for(get_config_store().current.active_vlm)
    if value is None or value == 'off':
        return DefaultVlmPlan(None, expected)
    prepared = await prepare_vlm_activation(
        name=value, revision=None, mode='settings', pack=pack, profile=profile
    )
    return DefaultVlmPlan(prepared, expected)


async def apply_default_vlm(plan: DefaultVlmPlan) -> Activated:
    if plan.prepared is None:
        return await write_vlm_activation(None, expected_active=plan.expected_active, ack_now=False)
    return await apply_vlm_activation(plan.prepared, expected_active=plan.expected_active)


__all__ = [
    'UNSET',
    'DefaultVlmPlan',
    'PreparedVlmActivation',
    'activate_vlm',
    'apply_default_vlm',
    'apply_vlm_activation',
    'deactivate_vlm',
    'expected_active_for',
    'plan_default_vlm',
    'prepare_vlm_activation',
    'project_vlm_state',
    'rollback_vlm',
    'write_vlm_activation',
]
