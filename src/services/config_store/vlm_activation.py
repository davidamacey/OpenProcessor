"""Per-project VLM activation (W9.3).

Which endpoint a project's VLM calls is that project's ``activation:vlm``
doc; the endpoints themselves are global
(:mod:`src.services.config_store.vlm_endpoints`). Every activation writer --
``POST /vlm/endpoints/{name}/activate``, rollback, deactivate, ``PUT
/settings`` and a project clone's activation copy -- goes through
:func:`activate_vlm` (or :func:`deactivate_vlm`), which runs
:func:`~src.services.config_store.vlm_gate.enforce_vlm_gate` first and then
writes, applies the result to this process's snapshot and publishes
``config.changed``. The activation doc also carries ``acked_refs`` (every
external ``name@revision`` this project has acknowledged, with the time) and
``external_ack_at`` (the ack of the ACTIVE endpoint): a later PUT to an
endpoint is a new revision and needs a new ack.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from src.core.logging import get_logger
from src.services.config_store import get_config_store
from src.services.config_store.index import activate as _activate, get_activation
from src.services.config_store.vlm_gate import (
    GateMode,
    GateResult,
    ProjectVlmState,
    current_pack_and_profile,
    enforce_vlm_gate,
    project_vlm_state,
)
from src.services.config_store.vlm_snapshot import fetch_revision
from src.services.labeling.vlm_endpoints import (
    ENV_ENDPOINT_NAME,
    VlmEndpoint,
    get_vlm_endpoint,
    refresh_vlm_state,
    resolve_vlm_source,
)


if TYPE_CHECKING:
    from src.services.config_store.store import AxisRef

logger = get_logger(__name__)

_MAX_ACKED_REFS = 200


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def expected_active_for(ref: AxisRef) -> dict[str, Any] | None:
    """The ``expected_active`` an OCC write must present for the current
    ``ref``: ``None`` = never activated; an explicit off is the stored
    ``{name: None, revision: None}`` (NOT Python ``None``)."""
    if isinstance(ref, tuple):
        return {'name': ref[0], 'revision': ref[1]}
    if ref == 'off':
        return {'name': None, 'revision': None}
    return None


def _carry(endpoint: VlmEndpoint | None, *, ack_now: bool):  # type: ignore[no-untyped-def]
    """The activation doc's extras, computed from the CURRENT doc inside the
    same OCC step so a concurrent activation's acknowledgements are never
    lost."""

    def fields(current: dict[str, Any] | None) -> dict[str, Any]:
        acked: dict[str, str] = dict((current or {}).get('acked_refs') or {})
        if endpoint is not None and ack_now:
            acked.setdefault(endpoint.ref, _now())
        if len(acked) > _MAX_ACKED_REFS:
            newest = sorted(acked.items(), key=lambda kv: kv[1])[-_MAX_ACKED_REFS:]
            acked = dict(newest)
        ack_at = acked.get(endpoint.ref) if endpoint is not None else None
        return {'acked_refs': acked, 'external_ack_at': ack_at}

    return fields


async def write_vlm_activation(
    client: Any,
    endpoint: VlmEndpoint | None,
    *,
    expected_active: dict[str, Any] | None,
    ack_now: bool,
) -> dict[str, Any]:
    from src.services.curation.event_hub import get_event_hub

    store = get_config_store()
    name = endpoint.name if endpoint is not None else None
    revision = endpoint.revision if endpoint is not None else None
    body = None
    if endpoint is not None and endpoint.source == 'stored' and revision is not None:
        body = await fetch_revision(client, endpoint.name, revision)
    result = await _activate(
        client,
        store.index,
        axis='vlm',
        name=name,
        revision=revision,
        expected_active=expected_active,
        doc_fields=_carry(endpoint, ack_now=ack_now),
    )
    ref: AxisRef = (name, revision) if name is not None else 'off'
    store.apply_local(
        config_revision=result['config_revision'],
        active_vlm=ref,
        active_vlm_body=body,
        active_vlm_ack_at=result.get('external_ack_at'),
        acked_refs=frozenset(result.get('acked_refs') or ()),
    )
    get_event_hub().publish(
        {
            'type': 'config.changed',
            'topic': 'config',
            'axis': 'vlm',
            'name': name,
            'config_revision': result['config_revision'],
        }
    )
    return result


async def _target(client: Any, name: str, revision: int | None) -> VlmEndpoint:
    from src.routers.curation._config_common_models import api_error

    endpoint = await resolve_vlm_source(client, name, revision)
    if endpoint is None:
        if revision is not None and get_vlm_endpoint(name) is not None:
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
        raise api_error(404, 'not_found', f'{name!r} is not a known VLM endpoint')
    return endpoint


#: "No override supplied: pair against the project's currently STORED
#: pack / profile". A real ``None`` override means "that side is being
#: cleared in this same request" (the R5-1 lesson: never fall back to the
#: stored value for a side a combined request is changing).
UNSET: Any = object()


@dataclass(frozen=True)
class PreparedVlmActivation:
    """A target that passed the gate and is ready to be written."""

    endpoint: VlmEndpoint
    gate: GateResult


async def prepare_vlm_activation(
    client: Any,
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
    Raises ``api_error`` (404 / 422)."""
    await refresh_vlm_state(client)
    endpoint = await _target(client, name, revision)
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
    client: Any,
    prepared: PreparedVlmActivation,
    *,
    expected_active: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write a prepared activation (OCC on ``expected_active``), apply it to
    this process's snapshot and publish ``config.changed``."""
    endpoint, gate = prepared.endpoint, prepared.gate
    result = await write_vlm_activation(
        client, endpoint, expected_active=expected_active, ack_now=gate.ack_now
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
    client: Any,
    *,
    name: str,
    revision: int | None,
    expected_active: dict[str, Any] | None,
    mode: GateMode = 'activate',
    force: bool = False,
    acknowledge_external: bool = False,
    state: ProjectVlmState | None = None,
) -> tuple[dict[str, Any], GateResult]:
    """Gate, then activate ``name@revision`` in the bound project. Raises
    ``api_error`` (404 / 422) or :class:`ActiveConflictError`."""
    prepared = await prepare_vlm_activation(
        client,
        name=name,
        revision=revision,
        mode=mode,
        force=force,
        acknowledge_external=acknowledge_external,
        state=state,
    )
    result = await apply_vlm_activation(client, prepared, expected_active=expected_active)
    return result, prepared.gate


async def deactivate_vlm(client: Any, *, expected_active: dict[str, Any] | None) -> dict[str, Any]:
    """VLM off for the bound project (nothing to gate)."""
    await refresh_vlm_state(client)
    return await write_vlm_activation(client, None, expected_active=expected_active, ack_now=False)


async def rollback_vlm(
    client: Any, *, expected_active: dict[str, Any] | None
) -> tuple[dict[str, Any], GateResult | None]:
    """Re-activate ``previous`` through the same gate (an endpoint that was
    validly active once is not necessarily valid now). ``LookupError``
    ``no_previous`` / ``previous_deleted`` with nothing written."""
    await refresh_vlm_state(client)
    store = get_config_store()
    doc = await get_activation(client, store.index, 'vlm')
    previous = (doc or {}).get('previous')
    if not previous:
        msg = 'no_previous'
        raise LookupError(msg)
    prev_name, prev_revision = previous.get('name'), previous.get('revision')
    if prev_name is None:
        return (await deactivate_vlm(client, expected_active=expected_active)), None
    if prev_name != ENV_ENDPOINT_NAME and get_vlm_endpoint(prev_name) is None:
        msg = 'previous_deleted'
        raise LookupError(msg)
    return await activate_vlm(
        client,
        name=prev_name,
        revision=prev_revision,
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
    client: Any, value: str | None, *, pack: Any = UNSET, profile: Any = UNSET
) -> DefaultVlmPlan:
    """Resolve and gate ``PUT /settings {"defaults": {"vlm": value}}`` without
    writing: a name = that endpoint's latest revision; ``'off'`` = VLM off;
    ``None`` = clear the project's choice so the ``env`` built-in applies
    (off when there is none). ``pack`` / ``profile`` are the pending sides
    of a combined request. Goes through the one gate like every other
    activation."""
    await refresh_vlm_state(client)
    expected = expected_active_for(get_config_store().current.active_vlm)
    if value == 'off':
        return DefaultVlmPlan(None, expected)
    if value is None:
        from src.services.labeling.vlm_endpoints import env_builtin

        if env_builtin() is None:
            return DefaultVlmPlan(None, expected)
        value = ENV_ENDPOINT_NAME
    prepared = await prepare_vlm_activation(
        client, name=value, revision=None, mode='settings', pack=pack, profile=profile
    )
    return DefaultVlmPlan(prepared, expected)


async def apply_default_vlm(client: Any, plan: DefaultVlmPlan) -> dict[str, Any]:
    if plan.prepared is None:
        return await write_vlm_activation(
            client, None, expected_active=plan.expected_active, ack_now=False
        )
    return await apply_vlm_activation(client, plan.prepared, expected_active=plan.expected_active)


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
