"""The ONE gate every VLM selection path calls (W9.3, W9.5, W9.9).

Anything that picks or switches the endpoint a project's VLM calls go
through -- ``POST /vlm/endpoints/{name}/activate``, rollback, ``PUT
/settings``, a project clone's activation copy, a per-run ``?vlm=`` on the
pipeline and the VLM routes, and the draft tests -- must call
:func:`enforce_vlm_gate` before it acts. The lesson of W3/W4 (the same
gate bypass found four times because each caller re-implemented the
checks): the checks live here and nowhere else, and
``tests/curation/test_vlm_selection_paths.py`` walks every path with an
input that has to be refused.

What it enforces, in one pass:

- endpoint validation (:func:`validate_vlm_endpoint`): URL syntax, the
  never-allowed targets (own services, metadata / link-local addresses),
  secret references, the external-images policy and, for activations, that a
  probe exists and did not fail;
- the external-images acknowledgement, per mode (see :func:`_ack_ok`):
  activation needs ``acknowledge_external`` (or a recorded ack of this exact
  ``name@revision`` in the project), rollback / settings / clone need the
  recorded ack, a one-off run or test needs the flag unless the endpoint is
  the project's acknowledged default;
- the pack / profile pairing (:func:`vlm_pairing_issues`).

``force`` is honoured only on ``activate`` and only for
:data:`~src.services.config_store.vlm_validation.BYPASSABLE_CODES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store.vlm_pairing import vlm_pairing_issues
from src.services.config_store.vlm_validation import (
    BYPASSABLE_CODES,
    build_report,
    issue,
    validate_vlm_endpoint,
)
from src.services.labeling.vlm_endpoints import external_ack_state
from src.services.labeling.vlm_url_policy import sends_images_externally


if TYPE_CHECKING:
    from src.config.detection_profile import DetectionProfile
    from src.routers.curation._config_common_models import ValidationIssue, ValidationReport
    from src.services.labeling.vlm_endpoints import ExternalAckState, VlmEndpoint
    from src.services.labeling.vlm_prompts import PromptPack
    from src.services.labeling.vlm_url_policy import Locality

GateMode = Literal['activate', 'rollback', 'settings', 'clone', 'run', 'test']
ACTIVATION_MODES: frozenset[str] = frozenset({'activate', 'rollback', 'settings', 'clone'})


@dataclass(frozen=True)
class ProjectVlmState:
    """What the bound project has recorded: its active ``name@revision``,
    the timestamp of that activation's acknowledgement, and every external
    ``name@revision`` it has acknowledged."""

    active_ref: str | None
    active_ack_at: str | None
    acked_refs: frozenset[str]


@dataclass(frozen=True)
class GateResult:
    report: ValidationReport
    locality: Locality | None
    ack: ExternalAckState
    #: An external endpoint is being acknowledged by THIS call (record it).
    ack_now: bool


def project_vlm_state() -> ProjectVlmState:
    """The bound project's VLM state from its snapshot (callers refreshed
    it with ``refresh_vlm_state`` first)."""
    from src.services.config_store import get_config_store
    from src.services.labeling.vlm_endpoints import env_builtin

    snapshot = get_config_store().current
    ref = snapshot.active_vlm
    active_ref: str | None = None
    if isinstance(ref, tuple):
        if ref[0] == 'env':
            env = env_builtin()
            active_ref = env.ref if env else None
        elif snapshot.active_vlm_body is not None and ref[1] is not None:
            active_ref = f'{ref[0]}@{ref[1]}'
    return ProjectVlmState(active_ref, snapshot.active_vlm_ack_at, snapshot.acked_refs)


def current_pack_and_profile() -> tuple[PromptPack | None, DetectionProfile | None]:
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.labeling.vlm_prompts import active_prompt_pack

    return active_prompt_pack(), get_active_region_profile()


def _registry_class_names() -> list[str]:
    from src.clients.curation_opensearch import get_class_registry

    try:
        return sorted(c.class_name for c in get_class_registry().load().classes if not c.deprecated)
    except Exception:
        return []


def _ack_ok(
    mode: GateMode,
    endpoint: VlmEndpoint,
    ack: ExternalAckState,
    *,
    flag: bool,
    state: ProjectVlmState,
) -> bool:
    if endpoint.source == 'env' or not ack.sends_images_externally:
        return True
    if mode == 'activate':
        return flag or endpoint.ref in state.acked_refs
    if mode in ('rollback', 'settings', 'clone'):
        # No request-level flag exists on these paths: only a recorded ack
        # (an earlier activation of this exact revision in this project).
        return endpoint.ref in state.acked_refs
    return flag or not ack.per_run_ack_required


def _refusal(
    mode: GateMode, endpoint: VlmEndpoint, report: ValidationReport, blocking: list[ValidationIssue]
) -> Any:
    from src.routers.curation._config_common_models import api_error

    only_ack = all(e.code == 'vlm_external_not_acknowledged' for e in blocking)
    if only_ack:
        via = f'POST /vlm/endpoints/{endpoint.name}/activate'
        return api_error(
            422,
            'vlm_external_not_acknowledged',
            f'{endpoint.name!r} sends crops to a service outside this deployment and has not been '
            'acknowledged for this use.',
            endpoint=endpoint.name,
            activate_via=via,
            report=report,
        )
    return api_error(
        422,
        'validation_failed',
        f'{endpoint.name!r} has {len(blocking)} blocking error(s) ({mode})',
        report=report,
    )


#: ``pending_vlm`` default: pair against the project's stored active VLM.
#: A real ``None`` means the VLM is being switched off in the same request.
UNSET_VLM: Any = object()


async def active_vlm_pairing_issues(
    client: Any,
    pack: PromptPack | None,
    profile: DetectionProfile | None,
    *,
    pending_vlm: Any = UNSET_VLM,
) -> list[ValidationIssue]:
    """The pairing issues between the project's ACTIVE VLM and the
    ``(pack, profile)`` about to be activated: the hook every pack / profile
    activation (``run_activation_gate``) calls, so a change to one side of
    the pack <-> profile <-> VLM triple is checked against the other two.

    An unresolvable VLM state (off, unconfigured, an activation that cannot
    be resolved) yields no pairing issues: a pack or profile is never held
    hostage to a VLM problem the VLM routes report on their own."""
    from src.services.labeling.vlm_endpoints import (
        VlmEndpointUnavailableError,
        active_vlm_endpoint,
        refresh_vlm_state,
    )

    if pending_vlm is not UNSET_VLM:
        # A combined settings request changes the VLM too: pair against the
        # VLM it is about to activate, not the one it replaces.
        endpoint = pending_vlm
    else:
        if client is not None:
            await refresh_vlm_state(client)
        try:
            endpoint = active_vlm_endpoint()
        except VlmEndpointUnavailableError:
            return []
    if endpoint is None:
        return []
    return vlm_pairing_issues(
        endpoint, pack, profile, mode='activate', class_names=_registry_class_names()
    )


async def enforce_vlm_gate(
    endpoint: VlmEndpoint,
    *,
    mode: GateMode,
    force: bool = False,
    acknowledge_external: bool = False,
    pack: PromptPack | None = None,
    profile: DetectionProfile | None = None,
    state: ProjectVlmState | None = None,
    class_names: list[str] | None = None,
) -> GateResult:
    """Refuse (raise ``api_error`` 422) unless ``endpoint`` may be used for
    ``mode``; return the report (its warnings included) otherwise.

    ``state`` defaults to the bound project's; a clone passes the TARGET's.
    """
    state = state or project_vlm_state()
    is_env = endpoint.source == 'env'
    activation = mode in ACTIVATION_MODES
    report, locality = await validate_vlm_endpoint(
        endpoint.body,
        name=None,
        probe=endpoint.last_probe,
        for_activation=activation,
        is_env=is_env,
    )
    issues: list[ValidationIssue] = [*report.errors, *report.warnings]
    ack = external_ack_state(
        endpoint,
        locality or 'unknown',
        active_ref=state.active_ref,
        active_ack_at=state.active_ack_at,
        acked_refs=state.acked_refs,
    )
    ack_ok = _ack_ok(mode, endpoint, ack, flag=acknowledge_external, state=state)
    if (
        not ack_ok
        and locality is not None
        and not any(i.code == 'vlm_external_not_acknowledged' for i in issues)
    ):
        issues.append(
            issue(
                'vlm_external_not_acknowledged',
                'error',
                'Crops would be sent to a service outside this deployment; acknowledge that '
                'to use it.',
            )
        )
    issues.extend(
        vlm_pairing_issues(
            endpoint,
            pack,
            profile,
            mode='activate' if activation else 'run',
            class_names=class_names if class_names is not None else _registry_class_names(),
        )
    )
    # The probe's image-cap error is raised by both the validator (activation)
    # and the pairing check (every mode): report it once.
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[ValidationIssue] = []
    for item in issues:
        key = (item.code, item.severity, item.field)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    full = build_report(unique)
    honoured_force = force and mode == 'activate'
    blocking = [e for e in full.errors if not (honoured_force and e.code in BYPASSABLE_CODES)]
    if blocking:
        raise _refusal(mode, endpoint, full, blocking)
    ack_now = (
        mode == 'activate'
        and not is_env
        and locality is not None
        and sends_images_externally(locality)
    )
    return GateResult(full, locality, ack, ack_now)


__all__ = [
    'ACTIVATION_MODES',
    'UNSET_VLM',
    'GateMode',
    'GateResult',
    'ProjectVlmState',
    'active_vlm_pairing_issues',
    'current_pack_and_profile',
    'enforce_vlm_gate',
    'project_vlm_state',
]
