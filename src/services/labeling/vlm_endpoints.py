"""VLM endpoints: the one abstraction every VLM call site resolves (W9.0).

A VLM is always a *registered endpoint*: the ``env`` built-in
(``OP_VLM_URL`` / ``OP_VLM_MODEL``, read-only) or a stored endpoint in the
global registry (``op_global_configs``). This module owns

- :class:`VlmEndpoint` (frozen; body + provenance ref + last probe),
- :func:`env_endpoint`, the built-in,
- secrets by reference (:func:`resolve_api_key`, :func:`list_secret_refs`):
  a key is a *reference* to ``/run/secrets/op_vlm/<slug>``, never stored,
  logged or served,
- the external-images acknowledgement rule (:func:`external_ack_state`),
  the single source for both what ``/methods`` serves and what a run enforces,
- the resolvers: :func:`active_vlm_endpoint` / :func:`get_vlm_endpoint`
  (sync, from the in-process snapshots -- callers ``await refresh_vlm_state``
  first) and :func:`resolve_vlm_source` (async, refreshes, pins a revision).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.core.logging import get_logger
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody, VlmProbeRecord
from src.services.labeling.vlm_url_policy import (
    Locality,
    external_warning,
    sends_images_externally,
    strip_userinfo,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.services.config_store.store import ConfigSnapshot, StoredConfig

logger = get_logger(__name__)

ENV_ENDPOINT_NAME = 'env'
ENV_KEY_REF = 'env:OP_VLM_API_KEY'
SECRET_REF_RE = re.compile(r'^secret:([a-z0-9_]{1,64})$')
_SECRET_SLUG_RE = re.compile(r'^[a-z0-9_]{1,64}$')
_MAX_SECRET_BYTES = 8192

#: Names an endpoint may not take (``env`` is the built-in; ``active`` etc.
#: would collide with the ``/vlm/endpoints/{name}`` static siblings).
RESERVED_NAMES: frozenset[str] = frozenset(
    {'env', 'off', 'none', 'default', 'local', 'active', 'schema', 'validate', 'deactivate'}
)
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_.-]{1,63}$')

EndpointStatus = Literal['ready', 'unprobed', 'probe_failed', 'unreachable']

#: Served labels for every enum on the endpoint wire shapes (the enums are
#: Literals in OpenAPI), so no client hardcodes a string.
STATUS_LABELS: dict[str, str] = {
    'ready': 'Ready',
    'unprobed': 'Not yet tested',
    'probe_failed': 'Last test failed',
    'unreachable': 'Unreachable',
}
LOCALITY_LABELS: dict[str, str] = {
    'compose': 'This stack',
    'host': 'This machine',
    'private': 'Private network',
    'external': 'Outside this deployment',
    'unknown': 'Unknown (treated as outside)',
}
SOURCE_LABELS: dict[str, str] = {'env': 'Built in (environment)', 'stored': 'Saved'}
CATALOG_STATUS_LABELS: dict[str, str] = {'tested': 'Tested', 'to_verify': 'Not yet verified'}


class VlmEndpointUnavailableError(RuntimeError):
    """An activation names an endpoint revision that cannot be resolved.
    Fail closed: never fall back to another endpoint."""


class VlmKeyRefInvalidError(ValueError):
    """``api_key_ref`` is not a reference this deployment may resolve."""


def _sha12(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def probe_fingerprint(body: VlmEndpointBody) -> str:
    """What a probe is a probe OF. A probe recorded for an endpoint name
    stays valid only while the fields it exercised are unchanged, so an edit
    to the URL, model, key reference or image cap can never keep serving the
    previous body's ``root`` / ``max_model_len`` / ``max_images_ok``."""
    return _sha12(
        [body.base_url, body.model, body.api_key_ref, body.max_images_per_call],
    )


def probe_key(name: str, revision: int | None) -> str:
    """What a probe is stored under: ``name@revision`` for a stored endpoint
    (probing one revision never touches another's), the bare name for the
    ``env`` built-in, which has no revisions."""
    return name if revision is None else f'{name}@{revision}'


@dataclass(frozen=True)
class VlmEndpoint:
    name: str
    source: Literal['env', 'stored']
    revision: int | None
    body: VlmEndpointBody
    etag: str
    last_probe: VlmProbeRecord | None = None
    description: str = ''
    created_at: str | None = None
    updated_at: str | None = None
    cloned_from: str | None = None

    @property
    def ref(self) -> str:
        """The provenance string stamped on writes: ``name@revision`` for a
        saved endpoint, ``name@<sha12 of the body>`` for one with no
        revision (the ``env`` built-in, an unsaved draft)."""
        if self.revision is None:
            return f'{self.name}@{_sha12(self.body.model_dump())}'
        return f'{self.name}@{self.revision}'

    @property
    def model_id(self) -> str:
        """The resolved model: the probe's ``root`` when known (vLLM reports
        the underlying repo even behind a served alias), else ``body.model``."""
        return (self.last_probe.root if self.last_probe else None) or self.body.model

    @property
    def probe_marker(self) -> str | None:
        """Changes whenever a re-probe changes what a labeler is built from
        (``json_mode`` resolution, ``max_model_len``, the model root); part
        of the labeler cache key and of the worker's swap decision."""
        p = self.last_probe
        if p is None:
            return None
        return _sha12([p.probed_at, p.json_mode_supported, p.root, p.max_model_len])

    @property
    def json_mode_on(self) -> bool:
        """``auto`` follows the last probe (``on`` when never probed)."""
        mode = self.body.json_mode
        if mode == 'auto':
            probe = self.last_probe
            return (
                True
                if probe is None or probe.json_mode_supported is None
                else bool(probe.json_mode_supported)
            )
        return mode == 'on'

    @property
    def status(self) -> EndpointStatus:
        return endpoint_status(self.last_probe)


def endpoint_status(probe: VlmProbeRecord | None) -> EndpointStatus:
    if probe is None:
        return 'unprobed'
    if probe.ok:
        return 'ready'
    if {'vlm_unreachable', 'vlm_timeout'} & set(probe.error_codes):
        return 'unreachable'
    return 'probe_failed'


# --- secrets by reference ------------------------------------------------------


def secrets_dir() -> Path:
    return Path(os.environ.get('OP_VLM_SECRETS_DIR', '/run/secrets/op_vlm'))


def _secret_file(slug: str) -> Path | None:
    """The regular, non-symlink file for ``slug`` with content no larger than
    a key can be, or ``None`` (so a listing never offers a reference that
    would not resolve)."""
    if not _SECRET_SLUG_RE.match(slug):
        return None
    path = secrets_dir() / slug
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if not 0 < path.stat().st_size <= _MAX_SECRET_BYTES:
            return None
    except OSError:
        return None
    return path


def list_secret_refs() -> list[str]:
    """``secret:<slug>`` for every non-empty regular file in the secrets
    mount. Names only: never a value, never a path outside the mount."""
    root = secrets_dir()
    try:
        names = sorted(p.name for p in root.iterdir())
    except OSError:
        return []
    return [f'secret:{n}' for n in names if _secret_file(n) is not None]


def api_key_present(ref: str | None, *, is_env_builtin: bool) -> bool:
    if ref is None:
        return False
    try:
        return bool(_resolve(ref, is_env_builtin=is_env_builtin))
    except VlmKeyRefInvalidError:
        return False


def _resolve(ref: str, *, is_env_builtin: bool) -> str | None:
    if ref == ENV_KEY_REF:
        if not is_env_builtin:
            msg = f'{ENV_KEY_REF} is only valid on the built-in env endpoint'
            raise VlmKeyRefInvalidError(msg)
        return (os.environ.get('OP_VLM_API_KEY') or '').strip() or None
    match = SECRET_REF_RE.match(ref)
    if match is None:
        msg = 'api_key_ref must be secret:<slug>'
        raise VlmKeyRefInvalidError(msg)
    path = _secret_file(match.group(1))
    if path is None:
        return None
    try:
        with path.open('rb') as fh:
            data = fh.read(_MAX_SECRET_BYTES + 1)
    except OSError:
        return None
    if len(data) > _MAX_SECRET_BYTES:
        return None
    return data.decode('utf-8', 'replace').strip() or None


def resolve_api_key(ref: str | None, *, is_env_builtin: bool = False) -> str | None:
    """The key ``ref`` points to, or ``None`` (no ref, or nothing there).
    Raises :class:`VlmKeyRefInvalidError` for a ref this deployment must never
    resolve (any ``env:`` name other than the built-in's own, a path)."""
    return None if ref is None else _resolve(ref, is_env_builtin=is_env_builtin)


# --- the env built-in -------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name) or default))
    except ValueError:
        return default


def env_endpoint(probe_doc: dict[str, Any] | None = None) -> VlmEndpoint | None:
    """The read-only ``env`` endpoint, or ``None`` when ``OP_VLM_URL`` is
    empty (then no VLM exists unless a stored one is activated).
    ``probe_doc`` is the registry's ``vlm_probe:env`` body (used only while
    its fingerprint still matches the env values)."""
    from src.services.labeling.vlm_catalog import catalog_entry_for_root

    raw_url = os.environ.get('OP_VLM_URL', '').strip()
    if not raw_url:
        return None
    # Credentials in the URL are dropped here, at the source, so no route,
    # log line or labeler ever sees them (the key goes in OP_VLM_API_KEY).
    try:
        url = strip_userinfo(raw_url).rstrip('/')
    except ValueError as exc:  # e.g. a non-numeric port: not a usable URL
        logger.warning('vlm_env_url_invalid', error=str(exc))
        return None
    base = VlmEndpointBody(
        base_url=url,
        model=os.environ.get('OP_VLM_MODEL', '').strip(),
        api_key_ref='secret:env' if _secret_file('env') is not None else ENV_KEY_REF,
        max_images_per_call=_env_int('OP_VLM_MAX_IMAGES_PER_CALL', 8),
        open_images_per_call=_env_int('OP_VLM_OPEN_IMAGES_PER_CALL', 3),
        timeout_s=240.0,
        requests_per_second=500.0,
        json_mode='auto',
        allow_external=True,
        catalog_id=None,
    )
    probe = _matching_probe(base, probe_doc)
    entry = catalog_entry_for_root(probe.root if probe else None)
    body = base.model_copy(update={'catalog_id': entry.id if entry else None})
    return VlmEndpoint(
        name=ENV_ENDPOINT_NAME,
        source='env',
        revision=None,
        body=body,
        etag=f'vlm:env:{_sha12(body.model_dump())}',
        last_probe=probe,
        description='From OP_VLM_URL / OP_VLM_MODEL',
    )


def stored_endpoint(item: StoredConfig, probes: Mapping[str, dict[str, Any]]) -> VlmEndpoint:
    """``item`` with ITS revision's probe out of the registry's ``probes``."""
    probe_doc = probes.get(probe_key(item.name, item.revision))
    body = VlmEndpointBody.model_validate(item.body)
    return VlmEndpoint(
        name=item.name,
        source='stored',
        revision=item.revision,
        body=body,
        etag=f'vlm:{item.name}:{item.revision}',
        last_probe=_matching_probe(body, probe_doc),
        description=item.description,
        created_at=item.created_at,
        updated_at=item.updated_at,
        cloned_from=item.cloned_from,
    )


def _matching_probe(
    body: VlmEndpointBody, probe_doc: dict[str, Any] | None
) -> VlmProbeRecord | None:
    if not probe_doc:
        return None
    if probe_doc.get('fingerprint') != probe_fingerprint(body):
        return None
    return VlmProbeRecord.model_validate(probe_doc.get('record') or {})


def draft_endpoint(body: VlmEndpointBody) -> VlmEndpoint:
    """An unsaved body as an endpoint (test-on-crop drafts, W5): never
    probed, never activatable, ref ``_draft@<sha12>``."""
    body = body.normalized()
    return VlmEndpoint(
        name='_draft',
        source='stored',
        revision=None,
        body=body,
        etag=f'vlm:_draft:{_sha12(body.model_dump())}',
    )


# --- the external-images acknowledgement ------------------------------------------


@dataclass(frozen=True)
class ExternalAckState:
    sends_images_externally: bool
    warning: str | None
    default_ack_recorded: bool
    per_run_ack_required: bool


def external_ack_state(
    endpoint: VlmEndpoint,
    locality: Locality,
    *,
    active_ref: str | None,
    active_ack_at: str | None,
    acked_refs: frozenset[str],
) -> ExternalAckState:
    """The ONE statement of the ack rule (W9.9 / §7.8.2): ``/methods``
    serves these flags and every run enforces them from this function, so
    the served flag and the enforced rule cannot drift.

    - ``default_ack_recorded``: this project acknowledged sending crops to
      this exact ``name@revision`` when activating it (an edit that changes
      the URL is a new revision and needs a new ack). ``True`` when nothing
      leaves the deployment.
    - ``per_run_ack_required``: a one-off ``?vlm=`` use of an external
      endpoint needs ``acknowledge_external`` UNLESS it is the project's
      active endpoint at the acknowledged ``name@revision`` (that run sends
      nothing the default doesn't).

    The ``env`` built-in is the operator's own deployment choice (set in
    ``.env``): its warning is served but no ack is asked for.
    """
    sends = sends_images_externally(locality)
    warning = external_warning(endpoint.body.base_url, locality)
    if not sends or endpoint.source == 'env':
        return ExternalAckState(sends, warning, True, False)
    acked = endpoint.ref in acked_refs
    is_acked_default = endpoint.ref == active_ref and bool(active_ack_at)
    return ExternalAckState(sends, warning, acked, not is_acked_default)


# --- resolvers ------------------------------------------------------------------------


def _registry() -> ConfigSnapshot:
    """The deployment-wide registry snapshot as this process last loaded it
    (usable with no project bound). Async callers ``await
    refresh_vlm_state(client)`` first; the worker refreshes its pinned
    stores itself."""
    from src.services.config_store import get_global_config_store

    return get_global_config_store().current


def _project() -> ConfigSnapshot:
    """The BOUND project's snapshot (raises ``ProjectNotBound`` unbound)."""
    from src.services.config_store import get_config_store

    return get_config_store().current


def env_builtin() -> VlmEndpoint | None:
    return env_endpoint(_registry().vlm_probes.get(ENV_ENDPOINT_NAME))


def available_vlm_endpoints() -> list[VlmEndpoint]:
    """The env built-in (when configured) then every stored endpoint at its
    current revision, by name."""
    registry = _registry()
    found = [e for e in [env_builtin()] if e is not None]
    found.extend(
        stored_endpoint(item, registry.vlm_probes)
        for name, item in sorted(registry.vlm_endpoints.items())
    )
    return found


def get_vlm_endpoint(name: str) -> VlmEndpoint | None:
    """``name`` at its CURRENT revision (or the env built-in)."""
    if name == ENV_ENDPOINT_NAME:
        return env_builtin()
    registry = _registry()
    item = registry.vlm_endpoints.get(name)
    return None if item is None else stored_endpoint(item, registry.vlm_probes)


def active_vlm_endpoint() -> VlmEndpoint | None:
    """The bound project's active endpoint, from the snapshots: the
    activation's PINNED revision body (never the endpoint's current doc, so
    a later PUT changes nothing until activated); ``env`` when the project
    never activated one; ``None`` when explicitly switched off.

    Raises :class:`VlmEndpointUnavailableError` when an activation names a
    revision that cannot be resolved (fail closed: no silent fall back)."""
    registry, project = _registry(), _project()
    ref = project.active_vlm
    if ref is None:
        return env_builtin()
    if ref == 'off':
        return None
    name, revision = ref
    if name == ENV_ENDPOINT_NAME:
        return env_builtin()
    body = project.active_vlm_body
    if body is None or body.name != name or body.revision != revision:
        msg = f'the active VLM endpoint {name}@{revision} cannot be resolved'
        raise VlmEndpointUnavailableError(msg)
    return stored_endpoint(body, registry.vlm_probes)


def vlm_configured() -> bool:
    """Whether the bound project has a VLM to call: its active endpoint (the
    ``env`` built-in until it activates one) resolves to something. The one
    answer to "is a VLM configured" for any code that is not itself calling
    it (profile validation); an activation that cannot be resolved counts as
    not configured. Callers refresh the snapshots first."""
    try:
        return active_vlm_endpoint() is not None
    except VlmEndpointUnavailableError:
        return False


async def refresh_vlm_state(client: Any) -> None:
    """Bring the global registry and the bound project's activation up to
    date in THIS process (both are separate snapshots in separate
    processes, so any route or job start that resolves a VLM calls this)."""
    from src.services.config_store import get_config_store, get_global_config_store

    await get_global_config_store().ensure_fresh(client)
    await get_config_store().ensure_fresh(client)


async def resolve_vlm_source(
    client: Any,
    name: str | None = None,
    revision: int | None = None,
    draft: VlmEndpointBody | None = None,
) -> VlmEndpoint | None:
    """The one resolver for "which endpoint": a draft body, a name (latest
    revision) or ``name@revision`` (an immutable copy, fetched by id so it
    resolves in a cold process), or nothing = the bound project's active
    endpoint. ``None`` when the name (or that revision) is unknown, or the
    project's VLM is off."""
    if draft is not None:
        return draft_endpoint(draft)
    if name is None:
        await refresh_vlm_state(client)
        return active_vlm_endpoint()
    # By name the registry alone decides: usable with no project bound.
    from src.services.config_store import get_global_config_store

    await get_global_config_store().ensure_fresh(client)
    endpoint = get_vlm_endpoint(name)
    if endpoint is None:
        return None
    if revision is None:
        return endpoint
    if endpoint.source == 'env':
        return None  # the built-in has no revisions
    if revision == endpoint.revision:
        return endpoint
    from src.services.config_store import vlm_snapshot

    item = await vlm_snapshot.fetch_revision(client, name, revision)
    if item is None:
        return None
    registry = _registry()
    return stored_endpoint(item, registry.vlm_probes)


__all__ = [
    'CATALOG_STATUS_LABELS',
    'ENV_ENDPOINT_NAME',
    'ENV_KEY_REF',
    'LOCALITY_LABELS',
    'NAME_RE',
    'RESERVED_NAMES',
    'SECRET_REF_RE',
    'SOURCE_LABELS',
    'STATUS_LABELS',
    'EndpointStatus',
    'ExternalAckState',
    'VlmEndpoint',
    'VlmEndpointUnavailableError',
    'VlmKeyRefInvalidError',
    'active_vlm_endpoint',
    'api_key_present',
    'available_vlm_endpoints',
    'draft_endpoint',
    'endpoint_status',
    'env_builtin',
    'env_endpoint',
    'external_ack_state',
    'get_vlm_endpoint',
    'list_secret_refs',
    'probe_fingerprint',
    'probe_key',
    'refresh_vlm_state',
    'resolve_api_key',
    'resolve_vlm_source',
    'secrets_dir',
    'stored_endpoint',
    'vlm_configured',
]
