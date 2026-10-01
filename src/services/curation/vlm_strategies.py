"""The ``vlm`` axis of ``GET /methods`` (W9.3, §7.8.2): one entry per
registered VLM endpoint plus ``off``, each carrying the external-images
facts so neither the settings dropdown nor the run dialog infers them.

Split out of ``strategy_registry`` (700-LOC ratchet) and kept here, next to
it, because it is registry logic: the served flags and the rule a run
enforces come from the SAME function
(:func:`~src.services.labeling.vlm_endpoints.external_ack_state`), so they
cannot drift.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger
from src.services.labeling.vlm_endpoints import (
    STATUS_LABELS,
    VlmEndpointUnavailableError,
    active_vlm_endpoint,
    available_vlm_endpoints,
    external_ack_state,
    refresh_vlm_state,
)
from src.services.labeling.vlm_url_policy import acompute_locality


logger = get_logger(__name__)

#: ``endpoint_status`` -> ``MethodInfo.status`` (whose Literal is fixed).
_METHOD_STATUS = {
    'ready': 'stable',
    'unprobed': 'experimental',
    'probe_failed': 'disabled',
    'unreachable': 'disabled',
}


def vlm_method_status(endpoint_status: str) -> str:
    """Map an endpoint's health onto the ``status`` Literal
    (``stable | experimental | shadow | disabled``)."""
    return _METHOD_STATUS[endpoint_status]


def active_default_id() -> str | None:
    """The bound project's active endpoint name, ``'off'`` when explicitly
    off, ``None`` when there is nothing to choose."""
    from src.services.config_store import get_config_store

    try:
        endpoint = active_vlm_endpoint()
    except VlmEndpointUnavailableError:
        return None
    if endpoint is not None:
        return endpoint.name
    return 'off' if get_config_store().current.active_vlm == 'off' else None


def advertised_ids() -> frozenset[str]:
    names = {e.name for e in available_vlm_endpoints()}
    return frozenset({*names, 'off'}) if names else frozenset()


async def vlm_strategies(opensearch: Any | None) -> list[dict[str, Any]]:
    """The ``vlm`` axis entries. Refreshes the registry and the bound
    project's activation first (they are per-process snapshots) when a
    client is available."""
    from src.services.config_store.vlm_gate import project_vlm_state

    if opensearch is not None:
        try:
            await refresh_vlm_state(opensearch)
        except Exception as exc:
            logger.warning('curation_methods_vlm_refresh_failed', error=str(exc))
    endpoints = available_vlm_endpoints()
    if not endpoints:
        return []
    default_id = active_default_id()
    state = project_vlm_state()
    entries: list[dict[str, Any]] = []
    for endpoint in endpoints:
        try:
            locality = await acompute_locality(endpoint.body.base_url, cached=True)
        except ValueError:
            locality = 'unknown'
        ack = external_ack_state(
            endpoint,
            locality,
            active_ref=state.active_ref,
            active_ack_at=state.active_ack_at,
            acked_refs=state.acked_refs,
        )
        entries.append(
            {
                'id': endpoint.name,
                'axis': 'vlm',
                'label': endpoint.name,
                'status': vlm_method_status(endpoint.status),
                'default': endpoint.name == default_id,
                'endpoint_status': endpoint.status,
                'endpoint_status_label': STATUS_LABELS[endpoint.status],
                'sends_images_externally': ack.sends_images_externally,
                'warning': ack.warning,
                'default_ack_recorded': ack.default_ack_recorded,
                'per_run_ack_required': ack.per_run_ack_required,
            }
        )
    entries.append(
        {
            'id': 'off',
            'axis': 'vlm',
            'label': 'off',
            'status': 'stable',
            'default': default_id == 'off',
        }
    )
    return entries


__all__ = ['active_default_id', 'advertised_ids', 'vlm_method_status', 'vlm_strategies']
