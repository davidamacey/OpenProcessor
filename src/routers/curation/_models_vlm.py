"""The VLM rows of ``GET /models/status`` (W9.1 M8): one row per registered
endpoint, the bound project's active one first. Split out of ``models.py``
(700-LOC ratchet)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.config.project_context import current_project
from src.core.logging import get_logger
from src.services.labeling.vlm_endpoints import (
    VlmEndpoint,
    VlmEndpointUnavailableError,
    active_vlm_endpoint,
    available_vlm_endpoints,
    refresh_vlm_state,
)


logger = get_logger(__name__)

#: Probe status -> the ``status`` a models card renders.
_STATUS = {
    'ready': 'ready',
    'unprobed': 'unknown',
    'probe_failed': 'unavailable',
    'unreachable': 'unavailable',
}


def strip_userinfo(url: str) -> str:
    """``url`` without any ``user:password@`` (an endpoint URL is served on
    a public listing and must never carry credentials)."""
    parts = urlsplit(url)
    host = parts.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    netloc = f'{host}:{parts.port}' if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


async def _refresh() -> None:
    from src.services.projects.guard import make_curation_opensearch

    try:
        await refresh_vlm_state(await make_curation_opensearch())
    except Exception as exc:
        logger.warning('models_status_vlm_refresh_failed', error=str(exc))


async def _live_health(endpoint: VlmEndpoint) -> tuple[str, str | None]:
    from src.services.labeling.vlm_factory import labeler_for
    from src.services.labeling.vlm_prompts import active_prompt_pack

    try:
        health = await labeler_for(endpoint, active_prompt_pack()).health()
    except Exception as exc:
        return 'unavailable', str(exc)
    return ('ready' if health.reachable else 'unavailable'), health.last_error


def _row(
    endpoint: VlmEndpoint,
    *,
    active: bool,
    active_in: list[str],
    status: str,
    last_error: str | None,
) -> dict[str, Any]:
    probe = endpoint.last_probe
    if last_error is None and probe is not None and not probe.ok:
        last_error = ', '.join(probe.error_codes) or None
    return {
        'name': endpoint.name,
        'model': endpoint.model_id,
        'friendly_name': f'VLM ({endpoint.name})',
        'role': 'Open-vocabulary labeling and region verification',
        'kind': 'vlm',
        'model_type': 'Vision-Language Model',
        'status': status,
        'version': None,
        'inference_count': None,
        'exec_count': None,
        'inference_failed': None,
        'avg_latency_ms': None,
        'last_error': last_error,
        'endpoint': strip_userinfo(endpoint.body.base_url),
        'unloadable': False,
        'optional': False,
        'active': active,
        'active_in': active_in,
    }


async def vlm_status_rows() -> list[dict[str, Any]]:
    """One row per endpoint (active first). Only the active endpoint gets a
    live health call; the rest report their last probe."""
    await _refresh()
    try:
        active = active_vlm_endpoint()
    except VlmEndpointUnavailableError as exc:
        logger.warning('models_status_vlm_active_unresolved', error=str(exc))
        active = None
    # A project-scoped listing names only the bound project: which OTHER
    # projects run an endpoint is deployment-wide and lives on the global
    # GET /vlm/endpoints.
    slug = current_project().record.slug
    rows: list[dict[str, Any]] = []
    for endpoint in available_vlm_endpoints():
        is_active = active is not None and endpoint.name == active.name
        if is_active:
            status, error = await _live_health(active)  # type: ignore[arg-type]
        else:
            status, error = _STATUS[endpoint.status], None
        rows.append(
            _row(
                endpoint,
                active=is_active,
                active_in=[slug] if is_active else [],
                status=status,
                last_error=error,
            )
        )
    rows.sort(key=lambda r: not r['active'])
    return rows


def external_service_names() -> set[str]:
    """Every registered endpoint's name AND model id: none of them is a
    Triton model, so ``DELETE /models/{name}`` must never reach Triton."""
    names: set[str] = set()
    for endpoint in available_vlm_endpoints():
        names.add(endpoint.name)
        names.add(endpoint.body.model)
        names.add(endpoint.model_id)
    return names


__all__ = ['external_service_names', 'strip_userinfo', 'vlm_status_rows']
