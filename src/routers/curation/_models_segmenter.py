"""Segmenter roster-entry probe for `GET /curation/models/status`.

Split out of ``models.py`` to stay under the repo's 700-LOC pre-commit
ratchet. The segmenter (``DetectionProfile.segmenter_name``, e.g.
``sam3``) is its own HTTP service at ``OP_SEGMENTER_URL`` — never a
Triton model. It has its own ``GET /health`` (``{"status": "healthy",
"model": "sam3", "loaded": true, "instances": 2}``), so it must be
probed like the VLM entry (``kind: 'external'``), not run through
Triton's ``/v2/repository/index`` the way every other roster entry is —
that always reported ``not_ready`` for a healthy segmenter, since Triton
never lists it at all.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


_SEGMENTER_HEALTH_TIMEOUT_S = 3.0


async def _segmenter_health(url: str) -> tuple[str, str | None]:
    """Probe the segmenter's own ``GET /health``.

    Returns ``(status, last_error)``. ``status`` is ``'ready'`` when the
    service answers 200 and reports ``loaded: true``, otherwise
    ``'unavailable'`` with ``last_error`` explaining why. A short,
    dedicated timeout keeps a dead/hung segmenter from blocking the
    whole ``/models/status`` response.
    """
    try:
        async with httpx.AsyncClient(timeout=_SEGMENTER_HEALTH_TIMEOUT_S) as client:
            r = await client.get(f'{url}/health')
        if r.status_code != 200:
            return 'unavailable', f'segmenter /health returned HTTP {r.status_code}'
        body = r.json()
        if not body.get('loaded'):
            return (
                'unavailable',
                f'segmenter reachable but not loaded (status={body.get("status")!r})',
            )
        return 'ready', None
    except (httpx.HTTPError, ValueError) as exc:
        return 'unavailable', str(exc)


async def build_segmenter_entry(
    name: str,
    friendly: str,
    role: str,
    mtype: str,
) -> dict[str, Any]:
    """Segmenter roster entry — an external HTTP service, never a Triton model.

    Mirrors the VLM entry's shape (``kind: 'external'``, ``unloadable:
    False``) rather than ``_build_triton_entry``'s.
    """
    url = os.environ.get('OP_SEGMENTER_URL', '').strip()
    if not url:
        return {
            'name': name,
            'friendly_name': friendly,
            'role': role,
            'kind': 'external',
            'model_type': mtype,
            'status': 'not_configured',
            'version': None,
            'inference_count': None,
            'exec_count': None,
            'inference_failed': None,
            'avg_latency_ms': None,
            'last_error': 'OP_SEGMENTER_URL is not configured',
            'endpoint': '',
            'unloadable': False,
        }
    # OP_SEGMENTER_URLS may load-balance across several hosts; the status
    # probe only needs one to characterize reachability, so use the first.
    first_url = url.split(',')[0].strip().rstrip('/')
    status, last_error = await _segmenter_health(first_url)
    return {
        'name': name,
        'friendly_name': friendly,
        'role': role,
        'kind': 'external',
        'model_type': mtype,
        'status': status,
        'version': None,
        'inference_count': None,
        'exec_count': None,
        'inference_failed': None,
        'avg_latency_ms': None,
        'last_error': last_error,
        'endpoint': first_url,
        'unloadable': False,
    }
