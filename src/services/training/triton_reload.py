"""Re-load promoted models into Triton after a Triton restart.

Split out of :mod:`src.services.training.triton_promote`.
"""

from __future__ import annotations

from typing import Any

import httpx

from src.core.logging import get_logger
from src.services.training.promote_errors import TritonLoadError
from src.services.training.triton_promote import TritonPromoter
from src.services.training.triton_repo import UNLOADED_MARKER


logger = get_logger(__name__)


async def reload_promoted_models(
    promoter: TritonPromoter | None = None, *, honor_unloaded: bool = True
) -> dict[str, Any]:
    """Re-``/load`` every promoted model Triton doesn't report READY.

    Triton in explicit-control mode only loads its ``--load-model`` list
    at startup. A model promoted through this module stays on disk (its
    ``promote.json`` marker is how :func:`src.routers.curation.models.
    _discover_promoted_models` finds it) but drops to UNAVAILABLE after
    any Triton restart until someone POSTs ``/load`` again. Call this
    once at API startup (see ``src/main.py``'s lifespan) so a Triton
    restart doesn't silently strand every previously-promoted model.

    A model an operator unloaded on purpose (``UNLOADED_MARKER``) stays
    unloaded unless ``honor_unloaded`` is false (the explicit
    ``POST /train/reload_promoted``, which loads it and clears the marker).

    Best-effort throughout: a scan failure, an unreachable Triton, or a
    single model's load failure is logged and folded into the return
    value rather than raised — this must never block API startup.
    """
    p = promoter or TritonPromoter()
    try:
        entries = sorted(p.triton_models_dir.iterdir())
    except OSError as exc:
        logger.warning('reload_promoted_models_scan_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc), 'reloaded': [], 'failed': []}

    promoted_names = [
        e.name
        for e in entries
        if e.is_dir()
        and (e / 'promote.json').is_file()
        and not (honor_unloaded and (e / UNLOADED_MARKER).exists())
    ]
    if not promoted_names:
        return {'status': 'ok', 'reloaded': [], 'failed': []}

    ready_names: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f'{p.triton_http_url}/v2/repository/index')
        if resp.status_code == 200:
            ready_names = {entry['name'] for entry in resp.json() if entry.get('state') == 'READY'}
        else:
            logger.warning('reload_promoted_models_index_failed', status=resp.status_code)
    except httpx.HTTPError as exc:
        logger.warning('reload_promoted_models_index_unreachable', error=str(exc))
        return {'status': 'error', 'error': str(exc), 'reloaded': [], 'failed': []}

    reloaded: list[str] = []
    failed: list[str] = []
    for name in promoted_names:
        if name in ready_names:
            continue
        try:
            ok = await p._trigger_load(name)
        except TritonLoadError as exc:
            logger.warning('reload_promoted_model_failed', name=name, error=str(exc))
            ok = False
        (reloaded if ok else failed).append(name)

    if reloaded or failed:
        logger.info('reload_promoted_models_done', reloaded=reloaded, failed=failed)
    return {'status': 'ok', 'reloaded': reloaded, 'failed': failed}


async def reload_promoted_models_best_effort(*, log_event: str) -> None:
    """:func:`reload_promoted_models`, but never raises and logs for you.

    Shared by ``src.main``'s lifespan startup, its periodic GPU-arbiter
    reconcile tick, and (indirectly, via the plain
    :func:`reload_promoted_models` call) ``POST
    {api_prefix}/train/reload_promoted`` -- a scan failure or unreachable
    Triton is logged and swallowed so it can run unattended on both
    startup and every reconcile tick without ever taking the loop down.
    """
    try:
        reload_result = await reload_promoted_models()
        if reload_result.get('reloaded') or reload_result.get('failed'):
            logger.info(
                log_event,
                reloaded=reload_result.get('reloaded'),
                failed=reload_result.get('failed'),
            )
    except Exception as exc:
        logger.warning('promoted_models_reload_skipped', error=str(exc))
