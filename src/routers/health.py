"""
Health and Monitoring Router

Provides liveness/readiness probes, service info, and connection pool
statistics.

- /live   — process responsive, no dependency probing (livenessProbe)
- /ready  — the database, the task queue, Triton, its required models, OpenSearch
            and free disk space actively probed, one after another (readinessProbe / LB drain)
- /health — backward-compat alias for /ready
"""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

import httpx
import psutil
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from opensearchpy import OpenSearchException

from src.app.startup_gate import startup_probes
from src.clients.triton_client_manager import get_async_triton_client, get_client_pool_stats
from src.config import get_settings
from src.config.faces import faces_enabled
from src.core import disk_space
from src.core.metrics import render_metrics
from src.db.schema_check import ProbeFailed
from src.services.face_privacy import FaceDataCounts, stored_face_counts, stored_summary
from src.services.model_readiness import models_not_ready, required_model_names
from src.services.triton_control import TritonControlService


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Health & Monitoring'],
)


# Readiness-probe timeout. Kept short so a degraded dep does not stall
# the probe past a typical k8s readiness-check window.
_READY_PROBE_TIMEOUT_S = 2.0
_FACES_OFF = 'disabled (OP_FACES_ENABLED=false)'
_FACE_STATE_ERRORS = (TimeoutError, OSError, RuntimeError, OpenSearchException)


async def _probe_http(url: str) -> tuple[bool, str]:
    """Issue a single GET against ``url`` and report (ok, detail).

    Treated as ready on any 2xx/3xx response. Network / timeout / 5xx all
    flip the service to not-ready; the detail string carries the reason
    so /ready consumers can see *which* dep is down without grepping logs.
    """
    try:
        async with httpx.AsyncClient(timeout=_READY_PROBE_TIMEOUT_S) as client:
            response = await client.get(url)
    except httpx.TimeoutException:
        return False, f'timeout after {_READY_PROBE_TIMEOUT_S}s'
    except httpx.HTTPError as exc:
        return False, f'{type(exc).__name__}: {exc}'
    if response.status_code >= 500:
        return False, f'HTTP {response.status_code}'
    return True, f'HTTP {response.status_code}'


async def _bounded_probe(name: str, check: Callable[[], Awaitable[str]]) -> tuple[bool, str]:
    """The one readiness boundary: run ``check`` and turn any failure into a not-ready entry.

    ``ProbeFailed`` and a timeout carry their own detail. Any other exception means the probe
    could not run (an unreadable secret, bad settings, a client error): the dependency is down
    for readiness, so ``/ready`` answers 503 naming it, never 500. The body gets only the
    exception class (the message may hold a path); the log line gets the message.
    """
    try:
        return True, await check()
    except ProbeFailed as exc:
        return False, str(exc)
    except TimeoutError:
        return False, f'timeout after {_READY_PROBE_TIMEOUT_S}s'
    except Exception as exc:
        logger.warning('readiness probe %s could not run: %s', name, exc)
        return False, type(exc).__name__


async def _probe_dependency(name: str) -> tuple[bool, str]:
    """Run the startup probe of ``name`` (database, queue) off the event loop, bounded.

    The same probes the startup wait uses, so readiness and start agree on what "up" means.
    """

    async def check() -> str:
        probe = startup_probes(get_settings())[name]
        await asyncio.wait_for(asyncio.to_thread(probe), timeout=_READY_PROBE_TIMEOUT_S)
        return 'ok'

    return await _bounded_probe(name, check)


async def _probe_triton() -> tuple[bool, str]:
    """Probe Triton via a real ``is_server_live()`` gRPC round-trip.

    Inspecting ``get_client_pool_stats().active_connections`` is not a
    valid readiness signal: it stays 0 until the first real infer call
    lands on the pool. Combined with ``depends_on: service_healthy`` on
    dependent containers, that produces a startup deadlock — dependents
    wait for api to report healthy, but nothing is calling Triton
    through api to wake the pool, so the count never moves.

    ``is_server_live()`` actively probes Triton via the async client
    (created lazily on first call). The side effect is exactly what we
    want for a readiness check: it both tests reachability AND warms the
    connection so subsequent infer requests skip the
    connection-establishment cost. Bounded by ``_READY_PROBE_TIMEOUT_S``
    so a degraded Triton can't stall the probe past the k8s readiness
    window.
    """

    async def check() -> str:
        client = await asyncio.wait_for(
            get_async_triton_client(get_settings().triton_url),
            timeout=_READY_PROBE_TIMEOUT_S,
        )
        if not await asyncio.wait_for(client.is_server_live(), timeout=_READY_PROBE_TIMEOUT_S):
            raise ProbeFailed('is_server_live returned False')
        return 'is_server_live OK'

    return await _bounded_probe('triton', check)


async def _probe_required_models() -> tuple[bool, str, list[dict[str, str]]]:
    """Check that every required model is READY in Triton's repository index.

    Triton runs with ``--exit-on-error=false``, so a model that failed to load
    leaves the server live; only the index shows it. An unreadable index counts
    as not ready (fail closed).
    """
    required = required_model_names()
    try:
        index = await asyncio.wait_for(
            TritonControlService().get_repository_index(),
            timeout=_READY_PROBE_TIMEOUT_S,
        )
    except TimeoutError:
        reason = f'repository index timeout after {_READY_PROBE_TIMEOUT_S}s'
        return False, reason, [{'name': n, 'state': 'UNKNOWN', 'reason': reason} for n in required]
    bad = models_not_ready(index, required)
    if bad:
        names = ', '.join(f'{b["name"]} ({b["state"]})' for b in bad)
        return False, f'required models not READY: {names}', bad
    return True, f'{len(required)} required models READY', []


async def _opensearch_free_ratio(url: str) -> float | None:
    """The smallest free fraction over the OpenSearch nodes (``_cat/allocation``), or ``None``
    when it cannot be read: the opensearch entry of the report already says it is down."""
    try:
        async with httpx.AsyncClient(timeout=_READY_PROBE_TIMEOUT_S) as client:
            response = await client.get(
                f'{url.rstrip("/")}/_cat/allocation', params={'format': 'json', 'bytes': 'b'}
            )
        response.raise_for_status()
        rows = response.json()
        ratios = [
            int(row['disk.avail']) / int(row['disk.total'])
            for row in rows
            if str(row.get('disk.total', '')).isdigit() and str(row.get('disk.avail', '')).isdigit()
        ]
    except (httpx.HTTPError, ValueError, TypeError, KeyError, ZeroDivisionError):
        return None
    return min(ratios) if ratios else None


async def _disk_entry(opensearch_url: str, *, opensearch_up: bool) -> dict[str, Any]:
    """The ``services.disk`` report: ``ok``, ``low`` (a warning; still ready) or ``critical``
    (not ready). The database volume lives in the Postgres container, which this process
    cannot see; its own healthcheck and the Postgres exporter cover it."""
    volumes = [disk_space.data_volume()]
    if opensearch_up:
        ratio = await _opensearch_free_ratio(opensearch_url)
        if ratio is not None:
            volumes.append(disk_space.measure('opensearch', ratio))
    level = disk_space.worst(volumes)
    return {
        'ok': level != 'critical',
        'status': level,
        'volumes': {
            v.volume: {'free_ratio': round(v.free_ratio, 4), 'level': v.level} for v in volumes
        },
    }


async def _face_data_counts() -> FaceDataCounts:
    return await asyncio.wait_for(stored_face_counts(), timeout=_READY_PROBE_TIMEOUT_S)


async def _faces_entry() -> dict[str, Any]:
    """The ``services.faces`` report. Informational: it never changes readiness, because
    stored face data is refused by every route and 503 would take the API down for it."""
    if faces_enabled():
        return {'enabled': True}
    try:
        counts = await _face_data_counts()
    except _FACE_STATE_ERRORS as exc:
        return {
            'enabled': False,
            'stored': None,
            'detail': f'face data state unknown: {type(exc).__name__}',
        }
    stored = asdict(counts)
    if counts.is_empty():
        return {'enabled': False, 'stored': stored, 'detail': 'faces are off; no face data stored'}
    return {'enabled': False, 'stored': stored, 'detail': stored_summary(counts)}


async def _readiness_payload() -> tuple[int, dict[str, Any]]:
    """Probe every dependency in turn (no fan-out); build the payload.

    Returns the HTTP status code (200 if every dep and required model is
    ready, 503 otherwise)
    and a JSON-serializable body listing per-service status + detail.
    """
    settings = get_settings()
    database_ok, database_detail = await _probe_dependency('database')
    queue_ok, queue_detail = await _probe_dependency('queue')
    triton_ok, triton_detail = await _probe_triton()
    opensearch_ok, opensearch_detail = await _probe_http(settings.opensearch_url)

    services: dict[str, Any] = {
        'database': {'ok': database_ok, 'detail': database_detail},
        'queue': {'ok': queue_ok, 'detail': queue_detail},
        'triton': {'ok': triton_ok, 'detail': triton_detail, 'url': settings.triton_url},
        'opensearch': {
            'ok': opensearch_ok,
            'detail': opensearch_detail,
            'url': settings.opensearch_url,
        },
    }
    all_ok = database_ok and queue_ok and triton_ok and opensearch_ok
    if triton_ok:
        # Without a reachable Triton the models cannot be judged; the triton
        # entry already fails the probe.
        models_ok, models_detail, not_ready = await _probe_required_models()
        services['models'] = {'ok': models_ok, 'detail': models_detail, 'not_ready': not_ready}
        all_ok = all_ok and models_ok
    services['faces'] = await _faces_entry()
    services['disk'] = await _disk_entry(settings.opensearch_url, opensearch_up=opensearch_ok)
    all_ok = all_ok and services['disk']['ok']
    payload: dict[str, Any] = {
        'status': 'ready' if all_ok else 'not_ready',
        'version': settings.api_version,
        'api_version': 'v1',
        'services': services,
    }
    return (200 if all_ok else 503), payload


@router.get('/', response_model=None)
def root() -> dict[str, Any]:
    """
    Service information endpoint.

    Returns available endpoints, models, and backend configuration.
    """
    settings = get_settings()
    faces_on = faces_enabled()

    return {
        'service': 'Visual AI API',
        'version': settings.api_version,
        'api_versions': {
            'current': 'v1',
            'supported': ['v1'],
            'deprecated': [],
        },
        'status': 'running',
        'endpoints': {
            'detection': '/detect (or /v1/detect)',
            'faces': '/faces (or /v1/faces)' if faces_on else _FACES_OFF,
            'embed': '/embed (or /v1/embed)',
            'search': '/search (or /v1/search)',
            'ingest': '/ingest (or /v1/ingest)',
            'analyze': '/analyze (or /v1/analyze)',
            'ocr': '/ocr (or /v1/ocr)',
            'clusters': '/clusters (or /v1/clusters)',
            'query': '/query (or /v1/query)',
            'persons': '/persons (or /v1/persons)' if faces_on else _FACES_OFF,
        },
        'models': {
            'detection': settings.models.YOLO_MODEL,
            'face_detection': settings.models.FACE_DETECT_MODEL if faces_on else None,
            'face_embedding': settings.models.ARCFACE_MODEL if faces_on else None,
            'clip_image': settings.models.CLIP_IMAGE_MODEL,
            'clip_text': settings.models.CLIP_TEXT_MODEL,
            'ocr_detection': settings.models.OCR_DET_MODEL,
            'ocr_recognition': settings.models.OCR_REC_MODEL,
        },
        'backend': {
            'triton_url': f'grpc://{settings.triton_url}',
            'gpu_available': torch.cuda.is_available(),
        },
    }


@router.get('/live')
def live() -> dict[str, str]:
    """Liveness probe — process is responsive, no dependency probing.

    Designed for k8s ``livenessProbe`` and the container HEALTHCHECK: a
    true response here means the event loop is healthy and the process
    should NOT be restarted. A degraded dependency (Triton / OpenSearch)
    must NOT flap liveness; use /ready for that.
    """
    return {'status': 'alive'}


@router.get('/ready')
async def ready() -> JSONResponse:
    """Readiness probe — the database, the queue, Triton, its required models and OpenSearch.

    Returns 200 with ``status='ready'`` when every dep responds; 503
    with per-service detail when at least one is unreachable. Designed
    for k8s ``readinessProbe`` and for an upstream LB to drain traffic
    from this instance during dependency degradation without killing
    the process.
    """
    status_code, payload = await _readiness_payload()

    # Best-effort process / GPU resource snapshot — informational only;
    # never gates readiness. Kept off /live to keep liveness probes
    # branch-free.
    settings = get_settings()
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    payload['resources'] = {
        'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
        'cpu_percent': process.cpu_percent(),
    }
    if torch.cuda.is_available():
        # What THIS API process holds in its own CUDA context (0 for an API
        # that runs inference on Triton), not the GPU's memory use: for the
        # GPU, read Triton or DCGM metrics.
        payload['resources']['process_cuda_allocated_mb'] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 2
        )
        payload['resources']['process_cuda_reserved_mb'] = round(
            torch.cuda.memory_reserved() / 1024 / 1024, 2
        )
    payload.setdefault('version', settings.api_version)
    return JSONResponse(status_code=status_code, content=payload)


@router.get('/health')
async def health() -> JSONResponse:
    """Backward-compat alias for :func:`ready`.

    Existing dashboards, scripts, and operator runbooks call /health and
    expect the dep-probing behavior. New callers should target /live
    (liveness) or /ready (readiness) explicitly.
    """
    return await ready()


@router.get('/health/pe_text')
def pe_text_health(request: Request) -> JSONResponse:
    """PE-Core text encoder status (backs ``GET /curation/search/text``).

    200 with the active backend (``onnx`` / ``triton`` / ``torch``) once
    warmed; 503 while cold or when no backend could load. Local to this
    process — does not probe Triton.
    """
    encoder = getattr(request.app.state, 'pe_encoder', None)
    if encoder is None:
        return JSONResponse(
            status_code=503, content={'ready': False, 'detail': 'PE encoder not constructed'}
        )
    status = encoder.text_status()
    return JSONResponse(status_code=200 if status['ready'] else 503, content=status)


@router.get('/metrics')
def metrics() -> Response:
    """Prometheus exposition endpoint (scraped by the monitoring stack)."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get('/connection_pool_info', response_model=None)
def connection_pool_info() -> dict[str, Any]:
    """
    Triton gRPC connection pool statistics.

    Useful for monitoring shared client usage and A/B testing.
    """
    stats = get_client_pool_stats()

    return {
        'shared_client_pool': {
            'active': stats['active_connections'] > 0,
            'connection_count': stats['active_connections'],
            'triton_urls': stats['urls'],
        },
        'usage_info': {
            'shared_client_enabled': 'Use ?shared_client=true (default)',
            'per_request_client': 'Use ?shared_client=false for testing',
            'performance_impact': {
                'shared_mode': 'Enables batching, 400-600 RPS (recommended)',
                'per_request_mode': 'No batching, 50-100 RPS (testing only)',
            },
        },
        'testing': {
            'example_shared': "curl 'http://localhost:9600/predict/small?shared_client=true' -F 'image=@test.jpg'",
            'example_per_request': "curl 'http://localhost:9600/predict/small?shared_client=false' -F 'image=@test.jpg'",
            'check_batching': "docker compose logs triton-server | grep 'batch size'",
        },
    }
