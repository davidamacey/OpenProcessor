"""
Health and Monitoring Router

Provides liveness/readiness probes, service info, and connection pool
statistics.

- /live   — process responsive, no dependency probing (livenessProbe)
- /ready  — Triton + OpenSearch actively probed (readinessProbe / LB drain)
- /health — backward-compat alias for /ready
"""

import asyncio
import logging
import os
from typing import Any

import httpx
import psutil
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from src.clients.triton_pool import get_async_triton_client, get_client_pool_stats
from src.config import get_settings
from src.core.metrics import render_metrics


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=['Health & Monitoring'],
)


# Readiness-probe timeout. Kept short so a degraded dep does not stall
# the probe past a typical k8s readiness-check window.
_READY_PROBE_TIMEOUT_S = 2.0


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


async def _probe_triton() -> tuple[bool, str]:
    """Probe Triton via a real ``is_server_live()`` gRPC round-trip.

    Inspecting ``get_client_pool_stats().active_connections`` is not a
    valid readiness signal: it stays 0 until the first real infer call
    lands on the pool. Combined with ``depends_on: service_healthy`` on
    dependent containers, that produces a startup deadlock — dependents
    wait for yolo-api to report healthy, but nothing is calling Triton
    through yolo-api to wake the pool, so the count never moves.

    ``is_server_live()`` actively probes Triton via the async client
    (created lazily on first call). The side effect is exactly what we
    want for a readiness check: it both tests reachability AND warms the
    connection so subsequent infer requests skip the
    connection-establishment cost. Bounded by ``_READY_PROBE_TIMEOUT_S``
    so a degraded Triton can't stall the probe past the k8s readiness
    window.
    """
    settings = get_settings()
    try:
        client = await asyncio.wait_for(
            get_async_triton_client(settings.triton_url),
            timeout=_READY_PROBE_TIMEOUT_S,
        )
        live = await asyncio.wait_for(
            client.is_server_live(),
            timeout=_READY_PROBE_TIMEOUT_S,
        )
    except TimeoutError:
        return False, f'is_server_live timeout after {_READY_PROBE_TIMEOUT_S}s'
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'
    if not live:
        return False, 'is_server_live returned False'
    return True, 'is_server_live OK'


async def _readiness_payload() -> tuple[int, dict[str, Any]]:
    """Probe Triton and OpenSearch concurrently; build the payload.

    Returns the HTTP status code (200 if every dep is ready, 503 otherwise)
    and a JSON-serializable body listing per-service status + detail.
    """
    settings = get_settings()
    triton_task = asyncio.create_task(_probe_triton())
    opensearch_task = asyncio.create_task(_probe_http(settings.opensearch_url))

    triton_ok, triton_detail = await triton_task
    opensearch_ok, opensearch_detail = await opensearch_task

    services = {
        'triton': {'ok': triton_ok, 'detail': triton_detail, 'url': settings.triton_url},
        'opensearch': {
            'ok': opensearch_ok,
            'detail': opensearch_detail,
            'url': settings.opensearch_url,
        },
    }
    all_ok = triton_ok and opensearch_ok
    payload: dict[str, Any] = {
        'status': 'ready' if all_ok else 'not_ready',
        'version': settings.api_version,
        'api_version': 'v1',
        'services': services,
    }
    return (200 if all_ok else 503), payload


@router.get('/')
def root():
    """
    Service information endpoint.

    Returns available endpoints, models, and backend configuration.
    """
    settings = get_settings()

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
            'faces': '/faces (or /v1/faces)',
            'embed': '/embed (or /v1/embed)',
            'search': '/search (or /v1/search)',
            'ingest': '/ingest (or /v1/ingest)',
            'analyze': '/analyze (or /v1/analyze)',
            'ocr': '/ocr (or /v1/ocr)',
            'clusters': '/clusters (or /v1/clusters)',
            'query': '/query (or /v1/query)',
            'persons': '/persons (or /v1/persons)',
        },
        'models': {
            'detection': settings.models.YOLO_MODEL,
            'face_detection': settings.models.FACE_DETECT_MODEL,
            'face_embedding': settings.models.ARCFACE_MODEL,
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
    """Readiness probe — Triton + OpenSearch both reachable.

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
        payload['resources']['gpu'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_allocated_mb': round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            'memory_reserved_mb': round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
        }
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


@router.get('/metrics')
def metrics() -> Response:
    """Prometheus exposition endpoint (scraped by the monitoring stack)."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get('/connection_pool_info')
def connection_pool_info():
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
