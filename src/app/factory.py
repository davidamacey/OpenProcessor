"""The FastAPI application factory: middleware, exception handlers and router mounting."""

import time
import uuid
from pathlib import Path

import orjson
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.app.lifespan import lifespan
from src.clients.occ import OCCFinalConflictError
from src.config import get_settings
from src.core.cors import add_lan_cors
from src.core.error_handlers import triton_unavailable_response, upstream_unavailable_response
from src.core.logging import bind_request_id, get_logger, get_request_id
from src.core.request_origin import RequestOriginMiddleware
from src.core.upload_limits import (
    UploadTooLargeError,
    enforce_file_cap,
    request_exceeds_ceiling,
    too_large_response,
    upload_too_large_handler,
)
from src.routers import (
    analyze_router,
    clusters_router,
    detect_router,
    embed_router,
    faces_router,
    health_router,
    ingest_router,
    models_router,
    ocr_router,
    persons_router,
    query_router,
    search_router,
    v1_router,
)
from src.routers.api_docs import install_api_docs
from src.routers.curation._mounting import mount_all_curation_routers
from src.utils.retry import RetryExhaustedError
from src.utils.upstream_errors import UpstreamError


logger = get_logger(__name__)


def _read_version() -> str:
    """Read version from VERSION file, falling back to settings default."""
    version_file = Path(__file__).resolve().parents[2] / 'VERSION'
    try:
        return version_file.read_text().strip()
    except FileNotFoundError:
        return '0.0.0'


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title='OpenProcessor',
        description=(
            'High-performance visual AI service providing object detection, '
            'face recognition, image embeddings, visual search, and OCR. '
            'All inference runs through NVIDIA Triton Inference Server.'
        ),
        version=_read_version(),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        dependencies=[Depends(enforce_file_cap)],
    )
    application.exception_handler(UploadTooLargeError)(upload_too_large_handler)
    install_api_docs(application)

    application.add_middleware(RequestOriginMiddleware)
    add_lan_cors(application)

    # Performance Middleware (defined first, runs second in LIFO order)
    @application.middleware('http')
    async def performance_middleware(request: Request, call_next):
        """
        Monitor request performance, validate file size, and inject timing into response.

        Industry standard: timing included in both header (X-Process-Time) and response body.
        """
        start_time = time.time()
        req_id = get_request_id()  # Get from context set by request_id_middleware

        # A body larger than any upload the server accepts is refused before it is read.
        # An exception raised in this layer never reaches the exception handlers (it would
        # answer 500), so the 413 is returned. The per-file cap is enforce_file_cap.
        if request.method == 'POST' and request_exceeds_ceiling(request):
            return too_large_response()

        response = await call_next(request)

        # Calculate timing
        duration_ms = (time.time() - start_time) * 1000

        # Add timing header (always)
        response.headers['X-Process-Time'] = f'{duration_ms:.2f}ms'

        # Inject timing into JSON response body for inference endpoints
        content_type = response.headers.get('content-type', '')
        is_inference_endpoint = any(
            path in request.url.path
            for path in [
                '/detect',
                '/faces',
                '/embed',
                '/search',
                '/ingest',
                '/analyze',
                '/ocr',
            ]
        )

        if 'application/json' in content_type and is_inference_endpoint:
            # Read response body
            body_chunks = [chunk async for chunk in response.body_iterator]
            body = b''.join(body_chunks)

            try:
                # Parse and inject timing + request ID
                data = orjson.loads(body)
                if isinstance(data, dict):
                    data['total_time_ms'] = round(duration_ms, 2)
                    data['request_id'] = req_id
                body = orjson.dumps(data)

                # Build new headers without content-length (will be recalculated)
                new_headers = {
                    k: v for k, v in response.headers.items() if k.lower() != 'content-length'
                }
                new_headers['X-Process-Time'] = f'{duration_ms:.2f}ms'
                new_headers['X-Request-ID'] = req_id

                # Create new response with modified body
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=new_headers,
                    media_type='application/json',
                )
            except Exception:
                # If parsing fails, return original response
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )

        # Log slow requests with request ID for correlation
        if duration_ms > settings.slow_request_threshold_ms:
            logger.warning(
                'slow_request',
                request_id=req_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )

        return response

    @application.exception_handler(OCCFinalConflictError)
    async def occ_final_conflict_handler(request: Request, exc: OCCFinalConflictError):
        """Map exhausted-OCC-retry conflicts to HTTP 409.

        Registered ahead of the generic ``Exception`` handler below so a
        human-write endpoint's re-raised :class:`OCCFinalConflictError`
        (see ``src.clients.occ`` call sites in ``routers/curation/``)
        surfaces as a client-actionable "someone else edited this concurrently"
        response instead of an opaque 500.
        """
        req_id = get_request_id()
        logger.warning(
            'occ_final_conflict',
            request_id=req_id,
            method=request.method,
            path=request.url.path,
            doc_id=exc.doc_id,
            retries=exc.retries,
        )
        return JSONResponse(
            status_code=409,
            content={
                'detail': 'concurrent write conflict; refresh and retry',
                'doc_id': exc.doc_id,
                'retries': exc.retries,
                'request_id': req_id,
            },
            headers={'X-Request-ID': req_id},
        )

    application.exception_handler(RetryExhaustedError)(triton_unavailable_response)
    application.exception_handler(UpstreamError)(upstream_unavailable_response)

    # Global Exception Handler - include request ID for debugging
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions with request context for debugging."""
        req_id = get_request_id()
        logger.error(
            'unhandled_exception',
            request_id=req_id,
            method=request.method,
            path=request.url.path,
            error_type=type(exc).__name__,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                'detail': 'Internal server error',
                'request_id': req_id,
                'error_type': type(exc).__name__,
            },
            headers={'X-Request-ID': req_id},
        )

    @application.middleware('http')
    async def http_duration_middleware(request: Request, call_next):
        """Record per-request latency into the Prometheus histogram."""
        from src.core.metrics import HTTP_REQUEST_DURATION_SECONDS

        started = time.monotonic()
        response = await call_next(request)
        # FastAPI populates scope['route'] once a route has matched. For
        # 404s (no match) fall back to a low-cardinality truncation of
        # the raw path (first 2 segments) so the label space isn't
        # exploded by `/v1/whatever/<random-id>` misses.
        route_obj = request.scope.get('route')
        route_template = getattr(route_obj, 'path', None)
        if not route_template:
            segments = request.url.path.strip('/').split('/')
            route_template = '/' + '/'.join(segments[:2]) if segments and segments[0] else '/'
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            route=route_template,
            status=str(response.status_code),
        ).observe(time.monotonic() - started)
        return response

    # Request ID Middleware (defined last, runs first in LIFO order)
    @application.middleware('http')
    async def request_id_middleware(request: Request, call_next):
        """
        Add correlation ID (X-Request-ID) to all requests.

        If client provides X-Request-ID header, use it. Otherwise generate a new UUID.
        The request ID is available via get_request_id() in any code path, and
        bind_request_id() also attaches it to every structlog event on this task.
        """
        # Get or generate request ID
        req_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
        bind_request_id(req_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers['X-Request-ID'] = req_id
        return response

    # Include Routers - Clean API structure without track naming
    application.include_router(health_router)  # /health - Health checks
    application.include_router(detect_router)  # /detect - Object detection
    application.include_router(faces_router)  # /faces - Face detection/recognition
    application.include_router(persons_router)  # /persons - Person management
    application.include_router(embed_router)  # /embed - CLIP embeddings
    application.include_router(search_router)  # /search - Visual similarity search
    application.include_router(ingest_router)  # /ingest - Data ingestion
    application.include_router(analyze_router)  # /analyze - Combined analysis
    application.include_router(clusters_router)  # /clusters - Clustering/albums
    application.include_router(query_router)  # /query - Data retrieval
    application.include_router(ocr_router)  # /ocr - Text extraction
    application.include_router(models_router)  # /models - Model management
    mount_all_curation_routers(application)  # /curation: global routes + /projects/{project}/...

    # Versioned API - All endpoints also available under /v1
    application.include_router(v1_router)  # /v1/* - Versioned API

    # The curation metric registry (src.services.curation.metrics) registers
    # Counter/Histogram objects on the prometheus_client default REGISTRY at
    # import time; import it here so its module-level side effects run even
    # if no curation route has been hit yet — the existing /metrics endpoint
    # (src.routers.health) picks them up automatically via generate_latest().
    import src.services.curation.metrics  # noqa: F401  # import registers the metrics on the default registry

    return application
