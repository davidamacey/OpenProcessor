"""
Unified Visual AI API Service.

A high-performance FastAPI service providing comprehensive visual AI capabilities:
- Object detection (YOLO11)
- Face detection and recognition (SCRFD + ArcFace)
- Image and text embeddings (MobileCLIP)
- Visual similarity search (OpenSearch k-NN)
- Data ingestion with duplicate detection
- OCR text extraction (PP-OCRv5)
- Clustering and album organization (FAISS IVF)

All inference runs through NVIDIA Triton Inference Server for optimal GPU utilization.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response

from src.clients.triton_pool import AsyncTritonPool
from src.config import get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory
from src.core.logging import (
    bind_request_id,
    configure_logging,
    get_logger,
    get_request_id,
    request_id_ctx,
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
from src.routers.curation import router as curation_router
from src.routers.curation_images import (
    crops_router as curation_crops_router,
    router as curation_images_router,
)
from src.routers.curation_train import router as curation_train_router
from src.routers.curation_umap import router as curation_umap_router


# Request correlation IDs (request_id_ctx / get_request_id) live in
# src.core.logging so service-layer modules and out-of-process workers can
# import them without pulling in the FastAPI app. Re-exported here for
# backward compatibility.
__all__ = ['get_request_id', 'request_id_ctx']


# =============================================================================
# Shared Resources (managed by lifespan)
# =============================================================================


class AppResources:
    """Container for shared application resources."""

    shared_executor: ThreadPoolExecutor | None = None
    async_triton_pool: AsyncTritonPool | None = None


def get_shared_executor() -> ThreadPoolExecutor:
    """
    Get the shared ThreadPoolExecutor for CPU-bound tasks.

    Use this for parallel preprocessing (JPEG decode, resize, etc.)
    instead of creating per-request executors.

    Returns:
        ThreadPoolExecutor: Shared executor instance

    Raises:
        RuntimeError: If called before lifespan initialization
    """
    if AppResources.shared_executor is None:
        raise RuntimeError('Shared executor not initialized. Call during lifespan.')
    return AppResources.shared_executor


def get_async_triton_pool() -> AsyncTritonPool:
    """
    Get the high-throughput async Triton connection pool.

    Use this for batch ingestion and high-concurrency operations.
    Features:
    - 4 gRPC channels with round-robin selection
    - Semaphore-based backpressure (max 64 concurrent)
    - Statistics tracking

    Returns:
        AsyncTritonPool: Shared pool instance

    Raises:
        RuntimeError: If called before lifespan initialization
    """
    if AppResources.async_triton_pool is None:
        raise RuntimeError('AsyncTritonPool not initialized. Call during lifespan.')
    return AppResources.async_triton_pool


# Initialize structured logging
settings = get_settings()
configure_logging(json_logs=settings.json_logs, log_level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Args:
        app: FastAPI application instance (required by API contract, not used in implementation)

    Startup:
    - Create shared ThreadPoolExecutor for CPU-bound tasks
    - Create AsyncTritonPool for high-throughput inference
    - Shared Triton gRPC client auto-created on first use

    Shutdown:
    - Close AsyncTritonPool
    - Shutdown shared ThreadPoolExecutor
    - Close all Triton gRPC connections
    - Close OpenSearch connections
    """
    settings = get_settings()

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info('startup_begin', phase='initialization')

    # Create shared ThreadPoolExecutor for CPU-bound tasks
    # (JPEG decode, resize, preprocessing)
    AppResources.shared_executor = ThreadPoolExecutor(
        max_workers=64,
        thread_name_prefix='ingest-worker-',
    )
    logger.info('executor_initialized', workers=64, type='ThreadPoolExecutor')

    # Create high-throughput async Triton connection pool
    # 4 gRPC channels with different user-agents = separate TCP connections
    AppResources.async_triton_pool = AsyncTritonPool(
        url=settings.triton_url,
        pool_size=4,
        max_concurrent=64,
        verbose=False,
    )
    await AppResources.async_triton_pool.initialize()
    logger.info('triton_pool_initialized', channels=4, max_concurrent=64)

    # Best-effort: pre-create curation indexes. Wrapped so a missing /
    # not-yet-up OpenSearch instance doesn't block startup; the curation
    # router retries the create on first /curation/* request.
    try:
        from src.clients.curation_opensearch import create_curation_indexes

        os_client = await OpenSearchClientFactory.get_client()
        await create_curation_indexes(os_client.client, force_recreate=False)
        logger.info('curation_indexes_bootstrapped')
    except Exception as exc:
        logger.warning('curation_indexes_bootstrap_skipped', error=str(exc))

    # Best-effort: warm the PE-Core text encoder for GET /curation/search/text.
    # Non-fatal if torch/perception_models isn't installed or the checkpoint
    # isn't available — the search endpoint surfaces a 503 in that case
    # rather than the whole service failing to start.
    from src.clients.pe_encoder import PEEncoder

    app.state.pe_encoder = PEEncoder(triton_pool=AppResources.async_triton_pool)
    try:
        app.state.pe_encoder.warm_text_encoder()
        logger.info('pe_text_encoder_warmed')
    except Exception as exc:
        logger.warning('pe_text_encoder_warm_skipped', error=str(exc))

    logger.info(
        'service_ready',
        triton_url=settings.triton_url,
        opensearch_url=settings.opensearch_url,
    )

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info('shutdown_begin', phase='cleanup')

    # Close AsyncTritonPool
    if AppResources.async_triton_pool is not None:
        try:
            await AppResources.async_triton_pool.close()
            logger.info('triton_pool_closed')
        except Exception as e:
            logger.warning('triton_pool_close_error', error=str(e))

    # Shutdown shared executor
    if AppResources.shared_executor is not None:
        try:
            AppResources.shared_executor.shutdown(wait=True)
            logger.info('executor_shutdown')
        except Exception as e:
            logger.warning('executor_shutdown_error', error=str(e))

    # Close Triton connections
    try:
        await TritonClientFactory.close_all()
    except Exception as e:
        logger.warning('triton_client_close_error', error=str(e))

    # Close OpenSearch connections
    try:
        await OpenSearchClientFactory.close()
    except Exception as e:
        logger.warning('opensearch_close_error', error=str(e))

    logger.info('shutdown_complete')


# =============================================================================
# FastAPI Application Factory
# =============================================================================
def _read_version() -> str:
    """Read version from VERSION file, falling back to settings default."""
    version_file = Path(__file__).resolve().parent.parent / 'VERSION'
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
        default_response_class=ORJSONResponse,
    )

    # CORS — allow a labeler/curation frontend and any LAN client to reach
    # the API. In production a reverse proxy usually handles routing so
    # cross-origin calls are rare, but this covers: dev mode (vite/webpack
    # dev servers on a different port), direct API access from LAN IPs, and
    # any other internal network clients. Ported from the reference
    # implementation's CORS block — dropped during the initial OSS port,
    # which broke any frontend dev server talking to this API
    # cross-origin (browser fetch fails with "Failed to fetch"/no CORS
    # headers, even though the server itself processes and logs the
    # request as 200).
    from fastapi.middleware.cors import CORSMiddleware

    application.add_middleware(
        CORSMiddleware,
        allow_origin_regex=(
            r'^https?://(localhost|127\.0\.0\.1|host\.docker\.internal'
            r'|192\.168\.\d+\.\d+'  # RFC-1918 class C
            r'|10\.\d+\.\d+\.\d+'  # RFC-1918 class A
            r'|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+'  # RFC-1918 class B
            r')(:\d+)?$'
        ),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
        expose_headers=['X-Request-ID', 'X-Process-Time'],
    )

    # Performance Middleware (defined first, runs second in LIFO order)
    @application.middleware('http')
    async def performance_middleware(request: Request, call_next):
        """
        Monitor request performance, validate file size, and inject timing into response.

        Industry standard: timing included in both header (X-Process-Time) and response body.
        """
        start_time = time.time()
        req_id = get_request_id()  # Get from context set by request_id_middleware

        # Validate file size for upload endpoints
        if request.method == 'POST':
            content_length = request.headers.get('content-length')
            # Batch endpoints allow up to 10GB for large photo library processing
            # Single endpoints use default limit (50MB)
            is_batch_endpoint = '/batch' in request.url.path
            max_size = (
                10 * 1024 * 1024 * 1024 if is_batch_endpoint else settings.max_file_size_bytes
            )
            max_size_label = '10GB' if is_batch_endpoint else f'{settings.max_file_size_mb}MB'
            if content_length and int(content_length) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f'File too large. Maximum: {max_size_label}',
                )

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
        return ORJSONResponse(
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
    application.include_router(curation_router)  # /curation/* - Curation/labeling pipeline
    application.include_router(curation_images_router)  # /curation/images/* - Source image serving
    application.include_router(
        curation_crops_router
    )  # /curation/crops/* - Crop thumbnails/overlays
    application.include_router(curation_umap_router)  # /curation/cluster/* - UMAP residual reducer
    application.include_router(curation_train_router)  # /curation/train/* - Training pipeline

    # Versioned API - All endpoints also available under /v1
    application.include_router(v1_router)  # /v1/* - Versioned API

    # The curation metric registry (src.services.curation.metrics) registers
    # Counter/Histogram objects on the prometheus_client default REGISTRY at
    # import time; import it here so its module-level side effects run even
    # if no curation route has been hit yet — the existing /metrics endpoint
    # (src.routers.health) picks them up automatically via generate_latest().
    import src.services.curation.metrics  # noqa: F401

    return application


# Create application instance
app = create_app()
