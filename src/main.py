"""
Unified Visual AI API Service.

A high-performance FastAPI service providing comprehensive visual AI capabilities:
- Object detection (YOLO11)
- Face detection and recognition (YOLO11-face + ArcFace)
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
from contextvars import ContextVar

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response

from src.clients.triton_pool import AsyncTritonPool
from src.config import get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory
from src.core.logging import configure_logging, get_logger
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
)


# =============================================================================
# Request Context (for correlation IDs)
# =============================================================================

# Context variable to store current request ID (thread-safe)
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='-')


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_ctx.get()


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
async def lifespan(app: FastAPI):  # noqa: ARG001 - Required by FastAPI lifespan API contract
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
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title='Visual AI API',
        description=(
            'High-performance visual AI service providing object detection, '
            'face recognition, image embeddings, visual search, and OCR. '
            'All inference runs through NVIDIA Triton Inference Server.'
        ),
        version='6.0.0',
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
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

    # Request ID Middleware (defined last, runs first in LIFO order)
    @application.middleware('http')
    async def request_id_middleware(request: Request, call_next):
        """
        Add correlation ID (X-Request-ID) to all requests.

        If client provides X-Request-ID header, use it. Otherwise generate a new UUID.
        The request ID is available via get_request_id() in any code path.
        """
        # Get or generate request ID
        req_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
        request_id_ctx.set(req_id)

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

    return application


# Create application instance
app = create_app()
