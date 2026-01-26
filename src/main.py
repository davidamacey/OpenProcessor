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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response

from src.clients.triton_pool import AsyncTritonPool
from src.config import get_settings
from src.core.dependencies import OpenSearchClientFactory, TritonClientFactory
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
    query_router,
    search_router,
)


# =============================================================================
# Shared Resources (managed by lifespan)
# =============================================================================
# Shared ThreadPoolExecutor for CPU-bound tasks (avoid per-request creation overhead)
_shared_executor: ThreadPoolExecutor | None = None

# High-throughput async Triton pool with true connection pooling
_async_triton_pool: AsyncTritonPool | None = None


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
    if _shared_executor is None:
        raise RuntimeError('Shared executor not initialized. Call during lifespan.')
    return _shared_executor


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
    if _async_triton_pool is None:
        raise RuntimeError('AsyncTritonPool not initialized. Call during lifespan.')
    return _async_triton_pool


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    global _shared_executor, _async_triton_pool

    settings = get_settings()

    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info('=== STARTUP: Initializing Resources ===')

    # Create shared ThreadPoolExecutor for CPU-bound tasks
    # (JPEG decode, resize, preprocessing)
    _shared_executor = ThreadPoolExecutor(
        max_workers=64,
        thread_name_prefix='ingest-worker-',
    )
    logger.info('Shared ThreadPoolExecutor initialized (64 workers)')

    # Create high-throughput async Triton connection pool
    # 4 gRPC channels with different user-agents = separate TCP connections
    _async_triton_pool = AsyncTritonPool(
        url=settings.triton_url,
        pool_size=4,
        max_concurrent=64,
        verbose=False,
    )
    await _async_triton_pool.initialize()
    logger.info('AsyncTritonPool initialized (4 channels, max_concurrent=64)')

    logger.info('=== SERVICE READY ===')
    logger.info(f'Triton URL: {settings.triton_url}')
    logger.info(f'OpenSearch URL: {settings.opensearch_url}')

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info('=== SHUTDOWN: Cleaning Up ===')

    # Close AsyncTritonPool
    if _async_triton_pool is not None:
        try:
            await _async_triton_pool.close()
            logger.info('AsyncTritonPool closed')
        except Exception as e:
            logger.warning(f'Error closing AsyncTritonPool: {e}')

    # Shutdown shared executor
    if _shared_executor is not None:
        try:
            _shared_executor.shutdown(wait=True)
            logger.info('Shared ThreadPoolExecutor shutdown')
        except Exception as e:
            logger.warning(f'Error shutting down executor: {e}')

    # Close Triton connections
    try:
        await TritonClientFactory.close_all()
    except Exception as e:
        logger.warning(f'Error closing Triton clients: {e}')

    # Close OpenSearch connections
    try:
        await OpenSearchClientFactory.close()
    except Exception as e:
        logger.warning(f'Error closing OpenSearch client: {e}')

    logger.info('=== SHUTDOWN COMPLETE ===')


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

    # Performance Middleware
    @application.middleware('http')
    async def performance_middleware(request: Request, call_next):
        """
        Monitor request performance, validate file size, and inject timing into response.

        Industry standard: timing included in both header (X-Process-Time) and response body.
        """
        start_time = time.time()

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
                # Parse and inject timing
                data = orjson.loads(body)
                if isinstance(data, dict):
                    data['total_time_ms'] = round(duration_ms, 2)
                body = orjson.dumps(data)

                # Build new headers without content-length (will be recalculated)
                new_headers = {
                    k: v for k, v in response.headers.items() if k.lower() != 'content-length'
                }
                new_headers['X-Process-Time'] = f'{duration_ms:.2f}ms'

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

        # Log slow requests
        if duration_ms > settings.slow_request_threshold_ms:
            logger.warning(
                f'Slow request: {request.method} {request.url.path} - {duration_ms:.2f}ms'
            )

        return response

    # Include Routers - Clean API structure without track naming
    application.include_router(health_router)      # /health - Health checks
    application.include_router(detect_router)      # /detect - Object detection
    application.include_router(faces_router)       # /faces - Face detection/recognition
    application.include_router(embed_router)       # /embed - CLIP embeddings
    application.include_router(search_router)      # /search - Visual similarity search
    application.include_router(ingest_router)      # /ingest - Data ingestion
    application.include_router(analyze_router)     # /analyze - Combined analysis
    application.include_router(clusters_router)    # /clusters - Clustering/albums
    application.include_router(query_router)       # /query - Data retrieval
    application.include_router(ocr_router)         # /ocr - Text extraction
    application.include_router(models_router)      # /models - Model management

    return application


# Create application instance
app = create_app()
