"""Shared application resources (managed by the lifespan)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.clients.triton_pool_core import AsyncTritonPool


class AppResources:
    """Container for shared application resources."""

    shared_executor: ThreadPoolExecutor | None = None
    async_triton_pool: AsyncTritonPool | None = None
    arbiter_task: asyncio.Task[None] | None = None
    curation_knn_warmup_task: asyncio.Task[None] | None = None
    project_registry_poll_task: asyncio.Task[None] | None = None
    ops_metrics_task: asyncio.Task[None] | None = None
    config_store_poll_task: asyncio.Task[None] | None = None
    curation_background_jobs: list[asyncio.Task[None]] = []
    event_bus_started: bool = False


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
