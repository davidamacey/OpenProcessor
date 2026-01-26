"""
Unified Triton gRPC Client Pool.

Provides both sync and async client management with connection pooling.
This is the canonical implementation - use this instead of the legacy
triton_async_client.py and triton_shared_client.py in utils/.

Features:
- Singleton pattern for connection reuse
- Thread-safe sync client access
- Async-safe async client access
- Connection pooling enables Triton dynamic batching (5-10x throughput)
- True gRPC connection pool with round-robin selection (AsyncTritonPool)
- Backpressure via semaphore to prevent server overload
- Monitoring and statistics

Usage:
    # Sync client (for background tasks, legacy code)
    client = TritonClientManager.get_sync_client("triton-api:8001")

    # Async client (for FastAPI endpoints - RECOMMENDED)
    client = await TritonClientManager.get_async_client("triton-api:8001")

    # High-throughput async pool (for batch ingestion)
    pool = AsyncTritonPool(url, pool_size=4, max_concurrent=64)
    await pool.initialize()
    result = await pool.infer(model_name, inputs, outputs)

    # Cleanup on shutdown
    await TritonClientManager.close_all()
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio


logger = logging.getLogger(__name__)


# =============================================================================
# gRPC Channel Options - Optimized for High Throughput ML Inference
# =============================================================================
# Based on Fortune 100 best practices (Google, NVIDIA, Netflix)
# Reference: https://grpc.io/docs/guides/performance/
GRPC_CHANNEL_OPTIONS = [
    # Keep connections alive (prevents reconnection overhead)
    ('grpc.keepalive_time_ms', 30000),  # Send keepalive ping every 30s
    ('grpc.keepalive_timeout_ms', 10000),  # Wait 10s for keepalive response
    ('grpc.keepalive_permit_without_calls', 1),  # Allow keepalive when idle

    # Handle large tensors (embeddings, image batches)
    # 100MB limit handles batch=128 × 512-dim embeddings + overhead
    ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB

    # Connection pooling - allow many concurrent streams per connection
    ('grpc.max_concurrent_streams', 1000),

    # HTTP/2 flow control optimization
    ('grpc.http2.min_time_between_pings_ms', 10000),  # Min 10s between pings
    ('grpc.http2.max_pings_without_data', 0),  # Allow pings without data

    # TCP optimization for low latency
    # NOTE: grpc.tcp_nodelay is not supported in Python gRPC, using socket option
    # ('grpc.so_reuseport', 1),  # Enable port reuse for multi-process

    # HTTP/2 window sizes for throughput (larger = more data in flight)
    ('grpc.http2.initial_stream_window_size', 1024 * 1024),  # 1MB per stream
    ('grpc.http2.initial_connection_window_size', 2 * 1024 * 1024),  # 2MB total
]


# =============================================================================
# Pool Statistics
# =============================================================================
@dataclass
class PoolStats:
    """Statistics for AsyncTritonPool."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    active_requests: int = 0
    peak_active_requests: int = 0
    requests_per_channel: dict = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100


# =============================================================================
# AsyncTritonPool - True Connection Pool with Round-Robin Selection
# =============================================================================
class AsyncTritonPool:
    """
    True gRPC connection pool with round-robin selection and backpressure.

    Fortune 100 pattern: Multiple channels with different user-agent forces
    separate TCP connections, enabling true parallel gRPC streams.

    Features:
    - pool_size channels for parallel connections
    - Round-robin client selection for load distribution
    - Semaphore-based backpressure (max_concurrent requests)
    - Statistics tracking for monitoring
    - Automatic retry with exponential backoff

    Usage:
        pool = AsyncTritonPool(url='triton-api:8001', pool_size=4)
        await pool.initialize()

        # Single inference
        result = await pool.infer('model_name', inputs, outputs)

        # Cleanup
        await pool.close()
    """

    def __init__(
        self,
        url: str = 'triton-api:8001',
        pool_size: int = 4,
        max_concurrent: int = 64,
        verbose: bool = False,
    ):
        """
        Initialize the async Triton connection pool.

        Args:
            url: Triton gRPC endpoint (host:port)
            pool_size: Number of gRPC channels to create
            max_concurrent: Maximum concurrent requests (backpressure)
            verbose: Enable verbose Triton client logging
        """
        self.url = url
        self.pool_size = pool_size
        self.max_concurrent = max_concurrent
        self.verbose = verbose

        self._clients: list[grpcclient_aio.InferenceServerClient] = []
        self._semaphore: asyncio.Semaphore | None = None
        self._index = 0
        self._lock: asyncio.Lock | None = None
        self._initialized = False
        self._stats = PoolStats()

    async def initialize(self) -> None:
        """
        Create pool of clients with different channel args for true parallelism.

        Each client gets a unique user-agent, forcing the gRPC library to
        create separate TCP connections (rather than multiplexing on one).
        """
        if self._initialized:
            logger.warning('AsyncTritonPool already initialized')
            return

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._lock = asyncio.Lock()

        for i in range(self.pool_size):
            # CRITICAL: Different user-agent forces separate TCP connections
            channel_args = [
                *GRPC_CHANNEL_OPTIONS,
                ('grpc.primary_user_agent', f'triton-pool-{i}'),
            ]

            client = grpcclient_aio.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                channel_args=channel_args,
            )
            self._clients.append(client)
            self._stats.requests_per_channel[i] = 0

        self._initialized = True
        logger.info(
            f'AsyncTritonPool initialized: {self.pool_size} channels, '
            f'max_concurrent={self.max_concurrent}, url={self.url}'
        )

    async def _get_client_index(self) -> int:
        """Round-robin client selection."""
        async with self._lock:
            idx = self._index
            self._index = (self._index + 1) % self.pool_size
            return idx

    async def infer(
        self,
        model_name: str,
        inputs: list,
        outputs: list,
        timeout: float = 30.0,
        retries: int = 2,
    ) -> Any:
        """
        Inference with automatic backpressure and client selection.

        Args:
            model_name: Name of the Triton model
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects
            timeout: Timeout in seconds
            retries: Number of retry attempts on failure

        Returns:
            InferResult from Triton

        Raises:
            Exception: If inference fails after all retries
        """
        if not self._initialized:
            raise RuntimeError('AsyncTritonPool not initialized. Call initialize() first.')

        # Acquire semaphore (backpressure)
        await self._semaphore.acquire()

        # Track active requests
        self._stats.active_requests += 1
        if self._stats.active_requests > self._stats.peak_active_requests:
            self._stats.peak_active_requests = self._stats.active_requests

        self._stats.total_requests += 1
        start_time = time.perf_counter()
        last_error = None

        try:
            for attempt in range(retries + 1):
                try:
                    # Get client via round-robin
                    client_idx = await self._get_client_index()
                    client = self._clients[client_idx]
                    self._stats.requests_per_channel[client_idx] += 1

                    # Execute inference
                    result = await client.infer(
                        model_name,
                        inputs,
                        outputs=outputs,
                        client_timeout=timeout,
                    )

                    # Success
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self._stats.successful_requests += 1
                    self._stats.total_latency_ms += elapsed_ms

                    return result

                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        # Exponential backoff: 10ms, 20ms, 40ms...
                        await asyncio.sleep(0.01 * (2 ** attempt))
                        logger.debug(
                            f'Retry {attempt + 1}/{retries} for {model_name}: {e}'
                        )
                    continue

            # All retries failed
            self._stats.failed_requests += 1
            raise last_error

        finally:
            self._stats.active_requests -= 1
            self._semaphore.release()

    async def infer_batch(
        self,
        model_name: str,
        inputs_list: list[list],
        outputs: list,
        timeout: float = 30.0,
    ) -> list[Any]:
        """
        Execute multiple inferences concurrently.

        Args:
            model_name: Name of the Triton model
            inputs_list: List of input lists (one per inference)
            outputs: List of InferRequestedOutput objects (shared)
            timeout: Timeout in seconds per inference

        Returns:
            List of InferResult objects
        """
        tasks = [
            self.infer(model_name, inputs, outputs, timeout)
            for inputs in inputs_list
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            'total_requests': self._stats.total_requests,
            'successful_requests': self._stats.successful_requests,
            'failed_requests': self._stats.failed_requests,
            'avg_latency_ms': round(self._stats.avg_latency_ms, 2),
            'success_rate': round(self._stats.success_rate, 2),
            'active_requests': self._stats.active_requests,
            'peak_active_requests': self._stats.peak_active_requests,
            'requests_per_channel': dict(self._stats.requests_per_channel),
            'pool_size': self.pool_size,
            'max_concurrent': self.max_concurrent,
            'url': self.url,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = PoolStats()
        for i in range(self.pool_size):
            self._stats.requests_per_channel[i] = 0

    async def health_check(self) -> bool:
        """Check if all clients are healthy."""
        if not self._initialized:
            return False

        try:
            # Check first client (they all connect to same server)
            return await self._clients[0].is_server_live()
        except Exception as e:
            logger.error(f'Health check failed: {e}')
            return False

    async def close(self) -> None:
        """Close all client connections."""
        if not self._initialized:
            return

        for i, client in enumerate(self._clients):
            try:
                await client.close()
                logger.debug(f'Closed pool client {i}')
            except Exception as e:
                logger.warning(f'Error closing pool client {i}: {e}')

        self._clients.clear()
        self._initialized = False
        logger.info(f'AsyncTritonPool closed (final stats: {self.get_stats()})')


class TritonClientManager:
    """
    Unified manager for Triton gRPC client connections.

    Provides both synchronous and asynchronous client access with
    connection pooling for optimal Triton batching performance.

    Global state justification:
    - Class-level dictionaries store shared client connections per URL
    - Required for connection pooling and enabling Triton's dynamic batching
    - Thread-safe locks prevent race conditions during client creation
    - Alternative (per-request clients) would disable batching and exhaust connections
    """

    # Global sync client pool (thread-safe)
    # Dictionary maps triton_url -> InferenceServerClient instance
    _sync_clients: dict[str, Any] = {}
    _sync_lock = threading.Lock()

    # Global async client pool (async-safe)
    # Dictionary maps triton_url -> AsyncInferenceServerClient instance
    _async_clients: dict[str, Any] = {}
    _async_lock: asyncio.Lock | None = None

    @classmethod
    def _get_async_lock(cls) -> asyncio.Lock:
        """Get or create async lock (must be created in event loop context)."""
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()
        return cls._async_lock

    # =========================================================================
    # Sync Client Pool
    # =========================================================================
    @classmethod
    def get_sync_client(
        cls,
        triton_url: str = 'triton-api:8001',
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: str | None = None,
        private_key: str | None = None,
        certificate_chain: str | None = None,
    ):
        """
        Get sync gRPC client with connection pooling.

        Thread-safe singleton pattern. Multiple callers share the same
        connection, enabling Triton's dynamic batching.

        Args:
            triton_url: Triton gRPC endpoint
            verbose: Enable verbose logging
            ssl: Use SSL/TLS
            root_certificates: SSL root certs path
            private_key: SSL private key path
            certificate_chain: SSL cert chain path

        Returns:
            tritonclient.grpc.InferenceServerClient instance
        """
        with cls._sync_lock:
            if triton_url not in cls._sync_clients:
                try:
                    client = grpcclient.InferenceServerClient(
                        url=triton_url,
                        verbose=verbose,
                        ssl=ssl,
                        root_certificates=root_certificates,
                        private_key=private_key,
                        certificate_chain=certificate_chain,
                        channel_args=GRPC_CHANNEL_OPTIONS,  # Optimized for throughput
                    )

                    cls._sync_clients[triton_url] = client
                    logger.info(f'Created sync Triton client: {triton_url} (optimized gRPC)')

                except Exception as e:
                    logger.error(f'Failed to create sync client for {triton_url}: {e}')
                    raise

            return cls._sync_clients[triton_url]

    @classmethod
    def close_sync_clients(cls):
        """Close all sync client connections."""
        with cls._sync_lock:
            for url, client in cls._sync_clients.items():
                try:
                    client.close()
                    logger.debug(f'Closed sync client: {url}')
                except Exception as e:
                    logger.warning(f'Error closing sync client {url}: {e}')
            cls._sync_clients.clear()
            logger.info('All sync Triton clients closed')

    @classmethod
    def get_sync_stats(cls) -> dict[str, Any]:
        """Get sync client pool statistics."""
        with cls._sync_lock:
            return {
                'active_connections': len(cls._sync_clients),
                'urls': list(cls._sync_clients.keys()),
                'type': 'sync',
            }

    # =========================================================================
    # Async Client Pool
    # =========================================================================
    @classmethod
    async def get_async_client(
        cls,
        triton_url: str = 'triton-api:8001',
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: str | None = None,
        private_key: str | None = None,
        certificate_chain: str | None = None,
    ):
        """
        Get async gRPC client with connection pooling.

        Async-safe singleton pattern. Multiple concurrent requests share
        the same connection, enabling Triton's dynamic batching.

        RECOMMENDED for FastAPI endpoints - native async without thread overhead.

        Args:
            triton_url: Triton gRPC endpoint
            verbose: Enable verbose logging
            ssl: Use SSL/TLS
            root_certificates: SSL root certs path
            private_key: SSL private key path
            certificate_chain: SSL cert chain path

        Returns:
            tritonclient.grpc.aio.InferenceServerClient instance
        """
        async with cls._get_async_lock():
            if triton_url not in cls._async_clients:
                try:
                    client = grpcclient_aio.InferenceServerClient(
                        url=triton_url,
                        verbose=verbose,
                        ssl=ssl,
                        root_certificates=root_certificates,
                        private_key=private_key,
                        certificate_chain=certificate_chain,
                        channel_args=GRPC_CHANNEL_OPTIONS,  # Optimized for throughput
                    )

                    cls._async_clients[triton_url] = client
                    logger.info(f'Created async Triton client: {triton_url} (optimized gRPC + batching)')

                except Exception as e:
                    logger.error(f'Failed to create async client for {triton_url}: {e}')
                    raise

            return cls._async_clients[triton_url]

    @classmethod
    async def close_async_clients(cls):
        """Close all async client connections."""
        async with cls._get_async_lock():
            for url, client in cls._async_clients.items():
                try:
                    await client.close()
                    logger.debug(f'Closed async client: {url}')
                except Exception as e:
                    logger.warning(f'Error closing async client {url}: {e}')
            cls._async_clients.clear()
            logger.info('All async Triton clients closed')

    @classmethod
    async def get_async_stats(cls) -> dict[str, Any]:
        """Get async client pool statistics."""
        async with cls._get_async_lock():
            return {
                'active_connections': len(cls._async_clients),
                'urls': list(cls._async_clients.keys()),
                'type': 'async',
            }

    # =========================================================================
    # Combined Operations
    # =========================================================================
    @classmethod
    async def close_all(cls):
        """Close all clients (both sync and async)."""
        cls.close_sync_clients()
        await cls.close_async_clients()
        logger.info('All Triton clients closed')

    @classmethod
    async def get_all_stats(cls) -> dict[str, Any]:
        """Get combined statistics for all client pools."""
        sync_stats = cls.get_sync_stats()
        async_stats = await cls.get_async_stats()

        return {
            'sync': sync_stats,
            'async': async_stats,
            'total_connections': (
                sync_stats['active_connections'] + async_stats['active_connections']
            ),
        }


# =============================================================================
# Factory Functions (for backward compatibility)
# =============================================================================
def get_triton_client(triton_url: str = 'triton-api:8001', **kwargs):
    """
    Get sync Triton client (backward compatible).

    Prefer TritonClientManager.get_sync_client() for new code.
    """
    return TritonClientManager.get_sync_client(triton_url, **kwargs)


async def get_async_triton_client(triton_url: str = 'triton-api:8001', **kwargs):
    """
    Get async Triton client (backward compatible).

    Prefer TritonClientManager.get_async_client() for new code.
    """
    return await TritonClientManager.get_async_client(triton_url, **kwargs)


def close_all_clients():
    """Close sync clients (backward compatible)."""
    TritonClientManager.close_sync_clients()


async def close_async_clients():
    """Close async clients (backward compatible)."""
    await TritonClientManager.close_async_clients()


def get_client_pool_stats() -> dict[str, Any]:
    """Get sync pool stats (backward compatible)."""
    return TritonClientManager.get_sync_stats()


async def get_async_client_pool_stats() -> dict[str, Any]:
    """Get async pool stats (backward compatible)."""
    return await TritonClientManager.get_async_stats()
