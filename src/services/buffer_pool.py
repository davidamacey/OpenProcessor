"""
Buffer Pooling for Zero-Allocation Hot Path.

Pre-allocated numpy buffer pools to eliminate allocation overhead
in high-throughput inference pipelines.

Pattern from clip-retrieval achieving 1,500 img/sec:
- Pre-allocate buffers at startup
- Acquire/release pattern for hot path
- Fallback allocation when pool exhausted (rare in steady state)

Usage:
    # Get buffer from pool
    buffer = YOLO_BUFFER_POOL.acquire()

    # Use buffer for preprocessing
    np.copyto(buffer, preprocessed_data)

    # Return to pool when done
    YOLO_BUFFER_POOL.release(buffer)

    # Async context manager
    async with CLIP_BUFFER_POOL.acquire_async() as buffer:
        # Use buffer
        pass  # Automatically released
"""

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from threading import Lock
from typing import Generator

import numpy as np


logger = logging.getLogger(__name__)


class BufferPool:
    """
    Pre-allocated buffer pool to eliminate allocation overhead.

    Thread-safe implementation for sync code paths.
    For async code, use acquire_async() context manager.

    Benefits:
    - No GC pressure in hot path
    - Consistent memory usage
    - Predictable allocation behavior
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
        pool_size: int = 32,
        name: str = 'buffer_pool',
    ):
        """
        Initialize buffer pool with pre-allocated arrays.

        Args:
            shape: Shape of each buffer (e.g., (1, 3, 640, 640))
            dtype: NumPy dtype (default: float32)
            pool_size: Number of buffers to pre-allocate
            name: Pool name for logging
        """
        self.shape = shape
        self.dtype = dtype
        self.pool_size = pool_size
        self.name = name

        self._pool: deque[np.ndarray] = deque()
        self._lock = Lock()
        self._async_lock: asyncio.Lock | None = None

        # Statistics
        self._allocations = 0
        self._reuses = 0
        self._fallback_allocations = 0

        # Pre-allocate buffers
        for _ in range(pool_size):
            self._pool.append(np.empty(shape, dtype=dtype))

        logger.debug(
            f'BufferPool "{name}" initialized: {pool_size} buffers, '
            f'shape={shape}, dtype={dtype}'
        )

    def acquire(self) -> np.ndarray:
        """
        Get buffer from pool (sync).

        Returns:
            Pre-allocated numpy array from pool, or new array if pool exhausted.
        """
        with self._lock:
            if self._pool:
                self._reuses += 1
                return self._pool.popleft()

        # Pool exhausted - allocate (should be rare in steady state)
        self._fallback_allocations += 1
        if self._fallback_allocations % 100 == 1:
            logger.warning(
                f'BufferPool "{self.name}" exhausted, allocating new buffer '
                f'(fallback #{self._fallback_allocations})'
            )
        return np.empty(self.shape, dtype=self.dtype)

    def release(self, buffer: np.ndarray) -> None:
        """
        Return buffer to pool.

        Args:
            buffer: Buffer to return (must match pool shape/dtype)
        """
        # Only return if it matches our shape (safety check)
        if buffer.shape != self.shape or buffer.dtype != self.dtype:
            logger.warning(
                f'BufferPool "{self.name}": releasing mismatched buffer, discarding'
            )
            return

        with self._lock:
            if len(self._pool) < self.pool_size:
                self._pool.append(buffer)
            # Else discard (pool full - shouldn't happen often)

    @contextmanager
    def acquire_ctx(self) -> Generator[np.ndarray, None, None]:
        """
        Context manager for buffer acquisition (sync).

        Usage:
            with pool.acquire_ctx() as buffer:
                # Use buffer
                pass  # Automatically released
        """
        buffer = self.acquire()
        try:
            yield buffer
        finally:
            self.release(buffer)

    @asynccontextmanager
    async def acquire_async(self):
        """
        Async context manager for buffer acquisition.

        Usage:
            async with pool.acquire_async() as buffer:
                # Use buffer
                pass  # Automatically released
        """
        # Create async lock on first use (must be in event loop context)
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            buffer = self.acquire()
        try:
            yield buffer
        finally:
            async with self._async_lock:
                self.release(buffer)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return {
                'name': self.name,
                'shape': self.shape,
                'dtype': str(self.dtype),
                'pool_size': self.pool_size,
                'available': len(self._pool),
                'reuses': self._reuses,
                'fallback_allocations': self._fallback_allocations,
                'hit_rate': (
                    self._reuses / (self._reuses + self._fallback_allocations) * 100
                    if (self._reuses + self._fallback_allocations) > 0
                    else 100.0
                ),
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._reuses = 0
            self._fallback_allocations = 0


class BatchBufferPool:
    """
    Dynamic batch buffer pool for variable batch sizes.

    Maintains pools for common batch sizes to optimize reuse.
    """

    def __init__(
        self,
        base_shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
        max_batch_size: int = 64,
        buffers_per_size: int = 4,
        name: str = 'batch_pool',
    ):
        """
        Initialize batch buffer pool.

        Args:
            base_shape: Shape per sample (e.g., (3, 640, 640))
            dtype: NumPy dtype
            max_batch_size: Maximum batch size to pre-allocate
            buffers_per_size: Buffers per batch size
            name: Pool name for logging
        """
        self.base_shape = base_shape
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.name = name

        # Pre-allocate pools for common batch sizes: 1, 2, 4, 8, 16, 32, 64
        self._pools: dict[int, BufferPool] = {}
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

        for bs in batch_sizes:
            if bs <= max_batch_size:
                batch_shape = (bs,) + base_shape
                self._pools[bs] = BufferPool(
                    shape=batch_shape,
                    dtype=dtype,
                    pool_size=buffers_per_size,
                    name=f'{name}_batch{bs}',
                )

        logger.debug(
            f'BatchBufferPool "{name}" initialized: batch_sizes={list(self._pools.keys())}'
        )

    def acquire(self, batch_size: int) -> np.ndarray:
        """
        Get buffer for specific batch size.

        Returns smallest pooled buffer that fits, or allocates if needed.
        """
        # Find smallest pool that fits
        for bs in sorted(self._pools.keys()):
            if bs >= batch_size:
                return self._pools[bs].acquire()

        # No pool large enough - allocate
        batch_shape = (batch_size,) + self.base_shape
        return np.empty(batch_shape, dtype=self.dtype)

    def release(self, buffer: np.ndarray) -> None:
        """Return buffer to appropriate pool."""
        batch_size = buffer.shape[0]
        if batch_size in self._pools:
            self._pools[batch_size].release(buffer)
        # Else discard (non-standard size)

    def get_stats(self) -> dict:
        """Get combined statistics for all pools."""
        return {
            'name': self.name,
            'base_shape': self.base_shape,
            'pools': {bs: pool.get_stats() for bs, pool in self._pools.items()},
        }


# =============================================================================
# Pre-configured Buffer Pools for Common Tensor Shapes
# =============================================================================

# YOLO detection (640x640)
YOLO_BUFFER_POOL = BufferPool(
    shape=(1, 3, 640, 640),
    dtype=np.float32,
    pool_size=64,
    name='yolo_640',
)

# YOLO batch (for face detection on person crops)
YOLO_BATCH_POOL = BatchBufferPool(
    base_shape=(3, 640, 640),
    dtype=np.float32,
    max_batch_size=64,
    buffers_per_size=8,
    name='yolo_batch',
)

# MobileCLIP image encoder (256x256)
CLIP_BUFFER_POOL = BufferPool(
    shape=(1, 3, 256, 256),
    dtype=np.float32,
    pool_size=64,
    name='clip_256',
)

# MobileCLIP batch (for box embeddings)
CLIP_BATCH_POOL = BatchBufferPool(
    base_shape=(3, 256, 256),
    dtype=np.float32,
    max_batch_size=128,
    buffers_per_size=8,
    name='clip_batch',
)

# ArcFace (112x112)
ARCFACE_BUFFER_POOL = BufferPool(
    shape=(1, 3, 112, 112),
    dtype=np.float32,
    pool_size=32,
    name='arcface_112',
)

# ArcFace batch (for multiple faces)
ARCFACE_BATCH_POOL = BatchBufferPool(
    base_shape=(3, 112, 112),
    dtype=np.float32,
    max_batch_size=64,
    buffers_per_size=8,
    name='arcface_batch',
)

# OCR (960x960)
OCR_BUFFER_POOL = BufferPool(
    shape=(1, 3, 960, 960),
    dtype=np.float32,
    pool_size=16,
    name='ocr_960',
)


def get_all_pool_stats() -> dict:
    """Get statistics for all pre-configured pools."""
    return {
        'yolo': YOLO_BUFFER_POOL.get_stats(),
        'yolo_batch': YOLO_BATCH_POOL.get_stats(),
        'clip': CLIP_BUFFER_POOL.get_stats(),
        'clip_batch': CLIP_BATCH_POOL.get_stats(),
        'arcface': ARCFACE_BUFFER_POOL.get_stats(),
        'arcface_batch': ARCFACE_BATCH_POOL.get_stats(),
        'ocr': OCR_BUFFER_POOL.get_stats(),
    }
