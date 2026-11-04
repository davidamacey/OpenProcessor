"""
Retry utilities for guaranteed request processing.

Implements exponential backoff with jitter for resilient inference.
Used to ensure requests are processed even under high load.
"""

import logging
import random
from collections.abc import Callable
from typing import Any, TypeVar

import grpc
from tritonclient.utils import InferenceServerException


logger = logging.getLogger(__name__)

T = TypeVar('T')

# ``InferenceServerException.status()`` carries ``str(grpc.StatusCode)`` for a gRPC failure
# (tritonclient ``get_error_grpc``). DEADLINE_EXCEEDED is not retried: the caller's deadline
# already passed, and INVALID_ARGUMENT never succeeds on a second try.
RETRYABLE_STATUSES = frozenset(
    str(code)
    for code in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
    )
)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def is_retryable_error(error: BaseException) -> bool:
    """Whether a Triton client error is transient: its ``status()`` is in ``RETRYABLE_STATUSES``."""
    return isinstance(error, InferenceServerException) and error.status() in RETRYABLE_STATUSES


def retry_sync[T](
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    **kwargs: Any,
) -> T:
    """
    Retry a sync function with exponential backoff and jitter.

    Same as retry_async but for synchronous functions.
    """
    import time

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not is_retryable_error(e):
                raise

            if attempt == max_retries:
                break

            delay = min(base_delay * (exponential_base**attempt), max_delay)
            # Note: random.random() is acceptable here for non-cryptographic jitter
            jitter_amount = delay * jitter * random.random()
            delay += jitter_amount

            logger.warning(f'Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}')

            time.sleep(delay)

    raise RetryExhaustedError(
        f'All {max_retries} retries exhausted',
        last_error=last_error,
    )
