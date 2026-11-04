"""One retry rule for Triton errors: decided on ``InferenceServerException.status()``."""

from __future__ import annotations

import asyncio
from typing import Any

import grpc
import pytest
from tritonclient.utils import InferenceServerException

from src.clients.triton_pool_core import AsyncTritonPool
from src.utils.retry import is_retryable_error


def _err(code: grpc.StatusCode) -> InferenceServerException:
    # tritonclient get_error_grpc builds status=str(rpc_error.code())
    return InferenceServerException(msg='x', status=str(code))


@pytest.mark.parametrize(
    ('code', 'expected'),
    [
        (grpc.StatusCode.UNAVAILABLE, True),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, True),
        (grpc.StatusCode.ABORTED, True),
        (grpc.StatusCode.INVALID_ARGUMENT, False),
        (grpc.StatusCode.DEADLINE_EXCEEDED, False),
    ],
)
def test_the_status_table(code: grpc.StatusCode, expected: bool) -> None:
    assert is_retryable_error(_err(code)) is expected


def test_a_plain_exception_is_not_retried_on_its_message() -> None:
    assert is_retryable_error(RuntimeError('server unavailable, timeout')) is False


class _Client:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def is_model_ready(self, *_a: Any, **_k: Any) -> bool:
        return True

    async def infer(self, *_a: Any, **_k: Any) -> Any:
        self.calls += 1
        raise self.error


async def _calls(error: Exception) -> int:
    pool = AsyncTritonPool(url='unused:1', pool_size=1, max_concurrent=2)
    client = _Client(error)
    pool._clients = [client]  # type: ignore[list-item]
    pool._semaphore = asyncio.Semaphore(2)
    pool._lock = asyncio.Lock()
    pool._stats.requests_per_channel[0] = 0
    pool._initialized = True
    with pytest.raises(InferenceServerException):
        await pool.infer('m', [], [])
    return client.calls


@pytest.mark.asyncio
async def test_the_pool_does_not_retry_invalid_argument() -> None:
    assert await _calls(_err(grpc.StatusCode.INVALID_ARGUMENT)) == 1


@pytest.mark.asyncio
async def test_the_pool_retries_unavailable() -> None:
    assert await _calls(_err(grpc.StatusCode.UNAVAILABLE)) == 3
