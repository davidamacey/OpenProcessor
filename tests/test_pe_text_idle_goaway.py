"""A text query whose first try meets the model server's GOAWAY is answered by one more try.

A gRPC channel that the server closed with GOAWAY (``too_many_pings``, a restart, a proxy's idle
timeout) fails the call that was in flight with UNAVAILABLE and reconnects for the next one
(https://github.com/grpc/grpc/blob/master/doc/connection-backoff.md). Inference and readiness are
reads of the model, so one more try is safe; ``src.utils.retry.retry_sync`` is the retry the sync
Triton clients already use.
"""

from __future__ import annotations

from typing import Any

import grpc
import numpy as np
import pytest
from tritonclient.utils import InferenceServerException

from src.clients.pe_encoder import PEEncoder, TextModelNotReady
from src.clients.pe_text_backends import PE_TEXT_OUTPUT, TritonTextBackend
from src.utils.retry import RetryExhaustedError


GOAWAY = InferenceServerException(msg='Too many pings', status=str(grpc.StatusCode.UNAVAILABLE))


class _Result:
    def as_numpy(self, name: str) -> np.ndarray:
        assert name == PE_TEXT_OUTPUT
        return np.ones((1, 4), dtype=np.float32)


class _Client:
    """Fails its first ``failures`` calls with ``error``, then answers."""

    def __init__(self, failures: int, error: Exception = GOAWAY) -> None:
        self.failures = failures
        self.error = error
        self.infers = 0
        self.readiness = 0

    def infer(self, *_args: Any, **_kwargs: Any) -> _Result:
        self.infers += 1
        if self.infers <= self.failures:
            raise self.error
        return _Result()

    def is_model_ready(self, _name: str) -> bool:
        self.readiness += 1
        if self.readiness <= self.failures:
            raise self.error
        return True


TOKENS = np.zeros((1, 3), dtype=np.int64)


def test_a_goaway_on_the_first_try_of_an_encode_is_answered_by_the_second() -> None:
    client = _Client(failures=1)
    out = TritonTextBackend(lambda: client).encode(TOKENS)
    assert out.shape == (1, 4)
    assert client.infers == 2


def test_two_goaways_in_a_row_surface() -> None:
    client = _Client(failures=2)
    with pytest.raises(RetryExhaustedError) as caught:
        TritonTextBackend(lambda: client).encode(TOKENS)
    assert caught.value.last_error is GOAWAY
    assert client.infers == 2


def test_an_error_that_is_not_transient_is_not_retried() -> None:
    bad = InferenceServerException(msg='bad shape', status=str(grpc.StatusCode.INVALID_ARGUMENT))
    client = _Client(failures=5, error=bad)
    with pytest.raises(InferenceServerException):
        TritonTextBackend(lambda: client).encode(TOKENS)
    assert client.infers == 1


def _encoder(client: _Client) -> PEEncoder:
    return PEEncoder(triton_client_factory=lambda: client)


def test_a_goaway_on_the_readiness_check_does_not_make_the_text_model_not_ready() -> None:
    client = _Client(failures=1)
    assert _encoder(client)._triton_backend().model_name
    assert client.readiness == 2


def test_a_readiness_check_that_fails_twice_is_not_ready() -> None:
    client = _Client(failures=2)
    with pytest.raises(TextModelNotReady):
        _encoder(client)._triton_backend()
