"""Sync Triton calls carry a client deadline; a missed deadline answers 504, never a retry."""

from __future__ import annotations

import io

import grpc
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tritonclient.utils import InferenceServerException

from src.clients.triton_deadline import (
    TRITON_SYNC_CLIENT_TIMEOUT_S,
    ainfer_with_deadline,
    infer_with_deadline,
)
from src.utils.retry import TritonDeadlineError, retry_sync


class _Hung:
    """Answers like tritonclient when ``client_timeout`` elapses; waits forever without one."""

    def __init__(self) -> None:
        self.calls: list[float | None] = []

    def infer(self, *_args, **kwargs):
        client_timeout = kwargs.get('client_timeout')
        self.calls.append(client_timeout)
        if client_timeout is None:
            raise AssertionError('infer called without client_timeout (waits forever)')
        raise InferenceServerException(
            msg='Deadline Exceeded', status=str(grpc.StatusCode.DEADLINE_EXCEEDED)
        )


class _Fast:
    def infer(self, *_args, **kwargs):
        assert kwargs['client_timeout'] == TRITON_SYNC_CLIENT_TIMEOUT_S
        return 'ok'


def test_a_hung_call_raises_the_deadline_error_without_retrying() -> None:
    client = _Hung()
    with pytest.raises(TritonDeadlineError):
        retry_sync(infer_with_deadline, client, 'm', [], [], max_retries=3, base_delay=0)
    assert client.calls == [TRITON_SYNC_CLIENT_TIMEOUT_S]


def test_a_fast_call_succeeds() -> None:
    assert infer_with_deadline(_Fast(), 'm', [], []) == 'ok'


@pytest.mark.asyncio
async def test_the_async_twin_raises_the_deadline_error() -> None:
    class Aio:
        async def infer(self, *_args, **kwargs):
            assert kwargs['client_timeout'] == TRITON_SYNC_CLIENT_TIMEOUT_S
            raise InferenceServerException(
                msg='Deadline Exceeded', status=str(grpc.StatusCode.DEADLINE_EXCEEDED)
            )

    with pytest.raises(TritonDeadlineError):
        await ainfer_with_deadline(Aio(), 'm', [], [])


def test_the_route_answers_504(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.main import app
    from src.services.inference import InferenceService

    def hung(*_a, **_k):
        infer_with_deadline(_Hung(), 'm', [], [])

    monkeypatch.setattr(InferenceService, 'encode_image', hung)
    buf = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buf, format='JPEG')
    response = TestClient(app).post(
        '/embed/image', files={'image': ('t.jpg', buf.getvalue(), 'image/jpeg')}
    )
    assert response.status_code == 504
    assert response.json()['error_type'] == 'TritonDeadlineError'
