"""A 503 with Retry-After from the ingest route (``database_busy``, Triton down) and a connection
that cannot be made (the API restarts) are a bounded retry of the batch, never a whole-batch
failure; other statuses, and a connection that breaks after the request went out, still fail at
once."""

from __future__ import annotations

import httpx
import pytest

from src.services.curation.bulk_ingest import sender as sender_mod
from src.services.curation.bulk_ingest.sender import BatchRequestError, HttpBatchSender


pytestmark = pytest.mark.asyncio

BUSY = {'error': 'database_busy', 'detail': 'pool timeout'}


slept: list[float] = []


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    slept.clear()

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(sender_mod, 'RETRY_SLEEP', fake_sleep)


def _sender(handler: httpx.MockTransport) -> HttpBatchSender:
    client = httpx.AsyncClient(transport=handler, base_url='http://api')
    return HttpBatchSender(client, api_prefix='/curation', project='p')


async def test_busy_503_is_retried_honouring_retry_after() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(503, json=BUSY, headers={'Retry-After': '7'})
        return httpx.Response(200, json={'results': []})

    body = await _sender(httpx.MockTransport(handler)).post_batch(
        ['/a.jpg'], source='s', run_id='r'
    )
    assert body == {'results': []}
    assert calls == 3
    assert len(slept) == 2
    assert all(s >= 7 for s in slept)


async def test_busy_503_gives_up_after_bounded_attempts() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json=BUSY, headers={'Retry-After': '1'})

    with pytest.raises(BatchRequestError, match='HTTP 503'):
        await _sender(httpx.MockTransport(handler)).post_batch(['/a.jpg'], source='s', run_id='r')
    assert calls == sender_mod.BUSY_MAX_ATTEMPTS


async def test_other_status_is_not_retried() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text='boom')

    with pytest.raises(BatchRequestError, match='HTTP 500'):
        await _sender(httpx.MockTransport(handler)).post_batch(['/a.jpg'], source='s', run_id='r')
    assert calls == 1


async def test_a_connection_that_cannot_be_made_is_retried_like_a_busy_503() -> None:
    """The API is restarting: nothing was processed, so the batch is sent again."""
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError('connection refused')
        return httpx.Response(200, json={'results': []})

    body = await _sender(httpx.MockTransport(handler)).post_batch(
        ['/a.jpg'], source='s', run_id='r'
    )
    assert body == {'results': []}
    assert calls == 3
    assert len(slept) == 2


@pytest.mark.parametrize('error', [httpx.ConnectError('refused'), httpx.ConnectTimeout('slow')])
async def test_a_connection_that_never_comes_up_gives_up_as_busy_with_no_retry_after(
    error: httpx.TransportError,
) -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(sender_mod.BatchBusyError) as caught:
        await _sender(httpx.MockTransport(handler)).post_batch(['/a.jpg'], source='s', run_id='r')
    assert calls == sender_mod.BUSY_MAX_ATTEMPTS
    assert caught.value.retry_after == 0.0


@pytest.mark.parametrize(
    'error',
    [httpx.ReadError('reset'), httpx.RemoteProtocolError('closed'), httpx.ReadTimeout('hung')],
)
async def test_a_failure_after_the_request_went_out_is_not_retried_and_not_busy(
    error: httpx.TransportError,
) -> None:
    """It may be a file that crashes or hangs the worker, so it uses an attempt."""
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(BatchRequestError) as caught:
        await _sender(httpx.MockTransport(handler)).post_batch(['/a.jpg'], source='s', run_id='r')
    assert not isinstance(caught.value, sender_mod.BatchBusyError)
    assert calls == 1
