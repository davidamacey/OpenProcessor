"""A 503 with Retry-After from the ingest route (``database_busy``, Triton down) is a bounded
retry of the batch, never a whole-batch failure; other statuses still fail at once."""

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
