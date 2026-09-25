"""Phase 4e — SAM3 circuit breaker + bounded retry.

Pins:

* 3 consecutive failures within 30s open the circuit; the 4th call
  short-circuits before any httpx invocation.
* The host recovers after 60s — next call goes through (HALF_OPEN);
  success closes the circuit again.
* ``httpx.ReadTimeout`` triggers up to 2 retries with backoff before
  the call counts as a circuit-breaker failure.
* When every base URL is UNHEALTHY, :class:`Sam3AllHostsDown` is
  raised so the worker can park the crop in ``pending_detection``
  rather than terminating it on infra noise.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from scripts.curation.worker.cascade import Sam3AllHostsDown, Sam3Client


pytestmark = pytest.mark.integration


_CROP_BYTES = b'\xff\xd8\xff\xe0fake-jpeg'  # not parsed by the mock server


def _ok_response(request: httpx.Request) -> httpx.Response:
    body = {
        'candidates': [
            {'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'score': 0.91},
        ]
    }
    return httpx.Response(200, json=body)


def _fail_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={'error': 'boom'})


def _build_client(
    *,
    base_urls: str,
    handler: Any,
    now_func: Any,
) -> Sam3Client:
    transport = httpx.MockTransport(handler)
    httpx_client = httpx.AsyncClient(transport=transport, timeout=5.0)
    sam = Sam3Client(base_url=base_urls, client=httpx_client)
    sam._now = now_func  # type: ignore[assignment]
    return sam


class _FakeClock:
    """Deterministic monotonic clock the Sam3Client can borrow."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _metric_value(counter: Any, **labels: str) -> float:
    return counter.labels(**labels)._value.get()


@pytest.mark.asyncio
async def test_three_failures_open_circuit() -> None:
    """3 x HTTP-500 within 30s open the circuit; 4th call short-circuits."""
    from src.services.curation.metrics import OP_SEGMENTER_CIRCUIT_OPEN_TOTAL

    clock = _FakeClock()
    call_count = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count['n'] += 1
        return _fail_response(request)

    sam = _build_client(
        base_urls='http://sam3-fake-1:7000',
        handler=handler,
        now_func=clock,
    )
    host = 'http://sam3-fake-1:7000'
    before = _metric_value(OP_SEGMENTER_CIRCUIT_OPEN_TOTAL, host=host)

    # 3 failures (each one HTTP 500, no retries — 5xx is not retried).
    for _ in range(3):
        out = await sam.segment_plate(_CROP_BYTES)
        assert out is None
        clock.advance(0.5)

    # 4th call must short-circuit — Sam3AllHostsDown (single host).
    n_before_fourth = call_count['n']
    with pytest.raises(Sam3AllHostsDown):
        await sam.segment_plate(_CROP_BYTES)
    assert call_count['n'] == n_before_fourth, 'expected zero httpx calls past open circuit'

    after = _metric_value(OP_SEGMENTER_CIRCUIT_OPEN_TOTAL, host=host)
    assert after == before + 1, f'expected exactly one open transition, got delta {after - before}'

    await sam.aclose()


@pytest.mark.asyncio
async def test_circuit_recovers_after_60s_window() -> None:
    """After 60s+ the host enters HALF_OPEN; a successful probe re-closes."""
    clock = _FakeClock()
    responses = {'mode': 'fail'}

    def handler(request: httpx.Request) -> httpx.Response:
        if responses['mode'] == 'fail':
            return _fail_response(request)
        return _ok_response(request)

    sam = _build_client(
        base_urls='http://sam3-fake-2:7000',
        handler=handler,
        now_func=clock,
    )

    # Open the circuit.
    for _ in range(3):
        await sam.segment_plate(_CROP_BYTES)
        clock.advance(0.1)
    with pytest.raises(Sam3AllHostsDown):
        await sam.segment_plate(_CROP_BYTES)

    # Advance past the 60s open window, flip mock to success.
    clock.advance(61.0)
    responses['mode'] = 'ok'

    cand = await sam.segment_plate(_CROP_BYTES)
    assert cand is not None, 'HALF_OPEN probe should attempt the host'

    # Circuit closed — a subsequent call also goes through with no
    # short-circuit even before any further time advance.
    cand2 = await sam.segment_plate(_CROP_BYTES)
    assert cand2 is not None

    await sam.aclose()


@pytest.mark.asyncio
async def test_bounded_retry_on_read_timeout() -> None:
    """ReadTimeout then success on retry 2 — call succeeds + retry counter ticks."""
    from src.services.curation.metrics import OP_SEGMENTER_REQUEST_RETRIES_TOTAL

    clock = _FakeClock()
    seq = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seq['n'] += 1
        if seq['n'] == 1:
            raise httpx.ReadTimeout('forced', request=request)
        return _ok_response(request)

    sam = _build_client(
        base_urls='http://sam3-fake-3:7000',
        handler=handler,
        now_func=clock,
    )
    host = 'http://sam3-fake-3:7000'
    before = _metric_value(
        OP_SEGMENTER_REQUEST_RETRIES_TOTAL, host=host, outcome='success_after_retry'
    )

    # Patch asyncio.sleep to a no-op so the test doesn't actually wait 1s.
    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def _no_sleep(_s: float) -> None:
        return None

    _asyncio.sleep = _no_sleep  # type: ignore[assignment]
    try:
        cand = await sam.segment_plate(_CROP_BYTES)
    finally:
        _asyncio.sleep = real_sleep  # type: ignore[assignment]

    assert cand is not None
    assert seq['n'] == 2, 'expected 1 timeout + 1 retry-success = 2 calls'

    after = _metric_value(
        OP_SEGMENTER_REQUEST_RETRIES_TOTAL, host=host, outcome='success_after_retry'
    )
    assert after == before + 1, f'success_after_retry should tick by 1, delta {after - before}'

    await sam.aclose()


@pytest.mark.asyncio
async def test_all_hosts_down_raises() -> None:
    """Two base URLs both UNHEALTHY → Sam3AllHostsDown."""
    clock = _FakeClock()

    sam = _build_client(
        base_urls='http://sam3-fake-4a:7000,http://sam3-fake-4b:7000',
        handler=_fail_response,
        now_func=clock,
    )

    # 6 failures total (3 per host) trip both circuits.
    for _ in range(6):
        out = await sam.segment_plate(_CROP_BYTES)
        assert out is None
        clock.advance(0.05)

    with pytest.raises(Sam3AllHostsDown):
        await sam.segment_plate(_CROP_BYTES)

    await sam.aclose()
