"""SAM3 HTTP client with circuit-breaker + bounded retry (Phase 4e).

Split out of :mod:`scripts.curation.worker.cascade` to keep
``cascade.py`` under the 700-LOC project ceiling. Public surface
re-exports unchanged via :mod:`scripts.curation.worker.cascade`
(``SegmenterAllHostsDown``, ``SegmenterClient``).

Failure model:

- Per-host failure counter (sliding 30s window). 3 failures within
  the window mark that host UNHEALTHY for 60s.
- Round-robin picks healthy hosts only. After 60s a host enters
  HALF_OPEN — the next caller probes it; success closes the circuit,
  failure re-opens for another 60s.
- ``SegmenterAllHostsDown`` is raised when every host is UNHEALTHY so the
  caller can park the crop in ``pending_detection`` rather than
  promoting it to a terminal status on infrastructure noise.
- ``httpx.ReadTimeout`` and ``httpx.ConnectTimeout`` retry up to 2
  more times with 1s, 2s backoff before counting as a failure. Other
  HTTPError classes (5xx, PoolTimeout, ConnectError) count as one
  failure with no retry budget.

The segmenter leg is optional: passing an empty/``None``
``base_url`` (e.g. ``OP_SEGMENTER_URL=''``) constructs a *disabled* client
instead of raising. A disabled client's :meth:`SegmenterClient.segment`
always returns ``None`` — the same "no candidate" result an unhealthy
or empty-response segmenter already produces — without attempting any
HTTP call, so callers that already treat ``None`` as "fall through to
the next cascade step" degrade cleanly with zero code changes. A
deployment with no segmentation service of its own simply leaves
``OP_SEGMENTER_URL`` unset/empty and documents that behavior; see
``docs/design/curation_design_rationale.md``.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from collections import deque
from typing import Any

import httpx

from src.core.logging import get_logger
from src.services.curation.metrics import (
    OP_SEGMENTER_CIRCUIT_OPEN_TOTAL,
    OP_SEGMENTER_REQUEST_INFLIGHT_SECONDS,
    OP_SEGMENTER_REQUEST_RESPONSE_SECONDS,
    OP_SEGMENTER_REQUEST_RETRIES_TOTAL,
    OP_SEGMENTER_REQUEST_WAIT_SECONDS,
)
from src.services.detection.cascade_detect import RegionCandidate


logger = get_logger('curation_worker')


class SegmenterAllHostsDown(RuntimeError):  # noqa: N818
    """Raised when every SAM3 host is marked UNHEALTHY by the circuit breaker.

    Distinct from "SAM3 returned no candidate" — this is an
    infrastructure-level failure and the worker must not promote a
    crop into a terminal status (e.g. ``detection_failed``) based on
    it. Callers should keep the crop in ``pending_detection`` for a
    later cascade pass.
    """


# Circuit-breaker tunables.
_FAILURE_WINDOW_S = 30.0
_FAILURE_THRESHOLD = 3
_OPEN_DURATION_S = 60.0
_MAX_ATTEMPTS = 3  # initial attempt + 2 retries
_RETRY_BACKOFFS = (1.0, 2.0)


class _HostState:
    """Per-host failure tracker used by :class:`SegmenterClient`.

    States:
      - CLOSED      : normal operation; failures are recorded.
      - OPEN        : threshold reached; calls are short-circuited
                       until ``open_until`` clocks past.
      - HALF_OPEN   : ``open_until`` passed; the next call may attempt
                       the host. Success closes the circuit; failure
                       re-opens for another ``_OPEN_DURATION_S``.
    """

    __slots__ = ('failures', 'half_open_in_flight', 'open_until')

    def __init__(self) -> None:
        self.failures: deque[float] = deque()
        self.open_until: float = 0.0
        # Latch so only one caller probes the host on transition out
        # of OPEN; concurrent callers continue to short-circuit until
        # the probe resolves.
        self.half_open_in_flight: bool = False


class SegmenterClient:
    """Thin async wrapper around ``POST /segment``.

    Owns the underlying ``httpx.AsyncClient`` so the worker can share
    connections across calls. Returns the **highest-scoring** candidate
    (still in crop frame) or ``None`` if SAM 3 found nothing.

    See the module docstring for the failure model.
    """

    def __init__(
        self,
        base_url: str | None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
        max_candidates: int = 4,
        text_prompt: str = '',
    ) -> None:
        urls = [u.strip().rstrip('/') for u in (base_url or '').split(',') if u.strip()]
        # No segmenter configured is a supported deployment shape, not
        # an error. Disabled clients skip the HTTP leg entirely (see
        # segment) rather than raising at construction time.
        self.enabled = bool(urls)
        self.base_urls = urls
        self._rr_lock = asyncio.Lock()
        self._rr_idx = 0
        self.timeout_s = timeout_s
        self.max_candidates = max_candidates
        self.text_prompt = text_prompt
        _max_conn = int(os.environ.get('OP_SEGMENTER_HTTPX_MAX_CONNECTIONS', '512'))
        _keepalive = int(os.environ.get('OP_SEGMENTER_HTTPX_KEEPALIVE', '128'))
        _limits = httpx.Limits(
            max_connections=_max_conn,
            max_keepalive_connections=_keepalive,
        )
        self._client = client or httpx.AsyncClient(timeout=timeout_s, limits=_limits)
        self._owns_client = client is None

        self._host_state: dict[str, _HostState] = {u: _HostState() for u in urls}
        self._cb_lock = asyncio.Lock()
        # Indirected for monkeypatching in tests; defaults to time.monotonic.
        self._now = time.monotonic

        if not self.enabled:
            logger.info(
                'segmenter_disabled',
                reason='no segmenter_url configured; segmenter leg skipped',
            )

        if len(urls) > 1:
            logger.info('segmenter_multi_url', urls=urls, count=len(urls))

    @property
    def base_url(self) -> str:
        """Back-compat: return the first URL when callers expect a single one.

        Empty string when disabled (no segmenter configured).
        """
        return self.base_urls[0] if self.base_urls else ''

    async def _next_url(self) -> str:
        """Round-robin pick the next SAM3 URL (legacy; no health check)."""
        if len(self.base_urls) == 1:
            return self.base_urls[0]
        async with self._rr_lock:
            url = self.base_urls[self._rr_idx]
            self._rr_idx = (self._rr_idx + 1) % len(self.base_urls)
        return url

    async def _pick_healthy_url(self) -> str:
        """Round-robin a healthy host. Raises :class:`SegmenterAllHostsDown`.

        Half-open semantics: when a host's ``open_until`` has passed,
        one caller probes the host (latches ``half_open_in_flight``);
        concurrent callers continue to skip until the probe resolves.
        """
        now = self._now()
        async with self._cb_lock:
            n = len(self.base_urls)
            for _ in range(n):
                idx = self._rr_idx
                self._rr_idx = (self._rr_idx + 1) % n
                url = self.base_urls[idx]
                state = self._host_state[url]
                if state.open_until <= now:
                    if state.open_until > 0.0 and not state.half_open_in_flight:
                        state.half_open_in_flight = True
                    elif state.open_until > 0.0 and state.half_open_in_flight:
                        # Another caller is already probing — keep skipping.
                        continue
                    return url
        raise SegmenterAllHostsDown(f'all segmenter hosts unhealthy ({self.base_urls})')

    async def _on_success(self, url: str) -> None:
        async with self._cb_lock:
            state = self._host_state[url]
            state.failures.clear()
            state.open_until = 0.0
            state.half_open_in_flight = False

    async def _on_failure(self, url: str) -> None:
        """Record a failure; open the circuit if threshold crossed."""
        now = self._now()
        async with self._cb_lock:
            state = self._host_state[url]
            cutoff = now - _FAILURE_WINDOW_S
            while state.failures and state.failures[0] < cutoff:
                state.failures.popleft()
            state.failures.append(now)
            if state.half_open_in_flight:
                # Probe failed — re-open the circuit.
                state.half_open_in_flight = False
                state.open_until = now + _OPEN_DURATION_S
                state.failures.clear()
                OP_SEGMENTER_CIRCUIT_OPEN_TOTAL.labels(host=url).inc()
                logger.warning('segmenter_circuit_reopen', url=url, open_for_s=_OPEN_DURATION_S)
                return
            if len(state.failures) >= _FAILURE_THRESHOLD:
                state.open_until = now + _OPEN_DURATION_S
                state.failures.clear()
                OP_SEGMENTER_CIRCUIT_OPEN_TOTAL.labels(host=url).inc()
                logger.warning('segmenter_circuit_open', url=url, open_for_s=_OPEN_DURATION_S)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        timing: dict[str, float] | None = None,
    ) -> httpx.Response | None:
        """Issue the POST with bounded retry on timeout classes only.

        Returns the response on success or ``None`` if every attempt
        failed. Each timeout-class failure counts as one circuit-breaker
        failure on this host (after final attempt).

        If ``timing`` is provided, the dict is mutated in place with two
        bookmarks per call: ``post_started`` (just before ``.post()``)
        and ``post_returned`` (after the final attempt completes — on
        success the last attempt's timestamps win; on failure the most
        recent attempt's timestamps are recorded). Bookmarks use
        ``time.monotonic`` (see ``self._now``) so callers can subtract
        without crossing a clock boundary.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                if timing is not None:
                    timing['post_started'] = self._now()
                resp = await self._client.post(
                    f'{url}/segment',
                    json=payload,
                    timeout=self.timeout_s,
                )
                if timing is not None:
                    timing['post_returned'] = self._now()
                resp.raise_for_status()
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                if timing is not None:
                    timing['post_returned'] = self._now()
                if attempt < _MAX_ATTEMPTS - 1:
                    backoff = _RETRY_BACKOFFS[attempt]
                    logger.warning(
                        'segmenter_retry',
                        url=url,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(backoff)
                    continue
                OP_SEGMENTER_REQUEST_RETRIES_TOTAL.labels(
                    host=url, outcome='failed_after_all_retries'
                ).inc()
                logger.warning(
                    'segmenter_http_error',
                    error=str(exc),
                    error_type=type(exc).__name__,
                    url=url,
                )
                return None
            except httpx.HTTPError as exc:
                # Non-timeout HTTP error (5xx, PoolTimeout, ConnectError).
                # One failure each — no retry budget.
                if timing is not None:
                    timing['post_returned'] = self._now()
                logger.warning(
                    'segmenter_http_error',
                    error=str(exc),
                    error_type=type(exc).__name__,
                    url=url,
                )
                return None
            else:
                if attempt > 0:
                    OP_SEGMENTER_REQUEST_RETRIES_TOTAL.labels(
                        host=url, outcome='success_after_retry'
                    ).inc()
                return resp
        if last_exc is not None:  # pragma: no cover
            logger.warning('segmenter_retry_unreachable', error=str(last_exc))
        return None

    def _record_timings(
        self,
        url: str,
        t0: float,
        timing: dict[str, float],
        *,
        outcome: str,
        t_end: float | None,
    ) -> None:
        """Observe the three split histograms.

        ``timing`` carries the bookmarks set by ``_post_with_retry``;
        if a bookmark is missing (e.g. ``_pick_healthy_url`` raised
        before the POST started — handled at call sites) we observe 0
        for the relevant bucket rather than skip. All durations are
        clamped to ``>= 0`` to defend against monotonic clock jitter
        and the no-op fast path where ``t_end is None`` (request never
        produced a body — only wait + inflight populate).
        """
        post_started = timing.get('post_started')
        post_returned = timing.get('post_returned')
        wait = max(0.0, (post_started - t0)) if post_started is not None else 0.0
        if post_started is not None and post_returned is not None:
            inflight = max(0.0, post_returned - post_started)
        else:
            inflight = 0.0
        if t_end is not None and post_returned is not None:
            response = max(0.0, t_end - post_returned)
        else:
            response = 0.0
        OP_SEGMENTER_REQUEST_WAIT_SECONDS.labels(host=url, outcome=outcome).observe(wait)
        OP_SEGMENTER_REQUEST_INFLIGHT_SECONDS.labels(host=url, outcome=outcome).observe(inflight)
        OP_SEGMENTER_REQUEST_RESPONSE_SECONDS.labels(host=url, outcome=outcome).observe(response)

    async def segment(self, crop_jpeg: bytes) -> RegionCandidate | None:
        """Segment one crop. Returns the top candidate in crop frame.

        Raises :class:`SegmenterAllHostsDown` if every host is UNHEALTHY.
        Returns ``None`` on a single-host failure (recorded against
        the circuit breaker), when SAM3 returned no candidate, or
        when this client is disabled — no segmenter configured.
        The disabled case never attempts an HTTP call.
        """
        if not self.enabled:
            return None
        # t0 = entry to segment (before any client-side work).
        # See module docstring + metrics.py for the wait/inflight/response
        # decomposition rationale.
        t0 = self._now()
        b64 = base64.b64encode(crop_jpeg).decode('ascii')
        payload = {
            'crop_jpeg_b64': b64,
            'text_prompt': self.text_prompt,
            'max_candidates': self.max_candidates,
        }
        url = await self._pick_healthy_url()
        timing: dict[str, float] = {}
        resp = await self._post_with_retry(url, payload, timing=timing)
        if resp is None:
            self._record_timings(url, t0, timing, outcome='error', t_end=None)
            await self._on_failure(url)
            return None

        try:
            body = resp.json()
        except ValueError:
            t_end = self._now()
            self._record_timings(url, t0, timing, outcome='error', t_end=t_end)
            logger.warning('segmenter_bad_json')
            await self._on_failure(url)
            return None

        await self._on_success(url)

        cands = body.get('candidates') or []
        if not cands:
            t_end = self._now()
            self._record_timings(url, t0, timing, outcome='miss', t_end=t_end)
            return None
        top = max(cands, key=lambda c: float(c.get('score') or 0.0))
        bbox = top.get('bbox_norm')
        if not bbox or len(bbox) != 4:
            t_end = self._now()
            self._record_timings(url, t0, timing, outcome='miss', t_end=t_end)
            return None
        t_end = self._now()
        self._record_timings(url, t0, timing, outcome='hit', t_end=t_end)
        return RegionCandidate(
            bbox_norm=(
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            ),
            score=float(top.get('score') or 0.0),
            source='sam3',
            rectangularity=(float(top['mask_iou']) if top.get('mask_iou') is not None else None),
        )


__all__ = ['SegmenterAllHostsDown', 'SegmenterClient']
