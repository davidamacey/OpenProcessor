"""Transport layer for an OpenAI-compatible vision-language-model endpoint.

Split out of the reference VLM labeler (see
``docs/design/curation_design_rationale.md`` §5 for why this split
exists — vlm_labeler.py is one of the ratchet-exempt oversize files)
— this half owns
"how do I reliably POST to a ``/chat/completions`` endpoint", not "what
do I ask it". Generic: works against any OpenAI-shaped vision chat API
(the reference deployment happens to run behind
OpenWebUI, but nothing here names that model).

Design notes
------------
- Per-process token-bucket rate limit so multiple labeler workers don't
  flood shared upstream capacity.
- Retry-with-backoff (tenacity) on transient 5xx + connection errors.
- ``VlmLabeler`` (``vlm_labeler.py``) owns the actual ``httpx.AsyncClient``
  instance (tests patch it directly), so ``post_chat_with_retry`` takes
  the client + bucket as arguments rather than owning them itself.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logging import get_logger


logger = get_logger(__name__)


# Defaults (env-overridable). No hardcoded vendor model id (S7): a
# deployment that sets OP_VLM_URL without OP_VLM_MODEL must fail loudly at
# VlmLabeler construction (see its __init__) rather than silently talking
# to the reference deployment's model name.
DEFAULT_BASE_URL = os.environ.get('OP_VLM_URL', '')
DEFAULT_MODEL = os.environ.get('OP_VLM_MODEL', '')
DEFAULT_API_KEY = os.environ.get('OP_VLM_API_KEY', 'EMPTY')


def _env_max_images_per_call(default: int = 8) -> int:
    """Per-request image cap for every VLM call (``OP_VLM_MAX_IMAGES_PER_CALL``).

    Must be <= the serving engine's own per-prompt image limit (for vLLM,
    ``--limit-mm-per-prompt '{"image": N}'``) — a request carrying more
    images than that is rejected upstream with a 400. Also acts as the
    hard clamp :class:`VlmLabeler` applies to any explicit
    ``max_images_per_call``. A malformed value raises rather than falling
    back, since a silently-too-high cap turns into upstream 400s.
    """
    raw = os.environ.get('OP_VLM_MAX_IMAGES_PER_CALL', '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f'OP_VLM_MAX_IMAGES_PER_CALL must be a positive integer, got {raw!r}'
        raise ValueError(msg) from exc
    if value < 1:
        msg = f'OP_VLM_MAX_IMAGES_PER_CALL must be >= 1, got {value}'
        raise ValueError(msg)
    return value


DEFAULT_MAX_IMAGES_PER_CALL = _env_max_images_per_call()


# Open-vocab labeling chunk size (label_or_propose_batch). The open-vocab
# prompt is denser than closed-vocab so a smaller default chunk avoids
# empty responses from smaller VLMs. Tunable via the
# OP_VLM_OPEN_IMAGES_PER_CALL env var (default 3; clamped to the
# labeler's max_images_per_call, i.e. OP_VLM_MAX_IMAGES_PER_CALL).
# Governs ONLY the open-vocab chunk size.
def _env_open_images_per_call(default: int = 3) -> int:
    try:
        v = int(os.environ.get('OP_VLM_OPEN_IMAGES_PER_CALL') or str(default))
    except (TypeError, ValueError):
        return default
    return max(1, v)


DEFAULT_OPEN_IMAGES_PER_CALL = _env_open_images_per_call()

# Per-process token-bucket rate limit. Override per call site via
# requests_per_second= when needed.
DEFAULT_REQUESTS_PER_SECOND = 500.0

# Retry budget: 3 attempts, exponential backoff to 60s.
RETRY_MAX_ATTEMPTS = 3
RETRY_WAIT_MIN_S = 1.0
RETRY_WAIT_MAX_S = 60.0


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Tiny asyncio-safe token bucket.

    One token = one upstream HTTP call. ``rate`` tokens are added per
    second up to ``capacity`` (== rate, so burst == 1 second of budget).
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = max(rate, 0.0001)
        self.capacity = capacity if capacity is not None else self.rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then deduct them."""

        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_s = deficit / self.rate
            await asyncio.sleep(wait_s)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def build_http_client(
    timeout_s: float,
    *,
    max_connections: int | None = None,
    max_keepalive_connections: int | None = None,
) -> httpx.AsyncClient:
    """Build a default ``httpx.AsyncClient`` sized for high-concurrency callers.

    The default pool (``max_connections=100``) is too small for a
    high-concurrency worker; callers that need more than the default
    should size it via env (``OP_VLM_HTTPX_MAX_CONNECTIONS`` /
    ``OP_VLM_HTTPX_KEEPALIVE``) or pass explicit values.
    """

    max_conn = max_connections or int(os.environ.get('OP_VLM_HTTPX_MAX_CONNECTIONS') or '512')
    keepalive = max_keepalive_connections or int(os.environ.get('OP_VLM_HTTPX_KEEPALIVE') or '128')
    limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=keepalive)
    return httpx.AsyncClient(timeout=timeout_s, limits=limits)


def build_auth_headers(api_key: str) -> dict[str, str]:
    """Build the ``/chat/completions`` request headers for ``api_key``.

    OpenAI-compatible servers (OpenWebUI / vLLM) accept any non-empty
    bearer token; the conventional placeholder is ``'EMPTY'``.
    """

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


_RETRYABLE_HTTPX_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
)


async def post_chat_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    bucket: _TokenBucket,
) -> dict[str, Any]:
    """POST ``payload`` to ``url`` with rate limiting + retry on 5xx / connection errors."""

    async def _attempt() -> dict[str, Any]:
        await bucket.acquire()
        resp = await client.post(url, headers=headers, json=payload)
        if 500 <= resp.status_code < 600:
            # Trip retry by raising HTTPStatusError.
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=RETRY_WAIT_MIN_S, max=RETRY_WAIT_MAX_S),
            retry=(
                retry_if_exception_type(_RETRYABLE_HTTPX_EXCEPTIONS)
                | retry_if_exception_type(httpx.HTTPStatusError)
            ),
            reraise=True,
        ):
            with attempt:
                return await _attempt()
    except RetryError as exc:  # pragma: no cover - reraise=True covers most cases
        raise exc.last_attempt.exception() from exc

    # AsyncRetrying always returns at least once with reraise=True; this
    # is unreachable but satisfies the type checker.
    raise RuntimeError('AsyncRetrying exited without producing a result')


def extract_message_content(response: dict[str, Any]) -> str:
    """Pull the assistant message text out of an OpenAI-shaped response."""

    try:
        choices = response['choices']
        if not choices:
            return ''
        content = choices[0]['message']['content']
        if isinstance(content, str):
            return content
        # Some servers return a list of content parts; concat the text ones.
        if isinstance(content, list):
            return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
        return ''
    except (KeyError, TypeError, IndexError):
        return ''


def extract_reasoning_content(response: dict[str, Any]) -> str:
    """Pull the assistant's reasoning-channel text, or ``''``.

    A server running a reasoning parser (vLLM ``--reasoning-parser``)
    splits the model output into ``reasoning_content`` (``reasoning`` on
    newer releases) and ``content``. When the model never emits the
    end-of-thinking marker the parser leaves the whole answer -- JSON
    included -- in the reasoning channel and ``content`` empty (or a
    trailing fragment such as ``"]"``).
    """
    try:
        message = response['choices'][0]['message']
    except (KeyError, TypeError, IndexError):
        return ''
    if not isinstance(message, dict):
        return ''
    for key in ('reasoning_content', 'reasoning'):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


__all__ = [
    'DEFAULT_API_KEY',
    'DEFAULT_BASE_URL',
    'DEFAULT_MAX_IMAGES_PER_CALL',
    'DEFAULT_MODEL',
    'DEFAULT_OPEN_IMAGES_PER_CALL',
    'DEFAULT_REQUESTS_PER_SECOND',
    'RETRY_MAX_ATTEMPTS',
    'RETRY_WAIT_MAX_S',
    'RETRY_WAIT_MIN_S',
    '_TokenBucket',
    'build_auth_headers',
    'build_http_client',
    'extract_message_content',
    'extract_reasoning_content',
    'post_chat_with_retry',
]
