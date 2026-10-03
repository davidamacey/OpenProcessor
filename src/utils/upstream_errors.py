"""How an upstream (VLM, segmenter) failure is worded when it reaches a caller.

An ``httpx`` error message names the request URL, which is an internal service
address; the logs keep the full text, a response body must not.
"""

from __future__ import annotations

import re

import httpx


_URL = re.compile(r'https?://\S+')


def describe_upstream_error(exc: BaseException) -> str:
    """A short, URL-free description of ``exc`` for an error response."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f'HTTP {exc.response.status_code}'
    if isinstance(exc, httpx.TimeoutException):
        return f'{type(exc).__name__}: timed out'
    if isinstance(exc, httpx.HTTPError):
        return f'{type(exc).__name__}: request failed'
    return f'{type(exc).__name__}: {_URL.sub("<upstream>", str(exc))}'
