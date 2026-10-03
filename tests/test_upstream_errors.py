"""Upstream failure wording never carries the internal service URL."""

from __future__ import annotations

import httpx

from src.utils.upstream_errors import describe_upstream_error


URL = 'http://seg.internal:8000/segment'


def test_a_status_error_is_the_status_alone() -> None:
    request = httpx.Request('POST', URL)
    exc = httpx.HTTPStatusError(
        f"Server error '503' for url '{URL}'",
        request=request,
        response=httpx.Response(503, request=request),
    )
    assert describe_upstream_error(exc) == 'HTTP 503'


def test_transport_errors_are_named_without_their_message() -> None:
    assert describe_upstream_error(httpx.ConnectError(f'refused {URL}')) == (
        'ConnectError: request failed'
    )
    assert describe_upstream_error(httpx.ReadTimeout(f'slow {URL}')) == ('ReadTimeout: timed out')


def test_any_other_error_has_urls_redacted() -> None:
    assert describe_upstream_error(RuntimeError(f'bad reply from {URL} today')) == (
        'RuntimeError: bad reply from <upstream> today'
    )
