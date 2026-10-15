"""Serving the app's own files through Playwright's HTTP client."""

from __future__ import annotations

from typing import Any


class AppAssets:
    """What `serve_app_through_harness` served (URLs) and failed to fetch."""

    def __init__(self) -> None:
        self.served: list[str] = []
        self.errors: list[str] = []


def fetch_app_asset(route: Any, attempts: int = 3) -> Any:
    """`route.fetch()` for one of the app's own files, retrying a connection
    reset. `vite preview` closes idle keep-alive sockets; a request that
    reuses one in that instant fails with `read ECONNRESET`, which used to be
    aborted as a failed chunk and left the page without its first element.
    Only a reset is retried (these are idempotent GETs); any other error, and
    a reset that persists, propagates."""
    for attempt in range(attempts):
        try:
            # 3xx goes to the browser as-is, so a navigation lands on the
            # URL the server actually redirected to.
            return route.fetch(max_redirects=0)
        except Exception as exc:  # noqa: BLE001 — re-raised unless a retryable reset
            if "ECONNRESET" not in str(exc) or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def serve_app_through_harness(page: Any, app_url: str) -> AppAssets:
    """Answer every request for the app's own origin (HTML, JS chunks,
    CSS, static files) with `route.fetch()` + `route.fulfill()` instead of
    letting Chromium's network stack load it.

    Why: Chromium aborts every request still queued for a socket with
    `net::ERR_NETWORK_CHANGED` whenever the host's network configuration
    changes — including a docker container starting or stopping on this
    machine (its host-side veth gaining/losing an IPv6 address). The SPA
    boots by loading ~60 module chunks over 6 connections per origin, so a
    container event in that ~100 ms window fails a chunk import, SvelteKit
    renders "500 Internal Error", and the test times out on its first
    element. Reproduced deterministically by starting a short-lived
    container while requests are queued; see the CHANGELOG entry.
    `route.fetch()` runs in Playwright's own HTTP client, so a fulfilled
    route never touches Chromium's socket pool. The `{api_prefix}` stub,
    registered after this, still answers the API (later routes win).

    A fetch error (a down preview server, say) is recorded and the route
    aborted rather than left hanging.
    """
    assets = AppAssets()

    def handler(route: Any) -> None:
        try:
            response = fetch_app_asset(route)
            # Recorded before fulfilling: the browser's `requestfinished`
            # can be dispatched while `fulfill` is still waiting.
            assets.served.append(route.request.url)
            route.fulfill(response=response)
        except Exception as exc:  # noqa: BLE001 — must never leave the route unresolved
            assets.errors.append(f"{route.request.method} {route.request.url}: {exc!r}")
            try:
                route.abort("failed")
            except Exception:  # noqa: BLE001 — already resolved; nothing else to do
                pass

    page.route(f"{app_url.rstrip('/')}/**", handler)
    return assets
