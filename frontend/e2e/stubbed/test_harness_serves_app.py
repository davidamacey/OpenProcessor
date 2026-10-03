"""The harness, not Chromium's network stack, loads the app's own files.

Guards the fix for the intermittent "first element never appears" flake:
Chromium fails every request still queued for a socket with
`net::ERR_NETWORK_CHANGED` when the host's network changes (a docker
container starting or stopping on this machine is enough). The SPA's ~60
boot chunks queue behind 6 connections, so one such event failed a chunk
import and left the page at SvelteKit's "500 Internal Error". The `stub`
fixture now fulfills every app-origin request through `route.fetch()`
(`serve_app_through_harness`, e2e/conftest.py), which never touches the
browser's socket pool. If that routing is removed or its pattern stops
matching, this test fails.
"""

from __future__ import annotations

from urllib.parse import urlparse

from conftest import ACTION_TIMEOUT_MS, API_PREFIX, fetch_app_asset


def test_every_app_origin_request_is_fulfilled_by_the_harness(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/models/status(\?|$)", {"models": []})

    origin = urlparse(app_url).netloc
    app_requests: list[str] = []

    def record(request) -> None:
        url = urlparse(request.url)
        if url.netloc == origin and not url.path.startswith(f"{API_PREFIX}/"):
            app_requests.append(request.url)

    page.on("requestfinished", record)
    page.goto(f"{app_url}/p/default/models")
    page.get_by_role("heading", name="Models").wait_for(timeout=ACTION_TIMEOUT_MS)

    served = stub.app_assets.served
    assert stub.app_assets.errors == []
    assert f"{app_url}/p/default/models" in served
    chunks = [u for u in served if "/_app/immutable/" in u and u.endswith(".js")]
    assert len(chunks) >= 10, served
    missed = [u for u in app_requests if u not in served]
    assert missed == [], f"app-origin requests loaded by the browser itself: {missed}"


class _FlakyRoute:
    """Stands in for a Playwright route whose first `fail` fetches raise."""

    def __init__(self, error: str, fail: int) -> None:
        self.error, self.fail, self.calls = error, fail, 0

    def fetch(self, **_kw):
        self.calls += 1
        if self.calls <= self.fail:
            raise Exception(self.error)
        return "response"


def test_a_connection_reset_on_an_app_file_is_retried():
    """CI flake: `vite preview` closed an idle keep-alive socket and one boot
    chunk failed with `read ECONNRESET`, aborting the page's first load."""
    route = _FlakyRoute("Route.fetch: read ECONNRESET", fail=2)
    assert fetch_app_asset(route) == "response"
    assert route.calls == 3


def test_a_persistent_reset_and_other_errors_still_surface():
    import pytest

    stuck = _FlakyRoute("Route.fetch: read ECONNRESET", fail=99)
    with pytest.raises(Exception, match="ECONNRESET"):
        fetch_app_asset(stuck)
    assert stuck.calls == 3

    refused = _FlakyRoute("Route.fetch: connect ECONNREFUSED", fail=99)
    with pytest.raises(Exception, match="ECONNREFUSED"):
        fetch_app_asset(refused)
    assert refused.calls == 1
