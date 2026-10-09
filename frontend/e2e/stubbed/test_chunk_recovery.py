"""A failed module-chunk load recovers with one guarded reload, and never
loops (src/hooks.client.ts + src/lib/chunkRecovery.ts).

The app origin is served through Playwright's own client (`conftest.py`
`serve_app_through_harness`), so a later, more specific `page.route` for the
dashboard's page-node chunk can abort it and otherwise delegate to the
harness behaviour (`route.fallback()`).
"""

from __future__ import annotations

import re
from pathlib import Path

from conftest import ACTION_TIMEOUT_MS, ROOT
from test_labeling_flow import register_base

DASHBOARD = "/p/default/dashboard"


def dashboard_node_pattern() -> re.Pattern[str]:
    """The built chunk of the dashboard page node, from SvelteKit's generated
    route table (the node index is not stable across route changes)."""
    app_js = Path(ROOT) / ".svelte-kit" / "generated" / "client" / "app.js"
    match = re.search(r'"/p/\[project\]/dashboard":\s*\[(\d+)', app_js.read_text())
    assert match, "dashboard route not found in the generated client route table"
    return re.compile(rf"/_app/immutable/nodes/{match.group(1)}\.[^/]+\.js$")


def fail_chunk(page, pattern, times):
    """Abort the matching chunk request `times` times (None = always)."""
    state = {"aborted": 0}

    def handler(route):
        if pattern.search(route.request.url) and (times is None or state["aborted"] < times):
            state["aborted"] += 1
            route.abort("failed")
        else:
            route.fallback()

    page.route("**/_app/immutable/**", handler)
    return state


def count_document_loads(page, app_url):
    loads: list[str] = []
    page.on(
        "request",
        lambda r: loads.append(r.url)
        if r.is_navigation_request() and r.url.startswith(f"{app_url}{DASHBOARD}")
        else None,
    )
    return loads


def test_chunk_failure_recovers_after_one_reload(stub, page, app_url):
    register_base(stub)
    state = fail_chunk(page, dashboard_node_pattern(), times=1)
    loads = count_document_loads(page, app_url)

    page.goto(f"{app_url}{DASHBOARD}")
    page.locator("main").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    assert state["aborted"] == 1
    assert len(loads) == 2, f"expected exactly one automatic reload, saw {loads}"
    assert page.get_by_test_id("app-error").count() == 0


def test_repeated_chunk_failure_shows_error_page_without_looping(stub, page, app_url):
    register_base(stub)
    state = fail_chunk(page, dashboard_node_pattern(), times=None)
    loads = count_document_loads(page, app_url)

    page.goto(f"{app_url}{DASHBOARD}")
    error = page.get_by_test_id("app-error")
    error.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert error.get_by_role("button", name="Reload").is_visible()
    # The cause sits in a collapsed <details>, so read the text, not what is visible.
    detail = error.get_by_test_id("app-error-detail").text_content()
    assert "Failed to fetch dynamically imported module" in detail

    # No further reload may follow the one guarded attempt.
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert len(loads) == 2, f"expected one automatic reload then a stable error page: {loads}"
    assert state["aborted"] >= 2
