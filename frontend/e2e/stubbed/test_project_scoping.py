"""Projects: the active project lives in the URL path (`/p/<slug>/...`)
and every request lands on the RIGHT prefix, not just a suffix that
happened to match.

Every other stubbed test's `.on(...)` patterns match by path SUFFIX
(`r"/health$"` matches both the global `/curation/health` and the
scoped `/curation/projects/default/health`), so none of them can catch
a call built from the wrong builder. These tests inspect
`stub.handled`'s exact recorded paths instead.

Waits are real (`expect_request` / `wait_for_url` / a visible element),
never a fixed sleep.

Mutation check (recorded here, not re-run every session): temporarily
making `scoped()` return `API_PREFIX` unconditionally in `src/lib/
api.ts` turns the first test red (every scoped call then lands on the
bare global prefix) — confirmed 2026-09-26, reverted byte-for-byte after.
"""

from __future__ import annotations

import re

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import project, projects_response

API_PREFIX = "/curation"
DEFAULT_PROJECT_PREFIX = f"{API_PREFIX}/projects/default"

# The only exact paths a GLOBAL call may hit: the project list, the
# global health and the global event stream.
GLOBAL_EXACT_PATHS = {
    f"{API_PREFIX}/projects",
    f"{API_PREFIX}/health",
    f"{API_PREFIX}/events",
}

CLASSES = {"classes": [], "thresholds": {}, "reserved_hotkeys": []}
EMPTY_QUEUE = {"items": [], "total": 0, "page": 1, "page_size": 50}


def _path(url: str) -> str:
    return re.sub(r"^https?://[^/]+", "", url).split("?")[0]


def test_review_mount_hits_exactly_the_global_or_scoped_prefix(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", CLASSES)
    stub.on("GET", r"/review/all(\?|$)", EMPTY_QUEUE)

    with (
        page.expect_request(lambda r: _path(r.url) == f"{DEFAULT_PROJECT_PREFIX}/review/all"),
        page.expect_request(lambda r: _path(r.url) == f"{DEFAULT_PROJECT_PREFIX}/classes"),
        page.expect_request(lambda r: _path(r.url) == f"{API_PREFIX}/health"),
    ):
        page.goto(f"{app_url}/p/default/review")
    page.get_by_test_id("queue-empty").wait_for(timeout=ACTION_TIMEOUT_MS)

    paths = [p for _, p in stub.handled]
    # (a) the project list and the status chip's health are GLOBAL.
    assert f"{API_PREFIX}/projects" in paths, paths
    assert f"{API_PREFIX}/health" in paths, paths
    # (b) the region-profile health and every project fact are scoped.
    assert f"{DEFAULT_PROJECT_PREFIX}/health" in paths, paths
    # (c) nothing leaks onto an unscoped `/curation/<scoped-route>`.
    leaked = [
        p for p in paths if p not in GLOBAL_EXACT_PATHS and not p.startswith(DEFAULT_PROJECT_PREFIX)
    ]
    assert leaked == [], f"unscoped route(s) leaked past scoped(): {leaked}"


def test_root_and_legacy_paths_redirect_under_the_served_default(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", CLASSES)
    stub.on("GET", r"/review/uncertainty(\?|$)", EMPTY_QUEUE)
    # The served default is NOT hardcoded: this deployment's is "main".
    stub.on(
        "GET",
        rf"^{re.escape(API_PREFIX)}/projects$",
        {
            **projects_response(API_PREFIX, [project(API_PREFIX, "main", is_default=True)]),
            "default_slug": "main",
        },
    )

    page.goto(f"{app_url}/review?tab=uncertainty")
    page.wait_for_url(re.compile(r"/p/main/review\?tab=uncertainty$"), timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("queue-empty").wait_for(timeout=ACTION_TIMEOUT_MS)

    page.goto(f"{app_url}/")
    page.wait_for_url(re.compile(r"/p/main/dashboard$"), timeout=ACTION_TIMEOUT_MS)


def test_unknown_and_unavailable_slugs_show_a_way_out(stub, page, app_url):
    stub.on(
        "GET",
        rf"^{re.escape(API_PREFIX)}/projects$",
        projects_response(
            API_PREFIX,
            [
                project(API_PREFIX, "default", is_default=True, deletable=False),
                project(API_PREFIX, "wip", status="building", selectable=False, writable=False),
            ],
        ),
    )
    stub.on(
        "GET",
        rf"^{re.escape(API_PREFIX)}/projects/ghost$",
        (404, {"detail": {"error": "project_not_found", "message": "no project named 'ghost'"}}),
    )

    page.goto(f"{app_url}/p/ghost/review")
    gone = page.get_by_test_id("project-unavailable")
    gone.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "Project not found" in gone.inner_text()
    assert gone.get_by_role("link", name="All projects").get_attribute("href") == "/projects"

    page.goto(f"{app_url}/p/wip/dashboard")
    page.get_by_test_id("project-unavailable-status").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-unavailable-status").inner_text() == "building"

    # Neither page fired a single scoped request.
    scoped = [p for _, p in stub.handled if p.startswith(f"{API_PREFIX}/projects/") and p.count("/") > 3]
    assert scoped == [], scoped
