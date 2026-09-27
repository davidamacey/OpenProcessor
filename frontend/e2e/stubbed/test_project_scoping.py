"""P1 projects cutover: proves every request actually lands on the
RIGHT prefix, not just that some suffix pattern matched.

Every other stubbed test's `.on(...)` patterns match by path SUFFIX
(`r"/health$"` matches both the global `/curation/health` and the
scoped `/curation/projects/default/health`), so none of them can catch
a call built from the wrong builder (e.g. `${globalApi()}/review/tabs`
instead of `${scoped()}/review/tabs`) — it would still 200. This test
inspects `stub.handled`'s exact recorded paths instead of relying on a
handler having matched.

Mutation check (recorded here, not re-run every session): temporarily
making `scoped()` return `API_PREFIX` unconditionally in `src/lib/
api.ts` turns this test red (every "hits the scoped prefix" assertion
fails, since every scoped call then lands on the bare global prefix
instead) — confirmed 2026-09-26, reverted byte-for-byte after.
"""

from __future__ import annotations

API_PREFIX = "/curation"
DEFAULT_PROJECT_PREFIX = f"{API_PREFIX}/projects/default"

# The only two exact paths a GLOBAL call may hit (P1: `/projects`,
# `/health`, `/events` — no other route stays global).
GLOBAL_EXACT_PATHS = {
    f"{API_PREFIX}/projects",
    f"{API_PREFIX}/health",
    f"{API_PREFIX}/events",
}


def test_review_mount_hits_exactly_the_global_or_scoped_prefix(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": [], "thresholds": {}, "reserved_hotkeys": []})
    stub.on("GET", r"/review/all(\?|$)", {"items": [], "total": 0, "page": 1, "page_size": 50})
    page.goto(f"{app_url}/review")
    page.wait_for_timeout(1200)

    paths = [p for _, p in stub.handled]
    assert paths, "no requests were handled — the mount didn't fire anything"

    # (a) the project list and the status chip's health hit the GLOBAL
    # routes, not a project-scoped path.
    assert f"{API_PREFIX}/projects" in paths, (
        f"the root layout's project-list bootstrap never hit the global "
        f"/projects route: {paths}"
    )
    assert f"{API_PREFIX}/health" in paths, (
        f"the status chip never hit the global /health route: {paths}"
    )

    # (b) region-profile health, /review/*, /classes and every other
    # project fact hit the scoped prefix.
    assert f"{DEFAULT_PROJECT_PREFIX}/health" in paths, (
        f"regionProfileStore's scoped health read never fired: {paths}"
    )
    assert any(p.startswith(f"{DEFAULT_PROJECT_PREFIX}/review") for p in paths), (
        f"no scoped /review/* request: {paths}"
    )
    assert any(p.startswith(f"{DEFAULT_PROJECT_PREFIX}/classes") for p in paths), (
        f"no scoped /classes request: {paths}"
    )

    # (c) NO request hits an unscoped `/curation/<scoped-route>` — every
    # path is either one of the three global exact routes above, or
    # lives under the scoped prefix. A bug that built a scoped call from
    # `API_PREFIX`/`globalApi()` instead of `scoped()` would show up here
    # as a path that is neither.
    leaked = [
        p
        for p in paths
        if p not in GLOBAL_EXACT_PATHS and not p.startswith(DEFAULT_PROJECT_PREFIX)
    ]
    assert leaked == [], f"unscoped route(s) leaked past scoped(): {leaked}"
