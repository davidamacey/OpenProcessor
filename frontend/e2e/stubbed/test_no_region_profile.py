"""Region features follow the served region profile (OpenProcessor
naming-w2; docs/design/domain-neutral-audit-2026-09-24.md §5.4, §7.3).

With `{API_PREFIX}/health` serving `region_profile: null`, every region
surface is absent (not disabled) and no request ever reaches a region
route — the backend would answer each with 409. With a profile, the
region tab is labelled by the served `display_name` even when
`/review/tabs` carries no label for it.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import re

from fixtures.wire import REGION_CLASS, REGION_PROFILE, make_item, review_tabs
from test_labeling_flow import CLASSES as ITEM_CLASSES
from test_labeling_flow import register_base

ROUTES = [
    "/p/default/dashboard",
    "/p/default/ingest",
    "/p/default/clusters",
    f"/p/default/clusters?class={REGION_CLASS}",
    "/p/default/clusters/1",
    "/p/default/review",
    "/p/default/review?tab=regions",
    "/p/default/classes",
    "/p/default/export",
    "/p/default/models",
    "/p/default/train",
    "/p/default/bakeoff",
    "/p/default/settings",
]

CORE_TABS = [
    "All",
    "Uncertainty",
    "Model Disagreements",
    "Classifier Blind Spots",
    "New Class Proposals",
]

# Any request to one of these is a region route (409 without a profile).
REGION_ROUTE = re.compile(r"/curation/(regions(/|$)|crops/[^/]+/region|ingest/region_drain)")

REGION_CLASS_ROW = {
    "class_id": 9,
    "class_name": REGION_CLASS,
    # No profile: the backend tags nothing as a region class.
    "kind": "item",
    "group": "widgets",
    "hotkey_letter": None,
    "sample_count": 40,
    "validated_count": 12,
    "cluster_size": 44,
    "deprecated": False,
}

_TAB_COUNT = re.compile(r"\s+[\d,]+$")


def tab_labels(page) -> list[str]:
    page.get_by_role("button", name=re.compile(r"^All(\s+[\d,]+)?$")).wait_for(timeout=ACTION_TIMEOUT_MS)
    texts = page.get_by_test_id("review-tabs").locator("button").all_inner_texts()
    return [_TAB_COUNT.sub("", t.strip()) for t in texts]


def register_no_profile(stub) -> None:
    register_base(stub)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": [*ITEM_CLASSES, REGION_CLASS_ROW]})
    stub.on("GET", r"/health$", {"status": "ok", "region_profile": None})
    # The backend omits the region tab without a profile.
    stub.on("GET", r"/review/tabs(\?|$)", review_tabs())
    stub.on("GET", r"/settings(\?|$)", {"settings": {}, "defaults": {}})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})
    stub.on("GET", r"/ingest/status(\?|$)", {"total": 0, "by_source": [], "by_day": []})

    # Every item carries region_* values on the wire (the item wire keeps
    # the keys whatever the profile), so a region surface that ignored the
    # gate would have something to render.
    def crops_handler(_request, _match):
        items = [
            make_item(
                crop_id=f"crop-{i}",
                image_id=f"img-{i}",
                class_id=1,
                class_name="ducati",
                cluster_id=1,
                label_validated=False,
                thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
            )
            for i in range(3)
        ]
        return (200, {"total": 3, "page": 1, "page_size": 60, "crops": items})

    stub.on("GET", r"/crops(\?|$)", crops_handler)


def region_requests(stub) -> list[tuple[str, str]]:
    return [(m, p) for (m, p) in stub.handled + stub.unhandled if REGION_ROUTE.search(p)]


def test_every_route_mounts_without_region_calls(stub, page, app_url):
    register_no_profile(stub)
    for route in ROUTES:
        stub.console_errors.clear()
        page.goto(f"{app_url}{route}")
        page.locator("main").first.wait_for(timeout=ACTION_TIMEOUT_MS)
        # Real wait for "the page is done firing its on-mount requests"
        # (the whole point of the following negative assertion) instead of
        # an arbitrary settle sleep.
        page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
        crashed = [c for c in stub.console_errors if c.startswith("pageerror") or "Uncaught" in c]
        assert not crashed, f"{route} should mount without errors: {crashed[:2]}"
        assert not region_requests(stub), f"{route} called a region route: {region_requests(stub)}"


def test_region_surfaces_absent(stub, page, app_url):
    register_no_profile(stub)

    # /review: exactly the core tabs; a ?tab=regions bookmark opens All.
    page.goto(f"{app_url}/p/default/review?tab=regions")
    assert tab_labels(page) == CORE_TABS
    # ...and says why, instead of switching to All silently.
    notice = page.get_by_test_id("tab-unavailable")
    notice.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "no region profile" in notice.inner_text()

    # /dashboard: no region detections panel.
    page.goto(f"{app_url}/p/default/dashboard")
    page.locator("main").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("verifier-confirmed").count() == 0

    # /clusters?class=<region class>: the normal class-filtered grid, not
    # a region gallery.
    page.goto(f"{app_url}/p/default/clusters?class={REGION_CLASS}")
    page.get_by_test_id("class-filter-chip").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_role("button", name=re.compile("⟳")).count() == 0

    # /clusters/[id]: crop cards offer no sub-box editor.
    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert page.locator('button[aria-label^="Edit "]').count() == 0
    assert page.get_by_text("✎").count() == 0

    # /ingest: no region drain panel.
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    # The next line's own real wait (a concrete selector) supersedes the
    # settle sleep that used to sit here.
    page.get_by_text("Ingest status").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("Region detection worklog").count() == 0

    assert not region_requests(stub), region_requests(stub)
    assert not [c for c in stub.console_errors if c.startswith("pageerror")]


def test_served_display_name_labels_the_region_tab(stub, page, app_url):
    register_base(stub)
    # No served label for the region tab: the label must come from the
    # profile's display_name alone.
    stub.on("GET", r"/review/tabs(\?|$)", review_tabs())
    stub.on("GET", r"/regions/statuses(\?|$)", {"statuses": []})
    page.goto(f"{app_url}/p/default/review")
    labels = tab_labels(page)
    assert labels == [*CORE_TABS, REGION_PROFILE["display_name"]], labels
    # With a profile, the root layout loads the region vocabularies.
    paths = [p for (_m, p) in stub.handled]
    assert any(p.endswith("/regions/vocabulary") for p in paths), paths
    assert any(p.endswith("/regions/statuses") for p in paths), paths
