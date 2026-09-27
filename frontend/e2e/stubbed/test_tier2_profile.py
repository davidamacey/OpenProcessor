"""Ported from scripts/playwright_tier2_profile.py (deleted; see
docs/design/test-audit-2026-09-24.md recommendation 5 / P1-1).

Three passes against `/review`, stubbing `**/annotation-profiles.json`
directly (it is fetched at runtime by the root layout's `load()`, not
baked at build time — safe to intercept against a `vite preview` build):

  Pass 1 — no deployment profile (route answers 200/text-html, exactly
           nginx's/adapter-static's SPA fallback for a file that doesn't
           exist). Only the core tabs + the region tab render.
  Pass 2 — the shipped `static/annotation-profiles.example.json` served
           with `content-type: application/json`, against a backend whose
           served region profile is `pallet_label` (the example customizes
           that profile's region slot). A "Pallet labels" tab replaces the
           plain served one, is clickable, and drives a real
           `GET /curation/review/regions` request.
  Pass 3 — a malformed profile. The page still renders, the region tab still
           works, and a toast mentions the deployment annotation profile.
  Pass 4 — the same example against a backend serving a DIFFERENT region
           profile, or none: the region-profile rule drops it with a
           warning (the backend has exactly one region profile and 409s
           region routes without one).
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import re

from fixtures.wire import REGION_CLASS, REGION_TAB_LABEL

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PALLET_REGION_PROFILE = {
    "name": "pallet_label",
    "display_name": "Pallet labels",
    "display_name_singular": "Pallet label",
    "region_class_name": "pallet_label",
    "text_reader": "ocr",
}
EXAMPLE_PROFILE = json.loads((REPO_ROOT / "static" / "annotation-profiles.example.json").read_text())
MALFORMED_PROFILE = {"version": 1, "slots": [{"key": "bad"}]}

CLASSES = [
    {"id": 1, "name": REGION_CLASS, "group": "widgets", "hotkey_letter": "l", "count": 40, "validated_count": 12, "cluster_size": 44, "deprecated": False},
    {"id": 2, "name": "pallet_label", "group": "warehouse", "hotkey_letter": "w", "count": 20, "validated_count": 5, "cluster_size": 22, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}
EMPTY_QUEUE = {"total": 0, "page": 1, "page_size": 30, "items": [], "sort_fallback_reason": None}


def register_curation(stub):
    review_calls: list[str] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)

    def review_handler(_request, match):
        review_calls.append(match.string)
        return (200, EMPTY_QUEUE)

    stub.on("GET", r"/review/", review_handler)
    # The catch-all above would also answer /review/tabs; keep the served
    # region-tab label conftest.py defaults to.
    stub.on("GET", r"/review/tabs(\?|$)", {"tabs": [{"id": "regions", "label": REGION_TAB_LABEL}]})
    return review_calls


def register_profile_route(page, *, status: int, body: str | None, content_type: str):
    def handler(route, _request):
        if body is None:
            # Simulate the SPA fallback for a genuinely absent file: HTTP
            # 200, text/html, some index-ish body.
            route.fulfill(status=200, content_type="text/html", body="<!doctype html><html><body>spa fallback</body></html>")
            return
        route.fulfill(status=status, content_type=content_type, body=body)

    page.route("**/annotation-profiles.json", handler)


_TAB_COUNT = re.compile(r"\s+[\d,]+$")


def tab_labels(page) -> list[str]:
    # The active tab carries a count badge once its queue loads ("All 0"),
    # so match the tab with or without it and strip counts from the labels.
    page.get_by_role("button", name=re.compile(r"^All(\s+[\d,]+)?$")).wait_for(timeout=ACTION_TIMEOUT_MS)
    texts = page.locator("div.border-b.border-zinc-800 button").all_inner_texts()
    return [_TAB_COUNT.sub("", t.strip()) for t in texts]


def test_tier2_profile_absent(stub, page, app_url):
    register_curation(stub)
    register_profile_route(page, status=200, body=None, content_type="text/html")

    page.goto(f"{app_url}/review")
    labels = tab_labels(page)
    assert "Pallet labels" not in labels, f"no 'Pallet labels' tab expected: {labels}"
    assert REGION_TAB_LABEL in labels, f"the region tab should still be present: {labels}"
    assert not [c for c in stub.console_errors if c.startswith("pageerror")]
    annotation_msgs = [c for c in stub.console_errors if "annotation-profiles" in c.lower()]
    assert not annotation_msgs, f"absent profile should be silent: {annotation_msgs}"

    page.unroute("**/annotation-profiles.json")


def test_tier2_profile_served(stub, page, app_url):
    review_calls = register_curation(stub)
    stub.on("GET", r"/health$", {"status": "ok", "region_profile": PALLET_REGION_PROFILE})
    # No served label: the tab label comes from the tier-2 entry.
    stub.on("GET", r"/review/tabs(\?|$)", {"tabs": []})
    register_profile_route(page, status=200, body=json.dumps(EXAMPLE_PROFILE), content_type="application/json")

    page.goto(f"{app_url}/review")
    labels = tab_labels(page)
    assert "Pallet labels" in labels, f"'Pallet labels' tab should be present: {labels}"
    assert REGION_TAB_LABEL not in labels, f"the example replaces the served slot: {labels}"
    assert labels.count("Pallet labels") == 1, labels

    pallet_tab = page.get_by_role("button", name="Pallet labels")
    assert pallet_tab.count() > 0
    review_calls.clear()
    pallet_tab.first.click()
    page.wait_for_timeout(800)
    assert any(c.endswith("/review/regions") or "/review/regions?" in c for c in review_calls), (
        f"clicking it should drive GET /curation/review/regions: {review_calls}"
    )

    assert page.get_by_text("SSCC", exact=True).count() > 0, "the text-filter input's label should read 'SSCC'"

    assert not [c for c in stub.console_errors if c.startswith("pageerror")]

    page.unroute("**/annotation-profiles.json")


def test_tier2_profile_malformed(stub, page, app_url):
    register_curation(stub)
    register_profile_route(page, status=200, body=json.dumps(MALFORMED_PROFILE), content_type="application/json")

    page.goto(f"{app_url}/review")
    labels = tab_labels(page)
    assert REGION_TAB_LABEL in labels, f"the page should still render the core tabs + the region tab: {labels}"
    assert "Pallet labels" not in labels, f"no 'Pallet labels' tab from a malformed document: {labels}"

    region_tab = page.get_by_role("button", name=REGION_TAB_LABEL)
    assert region_tab.count() > 0, "the region tab should still be clickable"
    region_tab.first.click()
    page.wait_for_timeout(300)

    toast_text = page.locator("text=/annotation profile/i")
    assert toast_text.count() > 0, "a toast should mention the deployment annotation profile"

    warn_msgs = [c for c in stub.console_errors if "annotation-profiles" in c.lower()]
    assert any("bind must be an object" in m for m in warn_msgs), (
        f"the console should name the specific reason ('bind must be an object'): {warn_msgs}"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"a malformed doc must not crash the app: {errors[:3]}"


def _assert_dropped(stub, page, app_url, *, expect_region_tab: bool, reason: str) -> None:
    register_profile_route(page, status=200, body=json.dumps(EXAMPLE_PROFILE), content_type="application/json")
    page.goto(f"{app_url}/review")
    labels = tab_labels(page)
    assert "Pallet labels" not in labels, f"the example must be dropped: {labels}"
    assert (REGION_TAB_LABEL in labels) is expect_region_tab, labels
    page.locator("text=/annotation profile/i").first.wait_for(timeout=5000)
    warn_msgs = [c for c in stub.console_errors if "annotation-profiles" in c.lower()]
    assert any(reason in m and "dropped" in m for m in warn_msgs), warn_msgs
    assert not [c for c in stub.console_errors if c.startswith("pageerror")]
    page.unroute("**/annotation-profiles.json")


def test_tier2_region_slot_dropped_under_another_served_profile(stub, page, app_url):
    register_curation(stub)
    # conftest's default /health serves the widget_tag profile.
    _assert_dropped(stub, page, app_url, expect_region_tab=True, reason='"widget_tag"')


def test_tier2_region_slot_dropped_without_a_region_profile(stub, page, app_url):
    register_curation(stub)
    stub.on("GET", r"/health$", {"status": "ok", "region_profile": None})
    stub.on("GET", r"/review/tabs(\?|$)", {"tabs": []})
    _assert_dropped(stub, page, app_url, expect_region_tab=False, reason="no region profile")
    paths = [p for (_m, p) in stub.handled]
    assert not [p for p in paths if "/regions" in p], paths
