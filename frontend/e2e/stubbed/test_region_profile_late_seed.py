"""F-78 (coordinator, 2026-09-25): a slow or failed first `/health` used to
seed "no region profile"; the next poll disagreed, a spurious "region
profile changed — reload" toast fired, and the region tab was missing.

Now a failed boot read leaves the profile unknown; the first successful
`/health` poll seeds it and the region tab appears without a reload, with
no "changed" toast.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import REGION_PROFILE, REGION_TAB_LABEL
from test_labeling_flow import register_base


def test_failed_boot_health_then_success_shows_region_tab_without_toast(stub, page, app_url):
    register_base(stub)
    stub.on("GET", r"/review/tabs(\?|$)", {"tabs": []})
    stub.on(
        "GET",
        r"/review/",
        lambda _r, _m: (200, {"items": [], "total": 0, "page": 1, "page_size": 30}),
    )
    calls = {"n": 0}

    def health(_request, _match):
        calls["n"] += 1
        # The three boot tries fail; the layout's first /health poll works.
        if calls["n"] <= 3:
            return (400, {"detail": "stub: health unavailable"})
        return (200, {"status": "ok", "region_profile": REGION_PROFILE})

    stub.on("GET", r"/health$", health)

    page.goto(f"{app_url}/review")
    page.get_by_test_id("review-tabs").get_by_role("button", name=REGION_TAB_LABEL).wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    assert calls["n"] >= 4, calls
    body = page.locator("body").inner_text()
    assert "region profile changed" not in body, "a late seed must not raise the reload toast"
    assert page.get_by_test_id("region-profile-loading").count() == 0
