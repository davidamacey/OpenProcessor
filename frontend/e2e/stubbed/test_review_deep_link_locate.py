"""DQ-M7 (docs/design/data-quality-pass-2026-09-24.md): `/review?crop_id=`
deep link at a deep page used to page 1..loc.page sequentially via
`queue.loadMore()` — 103 requests / 5.7s to reach rank 3000 (page 101) —
and for that whole window rendered page 1's item #1 with its keybindings
live, so a keypress acted on the wrong crop.

Fixed by `queue.loadPage(loc.page)` (one request, src/lib/pager.svelte.ts)
and a new `awaitingDeepLink` state that keeps the tab-action keybindings
unregistered and renders a "Locating crop…" placeholder instead of item #1
until the target page lands (src/routes/review/+page.svelte).

This test proves both halves against a real page mount:
  1. only ONE `/review/all` page request is made before the located page's
     request (page 1's own load, made before the deep link — never every
     page from 2..101);
  2. a keypress fired while the locate request is artificially held open
     does not reach crop-0 (item #1) — no /label PUT for it.
"""

from __future__ import annotations

import re
import threading
from urllib.parse import parse_qs, urlparse

from fixtures.wire import make_item

CLASSES = [
    {
        "id": 1,
        "name": "ducati",
        "group": "moto",
        "hotkey_letter": "k",
        "count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
]

METHODS = {"strategies": [], "flags": {}}


def review_item(i: int) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="ducati",
        proposed_class_id=1,
        proposed_class_name="ducati",
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )
    item["reason"] = "uncertainty"
    return item


def test_deep_link_fetches_only_the_located_page_and_ignores_early_keys(stub, page, app_url):
    review_page_requests: list[int] = []
    label_calls: list[str] = []
    locate_gate = threading.Event()

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        lambda req, m: (
            200,
            {
                "image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480},
                "items": [review_item(0)],
            },
        ),
    )

    def review_handler(request, _match):
        qparams = parse_qs(urlparse(request.url).query)
        page_num = int(qparams.get("page", ["1"])[0])
        review_page_requests.append(page_num)
        # Page 1 serves item #1 (crop-0) so the "shows item #1 while
        # locating" regression is provable. Page 101 (the located page)
        # serves the actual target, crop-target.
        if page_num == 101:
            item = review_item(999)
            item["id"] = "crop-target"
            items = [item]
        else:
            items = [review_item(0)]
        return (200, {"items": items, "total": 3010, "page": page_num, "page_size": 30})

    stub.on("GET", r"/review/all(\?|$)", review_handler)

    def locate_handler(_request, _match):
        # Held open until the test explicitly releases it, simulating the
        # locate round trip's real latency — this is the window during
        # which the regression showed item #1 with live keys. Playwright's
        # sync API dispatches route handlers on their own thread, so
        # blocking here does not freeze the test's own page.* calls.
        locate_gate.wait(timeout=10)
        return (
            200,
            {
                "crop_id": "crop-target",
                "in_queue": True,
                "rank": 3000,
                "page": 101,
                "page_size": 30,
                "total": 3010,
                "reason": None,
                "sort_applied": "atypicality",
                "sort_fallback_reason": None,
            },
        )

    stub.on("GET", r"/review/all/locate(\?|$)", locate_handler)

    def label_handler(request, match):
        label_calls.append(match.string)
        item = review_item(0)
        item["class_id"] = (request.post_data_json or {}).get("class_id")
        return (200, item)

    stub.on("PUT", r"/crops/([^/]+)/label$", label_handler)

    page.goto(f"{app_url}/review?tab=all&crop_id=crop-target")

    # Give page-1's own load (fired before the locate resolves) time to
    # land, then fire a keypress while the locate is still gated open —
    # this is exactly the window the bug fired a PUT for crop-0 in.
    page.wait_for_timeout(600)
    assert "Locating crop" in page.locator("body").inner_text()
    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    assert label_calls == [], (
        f"a keypress during the locate window must not act on item #1: {label_calls}"
    )

    locate_gate.set()
    page.wait_for_timeout(800)

    # Exactly the initial page-1 load plus the located page-101 request —
    # never anything for pages 2..100.
    assert review_page_requests == [1, 101], review_page_requests

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
