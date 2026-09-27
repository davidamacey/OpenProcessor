"""dq-queues cutover (2026-09-24): `GET {API_PREFIX}/review/tabs` now serves
each tab's `filters`/`filter_defaults`. `/review`'s filter bar renders
only the controls the active tab's served `filters` list names, and the
subject/max_rank toggle's "unset" label reflects the served default
instead of a hardcoded "Top 2". This proves the real browser-rendered
filter bar reflects a served, per-tab list this test controls — not a
hardcoded always-visible set.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

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

# `all`: a restricted filters list — no max_rank, no min_blur_ratio, no
# conf_min/conf_max. `primary_low_conf`: the full list, with a served
# max_rank default of 2 (so its subject-toggle "unset" label should read
# "Top 2").
REVIEW_TABS = {
    "tabs": [
        {
            "id": "all",
            "label": "All",
            "filters": ["class_id", "source"],
            "filter_defaults": {},
        },
        {
            "id": "primary_low_conf",
            "label": "Primary low-conf",
            "filters": ["class_id", "source", "max_rank", "min_blur_ratio", "conf_min", "conf_max"],
            "filter_defaults": {"max_rank": 2},
        },
    ]
}


def review_item(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="ducati",
    )


def _stub_common(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/tabs(\?|$)", REVIEW_TABS)
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        lambda req, m: (200, {"image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480}, "items": [review_item(0)]}),
    )


def test_review_filter_bar_hides_controls_the_active_tab_does_not_serve(stub, page, app_url):
    def review_handler(_request, _match):
        items = [review_item(i) for i in range(2)]
        return (200, {"items": items, "total": 2, "page": 1, "page_size": 30})

    # Registered BEFORE _stub_common so /review/tabs (a more specific,
    # later-registered pattern) wins over this broader /review/ regex —
    # dispatch checks the most-recently-registered handler first, and
    # r"/review/" also matches the /review/tabs path.
    stub.on("GET", r"/review/", review_handler)
    _stub_common(stub)

    page.goto(f"{app_url}/p/default/review?tab=all")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(800)

    # `all`'s served filters list has no max_rank/min_blur_ratio/conf —
    # those controls must not render.
    assert page.locator('div:has(> span:text-is("subject"))').count() == 0
    assert page.locator('label[title*="Hide crops blurrier"]').count() == 0
    assert page.locator('label:has-text("Conf")').count() == 0
    # class_id/source ARE served — those controls must render.
    assert page.locator('label:has-text("Class")').count() >= 1
    assert page.locator('label:has-text("Source")').count() >= 1


def test_review_subject_toggle_label_reflects_served_max_rank_default(stub, page, app_url):
    def review_handler(_request, _match):
        items = [review_item(i) for i in range(2)]
        return (200, {"items": items, "total": 2, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)
    _stub_common(stub)

    page.goto(f"{app_url}/p/default/review?tab=all&preset=primary_low_conf")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(800)

    subject = page.locator('div:has(> span:text-is("subject"))')
    subject.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "Top 2" in subject.inner_text()
