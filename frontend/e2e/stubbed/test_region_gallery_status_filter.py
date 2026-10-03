"""dq-region (2026-09-24): the /clusters region gallery gains a Status
filter (SlotGallery.svelte), backed by GET {API_PREFIX}/regions?status=
(400 server-side on an unknown value). Options are the served
GET {API_PREFIX}/regions/statuses vocabulary, not a hardcoded list —
mirrors the Detector filter's served-vocabulary pattern
(test_region_gallery_detector_filter.py). Picking a status must forward
it as ?status= on the next GET {API_PREFIX}/regions call — proves the
real browser-rendered <select> actually drives the query, not just
that it renders.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import REGION_CLASS
from playwright.sync_api import expect

CLASSES = [
    {
        "class_id": 1,
        "class_name": REGION_CLASS,
        "kind": "region",
        "group": "widgets",
        "hotkey_letter": "l",
        "sample_count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
]

STATUSES = {
    "statuses": [
        {
            "value": "detected",
            "label": "Detected",
            "role": "proposed",
            "terminal": False,
            "human_writable": True,
            "clears_box": False,
            "wants_reason": False,
        },
        {
            "value": "verify_rejected",
            "label": "Rejected (candidate kept)",
            "role": "rejected",
            "terminal": False,
            "human_writable": True,
            "clears_box": False,
            "wants_reason": False,
        },
    ],
    "confirm_status": "detected",
    "reject_status": "no_region_visible",
    "false_positive_status": "false_positive",
}


def test_region_status_filter_lists_served_statuses_and_forwards_the_query_param(
    stub, page, app_url
):
    region_list_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/regions/statuses(\?|$)", STATUSES)
    stub.on("GET", r"/regions/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/clusters(\?|$)", {"clusters": [], "count": 0})

    def regions_handler(request, match):
        region_list_calls.append(request.url)
        return (200, {"items": [], "total": 0})

    stub.on("GET", r"/regions(\?|$)", regions_handler)

    page.goto(f"{app_url}/p/default/clusters?class={REGION_CLASS}")

    select = page.locator('label:has-text("Status") select')
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    # Wait for the served vocabulary's options to actually populate the
    # <select> instead of an arbitrary settle sleep.
    expect(select.locator("option")).to_have_count(3, timeout=ACTION_TIMEOUT_MS)

    option_values = select.locator("option").evaluate_all("opts => opts.map(o => o.value)")
    option_labels = select.locator("option").evaluate_all(
        "opts => opts.map(o => o.textContent.trim())"
    )
    assert option_values == ["", "detected", "verify_rejected"], option_values
    assert option_labels == ["any", "Detected", "Rejected (candidate kept)"], option_labels

    region_list_calls.clear()
    with page.expect_response(
        lambda r: r.request.method == "GET" and "/regions" in r.url and "status=verify_rejected" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option("verify_rejected")

    assert region_list_calls, "picking a status must trigger a fresh GET {API_PREFIX}/regions call"
    assert any("status=verify_rejected" in url for url in region_list_calls), region_list_calls

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the region status-filter flow: {errors[:3]}"


def test_gallery_count_does_not_divide_boxes_by_items(stub, page, app_url):
    """A multi-box item is several rows. `/regions` serves `total` in items and
    `total_rows` in boxes; the count must not read 'rows listed / items total'
    (it showed '141 / 135 listed' on a project with 135 items and 292 boxes)."""
    from fixtures.wire import make_box, make_item

    boxes = [make_box(f"b{n}") for n in (1, 2, 3)]
    item = make_item(crop_id="multi-1", image_id="img-1", region_boxes=boxes)
    rows = [
        {**item, "region_box_id": b["box_id"], "row_key": f"multi-1#{b['box_id']}"} for b in boxes
    ]
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/regions/statuses(\?|$)", STATUSES)
    stub.on("GET", r"/regions/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/regions(\?|$)", {"items": rows, "total": 1, "total_rows": 3})

    page.goto(f"{app_url}/p/default/clusters?class={REGION_CLASS}")
    count = page.get_by_test_id("slot-gallery-count")
    count.wait_for(timeout=ACTION_TIMEOUT_MS)
    text = " ".join(count.inner_text().split())
    assert "3 / 3 boxes listed" in text, text
    assert "1 item" in text, text
    assert "3 / 1" not in text, text
