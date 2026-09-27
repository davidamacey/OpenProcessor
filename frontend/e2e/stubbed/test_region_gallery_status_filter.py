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
    page.wait_for_timeout(300)

    option_values = select.locator("option").evaluate_all("opts => opts.map(o => o.value)")
    option_labels = select.locator("option").evaluate_all(
        "opts => opts.map(o => o.textContent.trim())"
    )
    assert option_values == ["", "detected", "verify_rejected"], option_values
    assert option_labels == ["any", "Detected", "Rejected (candidate kept)"], option_labels

    region_list_calls.clear()
    select.select_option("verify_rejected")
    page.wait_for_timeout(500)

    assert region_list_calls, "picking a status must trigger a fresh GET {API_PREFIX}/regions call"
    assert any("status=verify_rejected" in url for url in region_list_calls), region_list_calls

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the region status-filter flow: {errors[:3]}"
