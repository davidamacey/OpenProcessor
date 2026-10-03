"""Issue #24 polish: a served enum filter never renders a blank default
option, and the review panel's Reason row stays a label/value row at 800px.
"""

from __future__ import annotations

import os

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item, review_tab, review_tabs

TABS = review_tabs(
    {
        **review_tab("all", "All", filters=["class_id", "source"]),
        "filter_specs": [
            {
                "param": "on_negative_frame",
                "kind": "enum",
                "label": "Negative frames",
                "options": [
                    {"value": "include", "label": "Include negative frames"},
                    {"value": "only", "label": "Only negative frames"},
                ],
            }
        ],
    }
)


def _setup(stub):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/review/tabs(\?|$)", TABS)
    item = make_item(crop_id="crop-0", image_id="img-0", class_id=1, class_name="ducati")
    item["reason"] = "model and label disagree on this crop"
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        (200, {"items": [item], "total": 1, "page": 1, "page_size": 30}),
    )
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        (200, {"image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480}, "items": [item]}),
    )


def _shot(page, name):
    out = os.environ.get("UI_POLISH_SHOTS")
    if out:
        os.makedirs(out, exist_ok=True)
        page.screenshot(path=f"{out}/{name}.png", full_page=True)


def test_negative_frames_select_shows_a_served_label_by_default(stub, page, app_url):
    _setup(stub)
    page.goto(f"{app_url}/p/default/review?tab=all")
    select = page.locator('label:has-text("Negative frames") select')
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    shown = select.evaluate("el => el.selectedOptions[0]?.textContent?.trim() ?? ''")
    assert shown == "Include negative frames", shown


def test_reason_row_is_a_label_value_row_at_800px(stub, page, app_url):
    _setup(stub)
    for width in (1600, 800):
        page.set_viewport_size({"width": width, "height": 1000})
        page.goto(f"{app_url}/p/default/review?tab=all")
        dt = page.locator('dt:text-is("Reason")')
        dt.wait_for(timeout=ACTION_TIMEOUT_MS)
        dd = page.locator('dt:text-is("Reason") + dd')
        a, b = dt.bounding_box(), dd.bounding_box()
        assert b["x"] >= a["x"] + a["width"] - 1, (width, a, b)
        # dt is a grid cell (as wide as its column), so compare left edges: the value
        # column must start right after the widest label, not at the half-way point.
        assert b["x"] - a["x"] < 160, (width, a, b)
        assert abs(b["y"] - a["y"]) < 6, (width, a, b)
        _shot(page, f"review-{width}")
