"""Issue #24 polish: a served enum filter with no served default reads "any"
(never a value the queue request does not send, never a blank), and the
review panel's Reason row stays a label/value row at 800px.
"""

from __future__ import annotations

import os

from conftest import ACTION_TIMEOUT_MS, expect_handled

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
                    {"value": "true", "label": "Only items on a reviewed-negative frame"},
                    {"value": "false", "label": "No items on a negative frame"},
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


def test_negative_frames_select_reads_any_and_sends_nothing_until_picked(stub, page, app_url):
    _setup(stub)
    queue_urls: list[str] = []
    page.on("request", lambda r: queue_urls.append(r.url) if "/review/all" in r.url else None)
    page.goto(f"{app_url}/p/default/review?tab=all")
    select = page.locator('label:has-text("Negative frames") select')
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.locator('dt:text-is("Reason")').wait_for(timeout=ACTION_TIMEOUT_MS)
    shown = select.evaluate("el => el.selectedOptions[0]?.textContent?.trim() ?? ''")
    assert shown == "any", shown
    assert queue_urls and all("on_negative_frame" not in u for u in queue_urls), queue_urls

    with expect_handled(
        page,
        lambda r: "/review/all" in r.url and "on_negative_frame=true" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        select.select_option("true")
    shown = select.evaluate("el => el.selectedOptions[0]?.textContent?.trim() ?? ''")
    assert shown == "Only items on a reviewed-negative frame", shown


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
