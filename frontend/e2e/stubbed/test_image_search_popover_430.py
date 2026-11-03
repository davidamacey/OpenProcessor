"""The search-by-image popover opens under its own button on a 430 px screen (not pinned
to the top of the viewport over the header) and stays inside the viewport. Layout needs a
real browser; jsdom has no geometry."""

from __future__ import annotations

import os

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_cluster_list_filters import clusters_handler

METHODS = {
    "strategies": [
        {"id": "semantic_search", "axis": "overlay", "label": "Semantic search", "status": "stable"}
    ],
    "flags": {},
}
# Optional: set to a directory to keep screenshots of the open popover for review.
SHOTS = os.environ.get("OP_E2E_SHOT_DIR")


def _open(stub, page, app_url, width, height):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/regions(\?|$)", {"items": [], "total": 0})
    stub.on("GET", r"/clusters(\?|$)", clusters_handler([]))
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"{app_url}/p/default/clusters")
    toggle = page.get_by_test_id("image-search-toggle")
    expect(toggle).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    panel = page.get_by_test_id("image-search-panel")
    expect(panel).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    return toggle.bounding_box(), panel.bounding_box()


def _shot(page, name):
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/{name}.png")


def test_popover_is_anchored_under_its_button_at_430(stub, page, app_url):
    button, panel = _open(stub, page, app_url, 430, 900)
    _shot(page, "image-search-430")
    assert abs(panel["y"] - (button["y"] + button["height"])) < 16, (button, panel)
    assert panel["x"] >= 0 and panel["x"] + panel["width"] <= 430 + 0.5, panel


def test_popover_is_anchored_under_its_button_at_1440(stub, page, app_url):
    button, panel = _open(stub, page, app_url, 1440, 900)
    _shot(page, "image-search-1440")
    assert abs(panel["y"] - (button["y"] + button["height"])) < 16, (button, panel)
    assert abs(panel["x"] - button["x"]) < 2, (button, panel)
