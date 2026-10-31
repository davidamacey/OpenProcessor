"""#405: a thumbnail the backend answers with the typed `source_missing` detail
renders a readable "Source not mounted" badge with the served alias, on a
crop card and in the review pane, instead of only a dimmed tile.

Set OP_E2E_SHOT_DIR to also save full-page screenshots of both states.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_cluster_discard import CLASSES, CLUSTERS, crop
from test_review_crop_viewport import review_item

MESSAGE = "source root 'archive' is not mounted or reachable; mount the dataset again or point the root at its new location"
SOURCE_MISSING = {
    "detail": {
        "error": "source_missing",
        "alias": "archive",
        "root_mounted": False,
        "image_path": "/data/archive/a.jpg",
        "message": MESSAGE,
    }
}

SIZES = {"desktop-1440": {"width": 1440, "height": 900}, "narrow-430": {"width": 430, "height": 2400}}


def serve_missing_thumbnails(stub) -> None:
    stub.on("GET", r"/crops/[^/]+/thumbnail(\?|$)", (404, SOURCE_MISSING))
    stub.on("GET", r"/crops/[^/]+/image(\?|$)", (404, SOURCE_MISSING))


def serve_cluster_grid(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})
    serve_missing_thumbnails(stub)


def serve_review_pane(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on(
        "GET",
        r"/review/(?!tabs)",
        (200, {"items": [review_item(i) for i in range(3)], "total": 3, "page": 1, "page_size": 30}),
    )
    serve_missing_thumbnails(stub)


def _no_visible_image(page) -> None:
    """No `<img>` is left in the DOM for a source that is known to be missing."""
    expect(page.locator("main img, [data-testid='review-source-panel'] img")).to_have_count(0)


def _shot(page, name: str, size: str) -> None:
    out = os.environ.get("OP_E2E_SHOT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(out) / f"{name}-{size}.png"), full_page=True)


@pytest.mark.parametrize("size", list(SIZES))
def test_crop_card_shows_source_not_mounted(stub, page, app_url, size):
    page.set_viewport_size(SIZES[size])
    serve_cluster_grid(stub)
    page.goto(f"{app_url}/p/default/clusters/1")
    badges = page.get_by_role("status").filter(has_text="Source not mounted")
    expect(badges.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(badges).to_have_count(3)
    expect(badges.first).to_contain_text("archive")
    assert badges.first.get_attribute("title") == MESSAGE
    _no_visible_image(page)
    _shot(page, "cluster-grid", size)
    assert stub.unhandled == [], stub.unhandled


@pytest.mark.parametrize("size", list(SIZES))
def test_review_pane_shows_source_not_mounted(stub, page, app_url, size):
    page.set_viewport_size(SIZES[size])
    serve_review_pane(stub)
    page.goto(f"{app_url}/p/default/review")
    badge = page.get_by_role("status").filter(has_text="Source not mounted")
    expect(badge.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(badge.first).to_contain_text("archive")
    expect(badge.first).to_contain_text("not mounted or reachable")
    panel = page.get_by_test_id("review-source-panel")
    expect(panel.get_by_role("status")).to_contain_text("Source not mounted")
    expect(panel.get_by_role("button", name="hide boxes")).to_have_count(0)
    expect(page.get_by_role("status").filter(has_text="Source not mounted")).to_have_count(2)
    _no_visible_image(page)
    _shot(page, "review-pane", size)
    assert stub.unhandled == [], stub.unhandled
