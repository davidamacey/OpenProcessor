"""`More like this`: a crop card's action and the review pane's link open the /clusters results
for `GET /search/similar`; a served refusal shows verbatim; the control is absent unless the
deployment serves semantic search.
"""

from __future__ import annotations

from pathlib import Path

from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import wait_for_paint
from playwright.sync_api import expect

from test_cluster_discard import CLASSES, CLUSTERS, crop

SHOTS = Path(__file__).resolve().parents[2] / "artifacts_local" / "similar-crops"

SEMANTIC_ON = {
    "strategies": [
        {"id": "semantic_search", "axis": "overlay", "label": "Semantic search", "status": "stable"}
    ],
    "flags": {},
}
SEMANTIC_OFF = {"strategies": [], "flags": {}}


def neighbour(i: int, score: float) -> dict:
    return {**crop(i), "semantic_score": score}


def serve_similar(stub, methods: dict = SEMANTIC_ON, similar=None) -> list[str]:
    """The /clusters stubs plus the methods list and `/search/similar`; returns the query
    strings the stub saw."""
    seen: list[str] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})
    stub.on("GET", r"/methods(\?|$)", methods)

    def handler(request, match):
        seen.append(request.url.split("?", 1)[1] if "?" in request.url else "")
        if similar is not None:
            return similar
        return {
            "items": [neighbour(1, 0.91), neighbour(2, 0.84)],
            "total": 2,
            "page": 1,
            "page_size": 200,
            "unembedded_in_scope": 0,
        }

    stub.on("GET", r"/search/similar(\?|$)", handler)
    return seen


def shoot(page, name: str, frame=None) -> None:
    """Both widths; `frame` re-frames the page after each resize (hover, scroll)."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    for width, height in ((1440, 900), (430, 932)):
        page.set_viewport_size({"width": width, "height": height})
        if frame is not None:
            frame()
        wait_for_paint(page)
        page.screenshot(path=str(SHOTS / f"similar-{name}-{width}.png"), full_page=True)


def assert_results(page, seen: list[str]) -> None:
    page.wait_for_url(lambda url: url.endswith("/p/default/clusters?similar_to=crop-0"), timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("search-summary")).to_contain_text("2", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text("91%").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert any("crop_id=crop-0" in q for q in seen), seen


def test_more_like_this_from_cluster_card(stub, page, app_url):
    seen = serve_similar(stub)
    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    card = page.locator('[data-testid="crop-similar"]').first
    page.get_by_test_id("card-hover-controls").first.hover()
    shoot(page, "card")
    page.set_viewport_size({"width": 1440, "height": 900})
    page.get_by_test_id("card-hover-controls").first.hover()
    card.click()

    assert_results(page, seen)
    shoot(page, "results", lambda: page.get_by_test_id("card-hover-controls").first.hover())
    page.set_viewport_size({"width": 1440, "height": 900})
    page.get_by_role("button", name="Back to clusters").click()
    page.wait_for_url(lambda url: "similar_to" not in url, timeout=ACTION_TIMEOUT_MS)
    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_more_like_this_from_review_pane(stub, page, app_url):
    from test_a11y_axe import serve_review  # imported here: that module imports this one

    serve_review(stub)
    seen = serve_similar(stub)
    page.goto(f"{app_url}/p/default/review")
    link = page.get_by_test_id("review-similar")
    expect(link).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shoot(page, "review", lambda: link.scroll_into_view_if_needed())
    page.set_viewport_size({"width": 1440, "height": 900})
    link.click()
    assert_results(page, seen)


def test_more_like_this_unembedded_shows_served_message(stub, page, app_url):
    message = "crop crop-0 has no vector (deferred)"
    serve_similar(
        stub,
        similar=(409, {"detail": {"error": "crop_not_embedded", "message": message}}),
    )
    page.goto(f"{app_url}/p/default/clusters?similar_to=crop-0")
    expect(page.get_by_test_id("similar-error")).to_have_text(message, timeout=ACTION_TIMEOUT_MS)
    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_more_like_this_hidden_when_semantic_search_off(stub, page, app_url):
    serve_similar(stub, methods=SEMANTIC_OFF)
    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("crop-similar").count() == 0

    from test_a11y_axe import serve_review

    serve_review(stub)
    stub.on("GET", r"/methods(\?|$)", SEMANTIC_OFF)
    page.goto(f"{app_url}/p/default/review")
    expect(page.locator("main")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("review-similar").count() == 0
