"""K6 (docs/design/k6-frontend-overlay-plan-2026-09-24.md): the backend
is removing its server-side burned-in box overlay from
`GET {API_PREFIX}/crops/{id}/image`. `/review`'s source panel now draws
every box itself from `GET {API_PREFIX}/crops/{id}/context`
(`SourceImageOverlay.svelte`). This proves the wiring end to end against
the stub: the context endpoint gets hit, and the overlay renders one box
per context item, and the "hide boxes" toggle removes them.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from playwright.sync_api import expect

from fixtures.wire import make_item

CLASSES = [
    {"id": 1, "name": "widget_a", "group": "widgets", "hotkey_letter": "w", "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}


def review_item(i: int) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id="img-shared",
        class_id=1,
        class_name="widget_a",
        proposed_class_id=None,
        proposed_class_name=None,
        label_validated=True,
        label_source="human",
        bbox_norm=[0.1 * i, 0.1, 0.1 * i + 0.2, 0.3],
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        # No region evidence on this item — this test proves the plain
        # per-item box; a region slot's own box/candidate rendering is
        # covered by the component mount test
        # (SourceImageOverlay.test.ts), not duplicated here.
        region_bbox_norm=None,
        region_bbox_in_parent=None,
        region_candidate_bbox_norm=None,
        region_candidate_bbox_in_parent=None,
    )
    item["reason"] = "uncertainty"
    return item


def test_review_source_panel_draws_client_side_boxes(stub, page, app_url):
    context_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)

    def review_handler(_request, _match):
        items = [review_item(i) for i in range(3)]
        return (200, {"items": items, "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)

    def context_handler(request, match):
        context_calls.append(match.string)
        return (
            200,
            {
                "image": {
                    "image_id": "img-shared",
                    "image_path": "/fixtures/img-shared.jpg",
                    "width": 640,
                    "height": 480,
                    "source": "fixture",
                    "indexed_at": None,
                },
                "items": [review_item(i) for i in range(3)],
            },
        )

    stub.on("GET", r"/crops/[^/]+/context$", context_handler)
    # The clean (no longer server-annotated) source image itself — a real
    # image response, not JSON, so the <img> element actually loads.
    stub.on("GET", r"/crops/[^/]+/image(\?|$)", stub._image)

    page.goto(f"{app_url}/review")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)

    panel = page.get_by_test_id("review-source-panel")
    overlay_boxes = panel.get_by_test_id("overlay-box")
    overlay_boxes.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert overlay_boxes.count() == 3, "one overlay box per context item"
    assert len(context_calls) >= 1, "the source panel must hit GET .../crops/{id}/context"

    toggle = panel.get_by_role("button", name="hide boxes")
    toggle.first.click()
    expect(panel.get_by_test_id("overlay-box")).to_have_count(0, timeout=ACTION_TIMEOUT_MS)

    toggle_back = panel.get_by_role("button", name="show boxes")
    toggle_back.first.click()
    expect(panel.get_by_test_id("overlay-box")).to_have_count(3, timeout=ACTION_TIMEOUT_MS)

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert errors == [], errors
    assert stub.unhandled == [], stub.unhandled
