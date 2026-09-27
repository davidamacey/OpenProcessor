"""W8 multi-box regions (docs/design/w8-multibox-frontend-plan-2026-09-26.md).
Supersedes the pre-W8 single-box nudge test this file originally covered —
the region slot's editor is `MultiBoxCanvas` now, not the legacy
`BboxCanvas`/"Save bbox" button, and edits flush through
`PUT /crops/{id}/regions`, not `PUT /crops/{id}/region` (a removed 410
route per W8.8 — no backward compatibility, owner decision 2026-09-26).

Enter edit (E) -> ArrowRight x3 nudges the selected box -> still in edit
mode, same crop -> Enter -> exactly one PUT /crops/{id}/regions, to the
crop being edited, with the nudged geometry.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import REGION_CLASS, REGION_TAB_URL_ID, make_item

CLASSES = [
    {
        "id": 1,
        "name": REGION_CLASS,
        "group": "widgets",
        "hotkey_letter": "l",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
]

METHODS = {"strategies": [], "flags": {}}

_BOX_A = [0.1, 0.1, 0.3, 0.3]


def _box(box_id: str, state: str, bbox: list[float]) -> dict:
    return {
        "box_id": box_id,
        "state": state,
        "bbox_norm": bbox,
        "bbox_in_parent": bbox,
        "score": 0.8,
        "detector": "tag_detector_v1",
        "detector_version": "1",
        "source": "detector",
        "bbox_correct": None,
        "confidence": None,
        "rejection_reason": None,
        "text": None,
        "text_raw": None,
        "text_confidence": None,
        "text_source": None,
        "text_engine_version": None,
        "text_vlm": None,
        "text_ocr": None,
        "text_disagreement": None,
        "text_choice": None,
        "text_vlm_invalid": None,
        "cluster_id": None,
        "cluster_subid": None,
        "cluster_distance": None,
        "detected_at": "2026-05-06T07:08:09Z",
        "thumbnail_url": f"/curation/crops/{box_id}-thumb/region_thumbnail?box_id={box_id}",
    }


def tag_item(i: int) -> dict:
    return make_item(
        crop_id=f"tag-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url=f"/curation/crops/tag-{i}/thumbnail",
        region_status="pending_verification",
        region_boxes=[_box("b1", "proposed", _BOX_A)],
    )


def test_nudges_stay_in_edit_mode_and_save_to_the_edited_crop(stub, page, app_url):
    region_puts: list[tuple[str, dict]] = []
    region_meta_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))
    stub.on(
        "GET",
        r"/review/",
        lambda _r, _m: (
            200,
            {"items": [tag_item(i) for i in range(1, 4)], "total": 3, "page": 1, "page_size": 30},
        ),
    )

    def region_put(request, match):
        region_puts.append((match.string, request.post_data_json or {}))
        return (200, {"item": tag_item(1)})

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    def region_meta(_request, match):
        region_meta_calls.append(match.string)
        return (200, {"crop_id": "x", "updated_fields": [], "item": tag_item(1)})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", region_meta)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(500)
    counter_before = page.get_by_test_id("queue-counter").first.inner_text()

    page.keyboard.press("e")
    canvas = page.get_by_test_id("multibox-canvas").first
    canvas.wait_for(timeout=5000)

    for _ in range(3):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(100)

    # Still editing, still the same crop.
    assert canvas.count() == 1, "an arrow nudge dropped edit mode"
    assert page.get_by_test_id("queue-counter").first.inner_text() == counter_before, (
        "arrow keys in edit mode paged the queue"
    )

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert region_meta_calls == [], f"Enter in edit mode must not PATCH region_meta: {region_meta_calls}"
    assert len(region_puts) == 1, region_puts
    path, body = region_puts[0]
    assert path.endswith("/crops/tag-1/regions"), path
    assert body["region_status"] == "detected", body
    moved = next(b for b in body["boxes"] if b.get("box_id") == "b1")
    assert "bbox_norm" in moved, moved
    x1 = moved["bbox_norm"][0]
    # b1 started at x1 = 0.1; three right nudges moved it.
    assert x1 > 0.1 + 1e-6, body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
