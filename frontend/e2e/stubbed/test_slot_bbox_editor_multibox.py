"""W8 multi-box editor modal (docs/design/w8-multibox-frontend-plan-
2026-09-26.md): `SlotBboxEditor.svelte` (the CropCard pencil ✎, `/clusters/
[id]`) now reuses `MultiBoxCanvas`/`multiBoxRegionController` for the
region slot instead of the legacy single-box canvas — the served region
slot has no `bboxField` at all under W8, so the old single-box modal was
unreachable for region. This proves the pencil opens the multi-box
canvas, adding a box and saving sends a plain `PUT /crops/{id}/regions`
with the accumulated diff and no `region_status` key (this modal has no
"confirm" concept, unlike /review's Enter).

No fixed sleeps (parallel e2e, CLAUDE.md): every wait is a real selector
or `page.expect_request`.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import REGION_CLASS, make_item

CLASSES = [
    {
        "id": 1,
        "name": REGION_CLASS,
        "group": "widgets",
        "hotkey_letter": "l",
        "count": 10,
        "validated_count": 5,
        "cluster_size": 10,
        "deprecated": False,
    },
]

CLUSTERS = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 1,
            "validated_count": 0,
            "dominant_class_id": 1,
            "dominant_class_name": REGION_CLASS,
            "purity": 1.0,
            "is_unlabeled": False,
            "representatives": [{"crop_id": "tag-1"}],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}


def _box(box_id: str, bbox: list[float]) -> dict:
    return {
        "box_id": box_id,
        "state": "accepted",
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
        "thumbnail_url": f"/curation/crops/tag-1/region_thumbnail?box_id={box_id}",
    }


def tag_item() -> dict:
    return make_item(
        crop_id="tag-1",
        image_id="img-1",
        class_id=1,
        class_name=REGION_CLASS,
        cluster_id=1,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url="/curation/crops/tag-1/thumbnail",
        region_status="pending_verification",
        region_boxes=[_box("b1", [0.1, 0.1, 0.3, 0.3])],
    )


def test_cropcard_pencil_opens_multibox_editor_and_saves_with_no_region_status(
    stub, page, app_url
):
    page.set_viewport_size({"width": 1280, "height": 1400})
    puts: list[dict] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 1, "page": 1, "page_size": 60, "crops": [tag_item()]})
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def region_put(request, _match):
        body = request.post_data_json or {}
        puts.append(body)
        return (200, {"item": tag_item()})

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    page.goto(f"{app_url}/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)

    thumbnail = page.locator("img").first
    thumbnail.hover()
    # `button[aria-label="Edit ..."]`, not get_by_role("Edit", exact=False) —
    # the card's own outer role="button" wrapper's computed accessible name
    # also matches a loose "Edit" substring, so a fuzzy role query picks the
    # card (toggling selection) instead of the nested pencil button.
    pencil = page.locator('button[aria-label="Edit widget tag"]')
    pencil.wait_for(timeout=ACTION_TIMEOUT_MS, state="visible")
    pencil.click()

    canvas = page.get_by_test_id("multibox-canvas")
    canvas.wait_for(timeout=ACTION_TIMEOUT_MS)

    # Drag-draw a second box on empty canvas.
    box = canvas.bounding_box()
    assert box is not None
    start_x, start_y = box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.6
    end_x, end_y = box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8
    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(end_x, end_y)
    page.mouse.up()

    save_button = page.get_by_role("button", name="Save", exact=True)
    save_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    with page.expect_request(
        lambda r: r.method == "PUT" and "/regions" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        save_button.click()

    assert len(puts) == 1, puts
    body = puts[0]
    assert "region_status" not in body, (
        f"the standalone bbox editor has no confirm concept — must never send "
        f"region_status: {body}"
    )
    assert body["boxes"][0] == {"box_id": "b1"}, "untouched sibling stays {box_id} only"
    assert body["boxes"][1]["box_id"] is None, "the newly drawn box has no box_id"
    assert "bbox_norm" in body["boxes"][1]

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the multi-box editor flow: {errors[:3]}"

    stub.assert_fail_closed()
