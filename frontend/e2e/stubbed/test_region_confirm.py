"""W8 multi-box regions (docs/design/w8-multibox-frontend-plan-2026-09-26.md).
Supersedes the pre-W8 B2 test this file originally covered: confirming an
UNCHANGED box must never rewrite the detector's own provenance
(`region_detector`/`region_score`). Under W8 this is no longer a
"status-only PATCH region_meta instead of PUT" split — it is the PUT
element-addressing rule itself (W8.8): a box included with `state` but NO
`bbox_norm` keeps its stored geometry AND provenance server-side, only its
state changes. So Enter on an untouched `proposed` box sends exactly one
`PUT {API_PREFIX}/crops/{id}/regions` with `boxes: [{box_id, state:
"accepted"}]` — no `bbox_norm` key at all, and no PATCH region_meta.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item, REGION_CLASS, REGION_TAB_URL_ID

CLASSES = [
    {"class_id": 1, "class_name": REGION_CLASS, "kind": "region", "group": "widgets", "hotkey_letter": "l", "sample_count": 40, "validated_count": 12, "cluster_size": 44, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}

_BOX_A = [0.1, 0.1, 0.3, 0.3]


def _box() -> dict:
    return {
        "box_id": "b1",
        "state": "proposed",
        "bbox_norm": _BOX_A,
        "bbox_in_parent": _BOX_A,
        "score": 0.91,
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
        "thumbnail_url": "/curation/crops/tag-1/region_thumbnail?box_id=b1",
    }


def tag_item() -> dict:
    return make_item(
        crop_id="tag-1",
        image_id="img-1",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url="/curation/crops/tag-1/thumbnail",
        region_status="pending_verification",
        region_boxes=[_box()],
    )


def test_region_confirm_unchanged_box_sends_state_only_no_bbox_norm(stub, page, app_url):
    region_puts: list[tuple[str, dict]] = []
    region_meta_calls: list[tuple[str, dict]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(_request, _match):
        return (200, {"items": [tag_item()], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)

    def region_put(request, match):
        region_puts.append((match.string, request.post_data_json or {}))
        accepted = _box()
        accepted["state"] = "accepted"
        return (
            200,
            {
                "item": make_item(
                    crop_id="tag-1",
                    image_id="img-1",
                    class_id=1,
                    class_name=REGION_CLASS,
                    thumbnail_url="/curation/crops/tag-1/thumbnail",
                    region_status="detected",
                    region_boxes=[accepted],
                )
            },
        )

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    def region_meta(request, match):
        region_meta_calls.append((match.string, request.post_data_json or {}))
        return (200, {"crop_id": "tag-1", "updated_fields": [], "item": tag_item()})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", region_meta)

    page.goto(f"{app_url}/p/default/review?tab={REGION_TAB_URL_ID}")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert region_meta_calls == [], (
        f"W8 confirm goes through PUT .../regions, never PATCH region_meta: {region_meta_calls}"
    )
    assert len(region_puts) == 1, region_puts
    path, body = region_puts[0]
    assert path.endswith("/crops/tag-1/regions"), path
    assert body["region_status"] == "detected", body
    assert body["boxes"] == [{"box_id": "b1", "state": "accepted"}], (
        f"an untouched box's confirm must carry state only, no bbox_norm "
        f"(preserves detector provenance server-side): {body}"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the region-confirm flow: {errors[:3]}"
