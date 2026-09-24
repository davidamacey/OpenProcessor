"""dq-region (backend openprocessor f7171cc, frontend adoption 2026-09-24):
a verify_rejected item carries no region_bbox_norm at all — the box
lives in region_candidate_bbox_norm/_score/_detector/... until a human
accepts it. /review's slot panel must seed its edit box from the
candidate (_seedSlotFromCurrent) so an unchanged Confirm still goes
through the existing boxUnchanged -> status-only PATCH region_meta
path (same B2 provenance-preserving contract test_plate_confirm.py
covers for a normal detected box) rather than warning "no bbox to
confirm" and doing nothing.

Server-side this status-only PATCH promotes the candidate into the
region box (region_writes.candidate_promotion / human_status_fields,
read directly from openprocessor f7171cc) — the frontend never computes
that promotion itself, it only has to send the write and render
whatever the server returns.
"""

from __future__ import annotations

from fixtures.wire import make_item

CLASSES = [
    {
        "id": 1,
        "name": "license_plate",
        "group": "vehicle",
        "hotkey_letter": "l",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
]

METHODS = {"strategies": [], "flags": {}}

_CANDIDATE_BBOX = [0.15, 0.25, 0.55, 0.75]


def rejected_plate_item() -> dict:
    return make_item(
        crop_id="plate-rejected-1",
        image_id="img-1",
        class_id=1,
        class_name="license_plate",
        thumbnail_url="/curation/crops/plate-rejected-1/thumbnail",
        region_bbox_norm=None,
        region_bbox_in_parent=None,
        region_status="verify_rejected",
        region_rejection_reason="verifier_no_verdict",
        region_verified=False,
        region_validated=False,
        region_auto_confirmed=False,
        region_candidate_bbox_norm=_CANDIDATE_BBOX,
        region_candidate_bbox_in_parent=_CANDIDATE_BBOX,
        region_candidate_score=0.51,
        region_candidate_detector="sam3",
        region_candidate_detector_version="3.0.0",
        region_candidate_source="segmenter",
    )


def test_confirming_a_verify_rejected_item_promotes_the_candidate_via_status_patch(
    stub, page, app_url
):
    region_calls: list[tuple[str, str, dict]] = []
    region_meta_calls: list[tuple[str, str, dict]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(_request, _match):
        return (
            200,
            {"items": [rejected_plate_item()], "total": 1, "page": 1, "page_size": 30},
        )

    stub.on("GET", r"/review/", review_handler)

    def region_handler(request, match):
        region_calls.append((request.method, match.string, request.post_data_json or {}))
        # A promoted item, as the backend would actually return it.
        promoted = make_item(
            crop_id="plate-rejected-1",
            image_id="img-1",
            class_id=1,
            class_name="license_plate",
            region_bbox_norm=_CANDIDATE_BBOX,
            region_status="detected",
            region_validated=True,
        )
        return (200, {"item": promoted})

    stub.on("PUT", r"/crops/([^/]+)/region$", region_handler)

    def region_meta_handler(request, match):
        region_meta_calls.append((request.method, match.string, request.post_data_json or {}))
        promoted = make_item(
            crop_id="plate-rejected-1",
            image_id="img-1",
            class_id=1,
            class_name="license_plate",
            region_bbox_norm=_CANDIDATE_BBOX,
            region_status="detected",
            region_validated=True,
            region_candidate_bbox_norm=None,
        )
        return (200, {"crop_id": "plate-rejected-1", "updated_fields": ["region_status"], "item": promoted})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", region_meta_handler)

    page.goto(f"{app_url}/review?tab=plates")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=15000)

    # The candidate-box hint renders before any action — proves the item
    # is recognized as a rejected candidate, not silently treated as
    # "no bbox at all".
    page.get_by_text("rejected candidate", exact=False).first.wait_for(timeout=10000)

    # Give the canvas a beat to seed editedSlotBox from the candidate box
    # before confirming (same pattern as test_plate_confirm.py).
    page.wait_for_timeout(500)

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert region_calls == [], (
        f"an unchanged candidate-box confirm must not PUT region: {region_calls}"
    )
    assert len(region_meta_calls) == 1, (
        f"Enter should PATCH region_meta exactly once to promote the candidate: {region_meta_calls}"
    )
    method, path, body = region_meta_calls[0]
    assert method == "PATCH"
    assert "plate-rejected-1" in path, path
    assert body.get("region_status") == "detected", body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the verify_rejected confirm flow: {errors[:3]}"
