"""W8 multi-box regions (feat/w8-multibox-lockstep, docs/design/
w8-multibox-frontend-plan-2026-09-26.md): forward-tolerance check, not a
full new-semantics test.

This branch ships the wire model (SubBoxCapability.listField,
readSlot -> SlotData.subBoxes, the new PUT/PATCH/batch_box_state api.ts
functions, MultiBoxCanvas.svelte) but does NOT yet wire the multi-box
editor into /review's 3000-line +page.svelte in this pass — see the
plan doc's "deliberately out of scope" section. So there is no new
Enter-confirms-only-proposed-boxes UI to end-to-end test yet, and
rewriting test_region_verify_rejected_confirm.py to assert that
semantics would describe behavior that doesn't exist, which is worse
than not testing it.

What this DOES verify, honestly: a backend that serves BOTH the legacy
scalar region_* fields (which is what /review's current UI still reads)
AND the new `region_boxes` list (W8's ItemDoc addition) doesn't crash the
page — mapCropSlots/readSlot populate SlotData.subBoxes from the list
alongside the existing subBox reading, with zero pageerror. This is the
actual regression risk this branch introduces to the shipped app: does
adding the new list-reading code path break the page for every existing
crop shape. It's proved here rather than only by the unit tests in
src/lib/annotations/multiBox.test.ts and readSlot.test.ts, which don't
exercise a real page mount.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item, REGION_CLASS, REGION_TAB_URL_ID

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
_BOX_B = [0.5, 0.5, 0.7, 0.7]


def multibox_item() -> dict:
    return make_item(
        crop_id="tag-multibox-1",
        image_id="img-1",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url="/curation/crops/tag-multibox-1/thumbnail",
        region_status="detected",
        region_verified=True,
        region_validated=True,
        # Legacy scalar fields — still what the shipped /review UI reads.
        region_bbox_norm=_BOX_A,
        region_bbox_in_parent=_BOX_A,
        # New W8 list — not yet read by /review's UI, but must not crash
        # mapCropSlots/readSlot when it's present alongside the legacy
        # fields (the exact shape a mid-migration backend would serve
        # per region_boxes_migration.pending).
        region_boxes=[
            {
                "box_id": "b1",
                "state": "accepted",
                "bbox_norm": _BOX_A,
                "bbox_in_parent": _BOX_A,
                "score": 0.9,
                "detector": "tag_detector_v1",
                "detector_version": "1",
                "source": "detector",
                "bbox_correct": True,
                "confidence": "high",
                "rejection_reason": None,
                "text": "TAG-001",
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
                "thumbnail_url": "/curation/crops/tag-multibox-1/region_thumbnail?box_id=b1",
            },
            {
                "box_id": "b2",
                "state": "rejected",
                "bbox_norm": _BOX_B,
                "bbox_in_parent": _BOX_B,
                "score": 0.4,
                "detector": "tag_segmenter",
                "detector_version": "3",
                "source": "segmenter",
                "bbox_correct": None,
                "confidence": None,
                "rejection_reason": "verifier_no_verdict",
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
                "thumbnail_url": "/curation/crops/tag-multibox-1/region_thumbnail?box_id=b2",
            },
        ],
    )


def test_review_region_tab_tolerates_a_served_region_boxes_list(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(_request, _match):
        return (
            200,
            {"items": [multibox_item()], "total": 1, "page": 1, "page_size": 30},
        )

    stub.on("GET", r"/review/", review_handler)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, (
        f"a served region_boxes list must not crash the page "
        f"(no /review UI reads it yet, but mapCropSlots must tolerate it): {errors[:3]}"
    )
