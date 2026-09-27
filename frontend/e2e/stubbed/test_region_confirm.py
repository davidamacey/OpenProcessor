"""B2 (frontend half, docs/design/interactive-pass-2026-09-24.md §6):
confirming a region whose box the operator never touched must go through
a status-only `PATCH {API_PREFIX}/crops/{id}/region_meta` using the
served `confirm_status`, not `PUT {API_PREFIX}/crops/{id}/region` with
the same box — a same-box PUT is indistinguishable, server-side, from a
human drawing a fresh box, and overwrites the detector's own
`region_detector`/`region_score` provenance.

Enter on the region tab -> confirmSlot() -> (unchanged box, served
confirm_status) -> `PATCH {API_PREFIX}/crops/{id}/region_meta` (src/routes/
review/+page.svelte).
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item, REGION_CLASS, REGION_TAB_URL_ID

CLASSES = [
    {"id": 1, "name": REGION_CLASS, "group": "widgets", "hotkey_letter": "l", "count": 40, "validated_count": 12, "cluster_size": 44, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}


def tag_item() -> dict:
    # region_bbox_in_parent / region_status / region_detector /
    # region_score all come from make_item's defaults — an untouched,
    # detector-found box with real provenance, exactly the case B2
    # covers (an operator confirming a box that's already correct).
    return make_item(
        crop_id="tag-1",
        image_id="img-1",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url="/curation/crops/tag-1/thumbnail",
    )


def test_region_confirm_unchanged_box_sends_patch_region_meta(stub, page, app_url):
    region_calls: list[tuple[str, str, dict]] = []
    region_meta_calls: list[tuple[str, str, dict]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    # The source-image-with-bbox <img> on the left panel.
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        (200, b"", "image/jpeg"),
    )

    def review_handler(_request, _match):
        return (200, {"items": [tag_item()], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)

    def region_handler(request, match):
        region_calls.append((request.method, match.string, request.post_data_json or {}))
        return (200, {"item": tag_item()})

    # PUT {API_PREFIX}/crops/{id}/region — must NOT be called for an
    # unchanged-box confirm.
    stub.on("PUT", r"/crops/([^/]+)/region$", region_handler)

    def region_meta_handler(request, match):
        region_meta_calls.append((request.method, match.string, request.post_data_json or {}))
        return (200, {"crop_id": "tag-1", "updated_fields": ["region_status"], "item": tag_item()})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", region_meta_handler)

    page.goto(f"{app_url}/p/default/review?tab={REGION_TAB_URL_ID}")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)

    # Give the slot canvas a beat to seed editedSlotBox from the served
    # region_bbox_in_parent before confirming.
    page.wait_for_timeout(500)

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert region_calls == [], (
        f"an unchanged-box confirm must not PUT region (rewrites detector provenance): {region_calls}"
    )
    assert len(region_meta_calls) == 1, f"Enter should PATCH region_meta exactly once: {region_meta_calls}"
    method, path, body = region_meta_calls[0]
    assert method == "PATCH"
    assert path.endswith("/region_meta")
    assert "tag-1" in path, path
    # confirm_status served by the stub's default GET /regions/statuses
    # (conftest.py) is "detected" — the same value the region slot's
    # own capabilities.lifecycle.confirmState uses.
    assert body.get("region_status") == "detected", body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the region-confirm flow: {errors[:3]}"
