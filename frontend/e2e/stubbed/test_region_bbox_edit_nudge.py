"""Region bbox editor (coordinator finding, 2026-09-25): every drag or
arrow nudge in edit mode used to revert the box and drop edit mode (the
reseed effect tracked `editedSlotBox`), after which further arrows paged
the queue and Enter confirmed a DIFFERENT crop.

Enter edit (E) -> ArrowRight x3 -> still in edit mode, same crop, box
moved -> Enter -> exactly one PUT, to the crop that was being edited, with
the nudged box.
"""

from __future__ import annotations

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


def tag_item(i: int) -> dict:
    return make_item(
        crop_id=f"tag-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url=f"/curation/crops/tag-{i}/thumbnail",
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

    stub.on("PUT", r"/crops/([^/]+)/region$", region_put)

    def region_meta(_request, match):
        region_meta_calls.append(match.string)
        return (200, {"crop_id": "x", "updated_fields": [], "item": tag_item(1)})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", region_meta)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=15000)
    page.wait_for_timeout(500)
    counter_before = page.get_by_test_id("queue-counter").first.inner_text()

    page.keyboard.press("e")
    save_btn = page.get_by_role("button", name="Save bbox")
    save_btn.wait_for(timeout=5000)

    for _ in range(3):
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(100)

    # Still editing, still the same crop.
    assert save_btn.count() == 1, "an arrow nudge dropped edit mode"
    assert page.get_by_test_id("queue-counter").first.inner_text() == counter_before, (
        "arrow keys in edit mode paged the queue"
    )

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert region_meta_calls == [], f"Enter in edit mode must not confirm: {region_meta_calls}"
    assert len(region_puts) == 1, region_puts
    path, body = region_puts[0]
    assert path.endswith("/crops/tag-1/region"), path
    x1 = body["region_bbox_norm"][0]
    # make_item's default parent-frame box starts at x1 = 0.1; three right
    # nudges moved it.
    assert x1 > 0.1 + 1e-6, body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
