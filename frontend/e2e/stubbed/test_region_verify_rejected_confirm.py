"""W8 multi-box regions (docs/design/w8-multibox-frontend-plan-2026-09-26.md;
owner decision recorded 2026-09-26): Enter confirms only `proposed` boxes,
leaving `rejected`/`false_positive` siblings exactly as they are — a
whole-set confirm never overrides a per-box decision. This supersedes the
pre-W8 file this replaces (`test_confirming_a_verify_rejected_item_...`),
which tested the OLD single-box "confirm promotes the rejected candidate"
behavior; W8 removes candidate promotion entirely (a rejected candidate is
just a box with `state: 'rejected'`, per §7.7).

The multi-box UI activates per-item, gated on the crop actually carrying a
served `region_boxes` list (`readSlot.ts`) — every fixture here serves it,
so these tests exercise `MultiBoxCanvas` / `multiBoxRegionController`, not
the legacy `BboxCanvas` path (still covered by `test_region_confirm.py`,
`test_region_bbox_edit_nudge.py`, `test_region_undo.py` against fixtures
that omit `region_boxes`).
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
        "rejection_reason": "verifier_no_verdict" if state == "rejected" else None,
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
        "thumbnail_url": f"/curation/crops/tag-mb-1/region_thumbnail?box_id={box_id}",
    }


def multibox_item(boxes: list[dict], region_status: str = "pending_verification") -> dict:
    return make_item(
        crop_id="tag-mb-1",
        image_id="img-1",
        class_id=1,
        class_name=REGION_CLASS,
        thumbnail_url="/curation/crops/tag-mb-1/thumbnail",
        region_status=region_status,
        region_boxes=boxes,
    )


def _setup(stub, items_provider):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))
    stub.on(
        "GET",
        r"/review/",
        lambda _r, _m: (
            200,
            {"items": [items_provider()], "total": 1, "page": 1, "page_size": 30},
        ),
    )


def test_enter_confirms_only_proposed_boxes_and_leaves_rejected_untouched(
    stub, page, app_url
):
    puts: list[dict] = []

    _setup(
        stub,
        lambda: multibox_item([_box("b1", "proposed", _BOX_A), _box("b2", "rejected", _BOX_B)]),
    )

    def region_put(request, _match):
        body = request.post_data_json or {}
        puts.append(body)
        return (
            200,
            {
                "item": multibox_item(
                    [_box("b1", "accepted", _BOX_A), _box("b2", "rejected", _BOX_B)],
                    region_status="detected",
                )
            },
        )

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert len(puts) == 1, puts
    body = puts[0]
    assert body["region_status"] == "detected", body
    assert body["boxes"] == [
        {"box_id": "b1", "state": "accepted"},
        {"box_id": "b2"},
    ], body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_per_box_accept_and_reject_keys_patch_immediately_without_removing_from_queue(
    stub, page, app_url
):
    patches: list[tuple[str, dict]] = []

    _setup(
        stub,
        lambda: multibox_item([_box("b1", "proposed", _BOX_A), _box("b2", "proposed", _BOX_B)]),
    )

    def region_patch(request, match):
        box_id = match.group(2)
        body = request.post_data_json or {}
        patches.append((box_id, body))
        boxes = [
            _box("b1", body.get("state", "proposed") if box_id == "b1" else "proposed", _BOX_A),
            _box("b2", body.get("state", "proposed") if box_id == "b2" else "proposed", _BOX_B),
        ]
        return (200, {"crop_id": "tag-mb-1", "box": boxes[0], "updated_fields": ["state"], "item": multibox_item(boxes)})

    stub.on("PATCH", r"/crops/([^/]+)/regions/([^/]+)$", region_patch)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)
    counter_before = page.get_by_test_id("queue-counter").first.inner_text()

    # b1 (selected by default, position 1) rejected via 'r'.
    page.keyboard.press("r")
    page.wait_for_timeout(300)
    # Tab to b2, accept via 'y'.
    page.keyboard.press("Tab")
    page.wait_for_timeout(100)
    page.keyboard.press("y")
    page.wait_for_timeout(300)

    assert [p[0] for p in patches] == ["b1", "b2"], patches
    assert patches[0][1]["state"] == "rejected", patches[0]
    assert patches[1][1]["state"] == "accepted", patches[1]
    # A per-box PATCH is not a whole-item confirm — the crop stays in view.
    assert page.get_by_test_id("queue-counter").first.inner_text() == counter_before

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_added_box_and_untouched_sibling_are_both_sent_in_one_confirm_write(
    stub, page, app_url
):
    puts: list[dict] = []

    _setup(stub, lambda: multibox_item([_box("b1", "proposed", _BOX_A)]))

    def region_put(request, _match):
        body = request.post_data_json or {}
        puts.append(body)
        return (
            200,
            {
                "item": multibox_item(
                    [_box("b1", "accepted", _BOX_A), _box("b2", "accepted", [0.6, 0.6, 0.8, 0.8])],
                    region_status="detected",
                )
            },
        )

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    canvas = page.get_by_test_id("multibox-canvas").first
    canvas.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    # Enter edit mode, then drag out a new box in an empty area of the canvas.
    page.keyboard.press("e")
    page.wait_for_timeout(200)
    box = canvas.bounding_box()
    assert box is not None
    x0 = box["x"] + box["width"] * 0.6
    y0 = box["y"] + box["height"] * 0.6
    x1 = box["x"] + box["width"] * 0.85
    y1 = box["y"] + box["height"] * 0.85
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=5)
    page.mouse.up()
    page.wait_for_timeout(200)

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert len(puts) == 1, puts
    body = puts[0]
    boxes = body["boxes"]
    assert len(boxes) == 2, boxes
    confirmed = next(b for b in boxes if b.get("box_id") == "b1")
    added = next(b for b in boxes if b.get("box_id") is None)
    # b1 was `proposed` -> Enter's confirm settles it to accepted (no
    # geometry change, so no bbox_norm on this element).
    assert confirmed == {"box_id": "b1", "state": "accepted"}, confirmed
    assert "bbox_norm" in added, added
    assert "state" not in added, added  # relies on the server's new_box_default

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_z_undo_after_a_multibox_confirm_calls_the_single_region_undo_route(
    stub, page, app_url
):
    puts: list[dict] = []
    undo_calls: list[str] = []

    _setup(
        stub,
        lambda: multibox_item([_box("b1", "proposed", _BOX_A), _box("b2", "rejected", _BOX_B)]),
    )

    def region_put(request, _match):
        puts.append(request.post_data_json or {})
        return (
            200,
            {
                "item": multibox_item(
                    [_box("b1", "accepted", _BOX_A), _box("b2", "rejected", _BOX_B)],
                    region_status="detected",
                )
            },
        )

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    def region_undo(request, match):
        undo_calls.append(match.string)
        # Backend-confirmed W8 undo contract: restores the whole prior
        # region_boxes list + region_status + region_revision in one step.
        return (
            200,
            multibox_item(
                [_box("b1", "proposed", _BOX_A), _box("b2", "rejected", _BOX_B)],
                region_status="pending_verification",
            ),
        )

    stub.on("POST", r"/crops/([^/]+)/region/undo$", region_undo)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    page.keyboard.press("Enter")
    page.wait_for_timeout(400)
    assert len(puts) == 1, puts

    page.keyboard.press("z")
    page.wait_for_timeout(400)

    assert len(undo_calls) == 1, undo_calls
    assert "tag-mb-1" in undo_calls[0], undo_calls

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"


def test_on_screen_confirm_button_uses_the_multibox_write_too(stub, page, app_url):
    """The Confirm button (not just Enter) must call the W8 multi-box
    confirm — a real bug found live: the action-button row was wired to
    the legacy single-box confirmSlot()/saveBboxAndExit() unconditionally,
    so clicking Confirm (or Save bbox in edit mode) sent nothing (or
    404ed) for a multi-box slot even though the keyboard path worked.
    """
    puts: list[dict] = []

    _setup(
        stub,
        lambda: multibox_item([_box("b1", "proposed", _BOX_A)]),
    )

    def region_put(request, _match):
        puts.append(request.post_data_json or {})
        return (
            200,
            {"item": multibox_item([_box("b1", "accepted", _BOX_A)], region_status="detected")},
        )

    stub.on("PUT", r"/crops/([^/]+)/regions$", region_put)

    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("multibox-canvas").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    page.get_by_role("button", name="Confirm", exact=False).first.click()
    page.wait_for_timeout(400)

    assert len(puts) == 1, puts
    assert puts[0]["region_status"] == "detected", puts[0]

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
