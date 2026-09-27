"""M5 (docs/design/interactive-pass-2026-09-24.md): `/clusters/[id]` M
(move selected to another cluster) followed by Z must undo the move
through the same label-undo route as every other write here — before
this fix `moveCropIds` never called `undoStore.recordWrites`, so Z after
a move sent no request at all (a silent no-op).

Single crop moved -> Z -> `POST {API_PREFIX}/crops/{id}/label/undo`
(the single-crop route, since exactly one crop's id was recorded).
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item

CLASSES = [
    {"class_id": 1, "class_name": "ducati", "kind": "item", "group": "moto", "hotkey_letter": "k", "sample_count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

CLUSTERS = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 2,
            "validated_count": 0,
            "dominant_class_id": 1,
            "dominant_class_name": "ducati",
            "purity": 1.0,
            "is_unlabeled": False,
            "representatives": [{"crop_id": "crop-0"}],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}


def crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="ducati",
        cluster_id=1,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def test_move_then_undo_sends_single_crop_label_undo(stub, page, app_url):
    move_calls: list[tuple[str, str, dict]] = []
    undo_single_calls: list[tuple[str, str]] = []
    undo_batch_calls: list[tuple[str, str, dict]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 2, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(2)]})

    def move_handler(request, match):
        body = request.post_data_json or {}
        move_calls.append((request.method, match.string, body))
        ids = body.get("crop_ids", [])
        return (200, {"updated": len(ids), "updated_ids": ids, "conflicts": []})

    stub.on("POST", r"/crops/move$", move_handler)

    def undo_single_handler(request, match):
        undo_single_calls.append((request.method, match.string))
        return (200, crop(0))

    stub.on("POST", r"/crops/([^/]+)/label/undo$", undo_single_handler)

    def undo_batch_handler(request, match):
        # Fail-closed proof this route is NOT the one hit for a
        # single-crop move undo.
        body = request.post_data_json or {}
        undo_batch_calls.append((request.method, match.string, body))
        return (200, {"items": [], "undone": 0, "nothing_to_undo": [], "conflicts": [], "not_found": []})

    stub.on("POST", r"/crops/label/undo_batch$", undo_batch_handler)

    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(400)

    page.locator("img").first.click()
    page.wait_for_timeout(150)

    # M opens the move picker; type a target cluster id and Enter to confirm.
    page.keyboard.press("m")
    page.wait_for_selector('input[placeholder="e.g. 42"]', timeout=5000)
    # F-55: the move dialog points at the class-relabel action.
    assert "Assign class to selected" in page.get_by_test_id("move-relabel-hint").inner_text()
    assert page.get_by_test_id("assign-selected").inner_text().strip() == "Assign class to selected"
    page.locator('input[placeholder="e.g. 42"]').fill("42")
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)

    assert len(move_calls) == 1, f"expected exactly one move request: {move_calls}"
    _, _, move_body = move_calls[0]
    assert move_body.get("cluster_id") == 42
    moved_ids = move_body.get("crop_ids", [])
    assert len(moved_ids) == 1, f"expected exactly one crop moved: {move_body}"

    page.keyboard.press("z")
    page.wait_for_timeout(500)

    assert len(undo_batch_calls) == 0, (
        f"a single-crop move's undo must not go through undo_batch: {undo_batch_calls}"
    )
    assert len(undo_single_calls) == 1, (
        f"Z after M should POST exactly one single-crop label/undo: {undo_single_calls}"
    )
    undo_method, undo_path = undo_single_calls[0]
    assert undo_method == "POST"
    assert undo_path.endswith("/label/undo")
    assert f"/crops/{moved_ids[0]}/label/undo" in undo_path

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the move/undo flow: {errors[:3]}"
