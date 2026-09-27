"""New coverage (undo-batching, docs/design/ — one-undo-per-action):
`/clusters/[id]` bulk label (class hotkey over several selected crops)
followed by ONE Z must send exactly one
`POST {API_PREFIX}/crops/label/undo_batch` request — not N single-crop
undo requests. See src/lib/stores/undo.svelte.ts's `recordWrites`/
`undoLast` and src/routes/clusters/[id]/+page.svelte's `assignClassToSelected`.
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
            "size": 3,
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


def test_cluster_bulk_label_then_single_undo_sends_one_batch_call(stub, page, app_url):
    label_calls: list[tuple[str, str, dict]] = []
    undo_calls: list[tuple[str, str, dict]] = []
    undo_single_calls: list[tuple[str, str]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})

    def label_handler(request, match):
        body = request.post_data_json or {}
        label_calls.append((request.method, match.string, body))
        ids = body.get("crop_ids", [])
        return (200, {"updated": len(ids), "updated_ids": ids, "conflicts": [], "skipped": []})

    stub.on("PUT", r"/crops/batch_label$", label_handler)

    def undo_batch_handler(request, match):
        body = request.post_data_json or {}
        undo_calls.append((request.method, match.string, body))
        ids = body.get("crop_ids", [])
        items = [crop(int(i.split("-")[1])) for i in ids]
        return (200, {"items": items, "undone": len(items), "nothing_to_undo": [], "conflicts": [], "not_found": []})

    stub.on("POST", r"/crops/label/undo_batch$", undo_batch_handler)

    def undo_single_handler(request, match):
        # Fail-closed proof: if the bulk write ever mis-routes onto N
        # single-crop undos, this handler is registered and would be hit
        # instead of 501ing — the assertion below on `undo_single_calls`
        # (not just `undo_calls`) is what actually catches that bug.
        undo_single_calls.append((request.method, match.string))
        return (200, crop(0))

    stub.on("POST", r"/crops/([^/]+)/label/undo$", undo_single_handler)

    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(400)

    # Select two crops, then bulk-label them via the class hotkey (no
    # drag — the layout-level keydown listener dispatches with an empty
    # droppedIds, so assignClassToSelected's dropOnClassStore handler
    # falls back to the current selection).
    page.locator("img").nth(0).click()
    page.keyboard.down("Shift")
    page.locator("img").nth(1).click()
    page.keyboard.up("Shift")
    page.wait_for_timeout(150)

    page.keyboard.press("k")
    page.wait_for_timeout(500)

    assert len(label_calls) == 1, f"bulk label should PUT exactly once: {label_calls}"
    _, _, label_body = label_calls[0]
    labeled_ids = label_body.get("crop_ids", [])
    assert len(labeled_ids) == 2, f"expected 2 crops in the bulk label request: {label_body}"

    page.keyboard.press("z")
    page.wait_for_timeout(500)

    assert len(undo_single_calls) == 0, (
        f"one Z over a 2-crop write must not fall back to single-crop undo: {undo_single_calls}"
    )
    assert len(undo_calls) == 1, f"Z should POST exactly one undo_batch call: {undo_calls}"
    undo_method, undo_path, undo_body = undo_calls[0]
    assert undo_method == "POST"
    assert undo_path.endswith("/label/undo_batch")
    assert sorted(undo_body.get("crop_ids", [])) == sorted(labeled_ids), undo_body

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the bulk-label/undo flow: {errors[:3]}"
