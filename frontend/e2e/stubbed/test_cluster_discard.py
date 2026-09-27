"""New coverage (docs/design/test-audit-2026-09-24.md recommendation 5):
`/clusters/[id]` D discards the current selection.

Single selection -> `POST {API_PREFIX}/crops/{id}/discard` (discardCrop).
Multi selection -> `POST {API_PREFIX}/crops/discard_batch`
(discardCropsBatch). See src/routes/clusters/[id]/+page.svelte's `reg('d', ...)`.
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


def test_cluster_single_discard(stub, page, app_url):
    discard_calls: list[tuple[str, str]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})

    def discard_handler(request, match):
        discard_calls.append((request.method, match.string))
        return (200, crop(1))

    stub.on("POST", r"/crops/([^/]+)/discard$", discard_handler)
    # Fail-closed proof this test relies on: batch discard is registered
    # too, so a UI bug that always calls the batch endpoint (even for one
    # selection) would still be handled, not 501 — the assertion below on
    # `discard_calls` catches that class of bug instead.
    stub.on("POST", r"/crops/discard_batch$", lambda req, m: (200, {"items": [], "discarded": 0, "conflicts": [], "not_found": []}))

    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(400)

    page.locator("img").nth(1).click()
    page.wait_for_timeout(150)

    discard_calls.clear()
    page.keyboard.press("d")
    page.wait_for_timeout(500)

    assert len(discard_calls) == 1, f"D on a single selection should call the single discard endpoint: {discard_calls}"
    method, path = discard_calls[0]
    assert method == "POST"
    assert path.endswith("/discard")
    assert "discard_batch" not in path

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
