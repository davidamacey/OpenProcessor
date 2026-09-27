"""New coverage (docs/design/test-audit-2026-09-24.md recommendation 5):
`/review` Enter-assign followed by Z-undo.

Enter -> confirmAndAdvance() -> assign(classId) -> `PUT
{API_PREFIX}/crops/{id}/label` (src/routes/review/+page.svelte).
Z -> undoLast() -> `POST {API_PREFIX}/crops/{id}/label/undo`
(src/lib/api.ts undoCropLabel).
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from fixtures.wire import make_item

CLASSES = [
    {"class_id": 1, "class_name": "ducati", "kind": "item", "group": "moto", "hotkey_letter": "k", "sample_count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}


def review_item(i: int) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="ducati",
        proposed_class_id=1,
        proposed_class_name="ducati",
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )
    item["reason"] = "uncertainty"
    return item


def test_review_assign_then_undo(stub, page, app_url):
    label_calls: list[tuple[str, str, dict]] = []
    undo_calls: list[tuple[str, str]] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    # The detail lightbox/hover path fetches a crop's full image + siblings.
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        lambda req, m: (200, {"image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480}, "items": [review_item(0)]}),
    )

    def review_handler(_request, _match):
        items = [review_item(i) for i in range(3)]
        return (200, {"items": items, "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)

    def label_handler(request, match):
        label_calls.append((request.method, match.string, request.post_data_json or {}))
        item = review_item(0)
        item["class_id"] = (request.post_data_json or {}).get("class_id")
        return (200, item)

    stub.on("PUT", r"/crops/([^/]+)/label$", label_handler)

    def undo_handler(request, match):
        undo_calls.append((request.method, match.string))
        item = review_item(0)
        item["proposed_class_id"] = 1
        item["proposed_class_name"] = "ducati"
        item["label_validated"] = False
        return (200, item)

    stub.on("POST", r"/crops/([^/]+)/label/undo$", undo_handler)

    page.goto(f"{app_url}/p/default/review")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    before = counter.first.inner_text()

    page.keyboard.press("Enter")
    page.wait_for_timeout(500)

    assert len(label_calls) == 1, f"Enter should PUT exactly one label: {label_calls}"
    method, path, body = label_calls[0]
    assert method == "PUT"
    assert path.endswith("/label")
    assert body.get("class_id") == 1, body

    after_label = counter.first.inner_text()
    assert after_label != before, "the queue should visibly advance after a successful assign"

    page.keyboard.press("z")
    page.wait_for_timeout(500)

    assert len(undo_calls) == 1, f"Z should POST exactly one .../label/undo: {undo_calls}"
    undo_method, undo_path = undo_calls[0]
    assert undo_method == "POST"
    assert undo_path.endswith("/label/undo")
    assert "crop-0" in undo_path, undo_path

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the assign/undo flow: {errors[:3]}"
