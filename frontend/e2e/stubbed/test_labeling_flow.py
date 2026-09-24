"""Ported from scripts/playwright_labeling_flow.py (deleted; see
docs/design/test-audit-2026-09-24.md recommendation 5 / P1-1).

Drives the real UI against the fail-closed `stub` fixture (conftest.py) so
it exercises: 1.1 class-letter hotkey labels the SELECTION; 1.2 Shift+N
flags the selection for a new class; 1.3 a failed label puts the review
item back in the queue; 1.9 Escape after a drag does not blow the stack;
every route still mounts; the Add-Class modal's backdrop-vs-inside-click
behavior.
"""

from __future__ import annotations

import re

from fixtures.wire import make_item

ROUTES = [
    "/",
    "/dashboard",
    "/clusters",
    "/clusters/1",
    "/review",
    "/classes",
    "/export",
    "/models",
    "/train",
    "/bakeoff",
]

CLASSES = [
    {
        "id": 1,
        "name": "ducati",
        "group": "moto",
        "hotkey_letter": "k",
        "count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "brand_a",
        "group": "moto",
        "hotkey_letter": "j",
        "count": 8,
        "validated_count": 3,
        "cluster_size": 9,
        "deprecated": False,
    },
]


def crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        image_path=f"/nas/img-{i}.jpg",
        class_id=1,
        class_name="ducati",
        class_source="v6_model",
        confidence=0.9,
        cluster_id=1,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


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


def register_base(stub, *, fail_put_label: bool = False) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/models/status", {"models": []})

    def label_handler(_request, _match):
        if fail_put_label:
            return (422, {"detail": "stub: label rejected"})
        return (200, {"ok": True})

    def batch_label_handler(req, _m):
        ids = (_json(req) or {}).get("crop_ids", [])
        return (200, {"updated": len(ids), "updated_ids": ids, "conflicts": []})

    stub.on("PUT", r"/crops/[^/]+/label$", label_handler)
    stub.on("PUT", r"/crops/batch_label$", batch_label_handler)
    stub.on(
        "POST",
        r"/crops/flag_new_class$",
        lambda req, m: (200, {"flagged": len((_json(req) or {}).get("crop_ids", [])), "errors": 0}),
    )
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [crop(i) for i in range(3)]})
    stub.on("GET", r"/crops/[^/]+/image$", lambda req, m: (200, {"image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480}, "items": [crop(0)]}))

    # The "every route still mounts" loop below visits every top-level
    # route, each of which fires its own on-mount GETs — stubbed with
    # harmless empty/idle shapes since none of their content is asserted
    # on, only that the route renders without an unhandled request or a
    # page error.
    stub.on("GET", r"/pipeline/auto_label/status", {
        "job_id": "job-0", "status": "idle", "stage": "", "processed": 0, "total": 0,
        "started_at": 0, "finished_at": 0, "error": None, "result": {}, "args": {},
        "eta_seconds": None, "elapsed_seconds": 0,
    })
    stub.on("GET", r"/stats/dataset(\?|$)", {"total_crops": 3, "validated": 0, "test_holdout": 0, "by_source": {}})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": {}})
    stub.on("GET", r"/export/status(\?|$)", {"status": "idle", "last_run": None})
    stub.on("GET", r"/export/datasets(\?|$)", {"datasets": []})
    stub.on("GET", r"/train/status(\?|$)", {"jobs": []})
    stub.on("GET", r"/train/runs(\?|$)", {"items": [], "total": 0})
    stub.on("GET", r"/train/profiles(\?|$)", {"profiles": []})
    stub.on("GET", r"/train/presets(\?|$)", {"class_subset_presets": []})
    stub.on("GET", r"/train/gpus(\?|$)", {"options": [], "allowed_ids": [], "unrestricted": True})
    stub.on("GET", r"/training_cohorts(\?|$)", {"cohorts": []})
    stub.on("GET", r"/bakeoff/profiles(\?|$)", {"profiles": []})
    stub.on("GET", r"/bakeoff/eval_datasets(\?|$)", {"datasets": []})
    stub.on("GET", r"/bakeoff/trained_models(\?|$)", {"models": []})
    stub.on("GET", r"/bakeoff/baseline_models(\?|$)", {"models": []})

    def review_handler(_request, _match):
        items = []
        for i in range(3):
            c = crop(i)
            c["reason"] = "uncertainty"
            c["proposed_class_id"] = 1
            c["proposed_class_name"] = "ducati"
            items.append(c)
        return (200, {"items": items, "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)


def _json(request):
    import json

    if not request.post_data:
        return None
    try:
        return json.loads(request.post_data)
    except (ValueError, TypeError):
        return None


def test_labeling_flow(stub, page, app_url):
    register_base(stub)

    # ---- cluster detail page ------------------------------------
    page.goto(f"{app_url}/clusters/1")
    page.wait_for_selector("img", timeout=15000)
    page.wait_for_timeout(600)
    assert page.locator("article, li, div").count() > 0, "grid did not render"

    # 1.1 — with nothing ever dragged, a class letter must label the selection.
    page.locator("body").click(position={"x": 5, "y": 5})
    page.locator("img").nth(1).click()
    page.wait_for_timeout(150)
    stub.calls.clear()
    page.keyboard.press("k")
    page.wait_for_timeout(400)
    labeled = [c for c in stub.calls if "batch_label" in c[1] or c[1].endswith("/label")]
    assert len(labeled) == 1, f"1.1 class hotkey should label exactly the selection: calls={stub.calls}"

    # 1.2 — Shift+N flags for a new class (plain N must not).
    page.locator("img").nth(1).click()
    page.wait_for_timeout(150)
    stub.calls.clear()
    page.keyboard.press("Shift+N")
    page.wait_for_timeout(400)
    flagged = [c for c in stub.calls if "flag_new_class" in c[1]]
    assert len(flagged) == 1, f"1.2 Shift+N should flag for new class: calls={stub.calls}"

    stub.calls.clear()
    page.keyboard.press("n")
    page.wait_for_timeout(300)
    assert not any("flag_new_class" in c[1] for c in stub.calls), "1.2 plain N must not flag"

    # 1.9 — Escape during/after a real pointer drag must not blow the stack.
    card = page.locator("img").nth(1)
    box = card.bounding_box()
    assert box is not None
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    for dx in (6, 18, 40, 70):
        page.mouse.move(cx + dx, cy + dx // 2)
        page.wait_for_timeout(60)
    page.keyboard.press("Escape")  # during the drag
    page.wait_for_timeout(200)
    page.mouse.up()
    page.wait_for_timeout(300)
    for _ in range(3):  # and again after it completed
        page.keyboard.press("Escape")
        page.wait_for_timeout(80)
    overflow = [
        c
        for c in stub.console_errors
        if "Maximum call stack" in c or "RangeError" in c or "too much recursion" in c
    ]
    assert not overflow, f"1.9 Escape during/after a drag must not overflow the stack: {overflow[:2]}"

    # ---- review page --------------------------------------------
    register_base(stub, fail_put_label=True)
    page.goto(f"{app_url}/review")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=15000)
    page.wait_for_timeout(700)
    before = counter.first.inner_text()
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    after = counter.first.inner_text()
    assert before.strip() == after.strip(), f"1.3 failed label must return the item to the queue: before={before!r} after={after!r}"
    assert page.get_by_text(re.compile("Label failed", re.I)).count() > 0, "1.3 failure toast missing"
    assert page.get_by_text(re.compile("stub: label rejected")).count() > 0, "1.8 toast must carry the server detail"

    review_errors = [c for c in stub.console_errors if c.startswith("pageerror") or "Uncaught" in c]
    assert not review_errors, f"no uncaught review page errors expected: {review_errors[:3]}"
    register_base(stub, fail_put_label=False)

    # ---- every route still mounts --------------------------------
    for route in ROUTES:
        stub.console_errors.clear()
        page.goto(f"{app_url}{route}")
        page.locator("main").first.wait_for(timeout=15000)
        page.wait_for_timeout(300)
        crashed = [c for c in stub.console_errors if c.startswith("pageerror") or "Uncaught" in c]
        body = page.locator("main").first.inner_text()
        assert not crashed and len(body.strip()) > 0, (
            f"{route} should mount without errors: errors={crashed[:2]} body_len={len(body.strip())}"
        )

    # ---- modal backdrops -----------------------------------------
    page.goto(f"{app_url}/classes")
    page.wait_for_timeout(400)
    page.get_by_role("button", name="+ Add Class").first.click()
    dialog = page.get_by_role("dialog", name="Add class")
    assert dialog.count() > 0, "Add Class modal should open"
    dialog.get_by_text("Add Class").first.click()
    page.wait_for_timeout(250)
    assert dialog.count() > 0, "click inside the panel should keep the modal open"
    box = dialog.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + 6, box["y"] + 6)  # backdrop corner
    page.wait_for_timeout(300)
    assert dialog.count() == 0, "click on the backdrop should close the modal"

    page.get_by_role("button", name="+ Add Class").first.click()
    page.wait_for_timeout(200)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert page.get_by_role("dialog", name="Add class").count() == 0, "Escape should close the modal"

    modal_errors = [c for c in stub.console_errors if c.startswith("pageerror") or "Uncaught" in c]
    assert not modal_errors, f"no uncaught errors in the modal flow expected: {modal_errors[:3]}"
