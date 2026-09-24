"""DQ-M5 (docs/design/data-quality-pass-2026-09-24.md): the review crop
was upscaled to the full column height (`h-full w-full object-contain`,
phase-A p9's fit-width fix applied to height too), pushing Reason,
Proposed and Confirm/Skip/Discard below the fold at 1280×720, 1600×1000
and 1920×1080.

Fixed in src/routes/review/+page.svelte: the crop panel now carries a
`max-h-[46%]` ceiling alongside its existing floor, and the <img> itself
is capped to at most 4x its natural pixel size
(src/lib/review/cropDisplaySize.ts) — so no crop, however small, can
consume the whole panel.

This proves the container-cap half of the fix at 1280×720 (pytest-
playwright's default viewport, set explicitly here so it isn't silently
dependent on that default): the Confirm/Skip/Discard buttons must be
fully inside the viewport with no scrolling.
"""

from __future__ import annotations

from fixtures.wire import make_item

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


def test_review_actions_stay_within_1280x720_viewport(stub, page, app_url):
    page.set_viewport_size({"width": 1280, "height": 720})

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on(
        "GET",
        r"/crops/[^/]+/image$",
        lambda req, m: (
            200,
            {
                "image": {"path": "/nas/img-0.jpg", "width": 640, "height": 480},
                "items": [review_item(0)],
            },
        ),
    )

    def review_handler(_request, _match):
        items = [review_item(i) for i in range(3)]
        return (200, {"items": items, "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)

    page.goto(f"{app_url}/review")

    confirm_btn = page.get_by_role("button", name="Confirm", exact=True)
    confirm_btn.wait_for(timeout=15000)
    skip_btn = page.get_by_role("button", name="Skip", exact=True)
    discard_btn = page.get_by_role("button", name="Discard", exact=True)

    viewport_height = 720
    for btn, label in ((confirm_btn, "Confirm"), (skip_btn, "Skip"), (discard_btn, "Discard")):
        box = btn.bounding_box()
        assert box is not None, f"{label} button has no layout box"
        bottom = box["y"] + box["height"]
        assert bottom <= viewport_height, (
            f"{label} button bottom edge at {bottom}px is below the "
            f"{viewport_height}px viewport (DQ-M5 regression): {box}"
        )
        assert box["y"] >= 0, f"{label} button top edge above the viewport: {box}"

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected: {errors[:3]}"
