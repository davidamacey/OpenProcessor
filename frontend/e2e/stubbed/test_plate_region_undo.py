"""M6 (docs/design/interactive-pass-2026-09-24.md §6, BOTH — now FIXED):
once a Plates-tab write lands (confirm/reject/false-positive/box-edit),
Z must reverse it server-side via `POST {API_PREFIX}/crops/{id}/region/undo`
(backend: OpenProcessor main `07cc061`) — distinct from the pre-existing
"← step back" action, which only re-queues the crop locally without
touching what the server saved.

Repro this proves is fixed: press D (reject) on the Plates tab, then
press Z. Before this fix there was no Z handler on the slot tabs at all
(the label-undo Z ran but had nothing recorded, since a region write
never pushed onto `undoStore`) — `POST .../region/undo` was never called.
"""

from __future__ import annotations

from fixtures.wire import make_item

CLASSES = [
    {"id": 1, "name": "license_plate", "group": "vehicle", "hotkey_letter": "l", "count": 40, "validated_count": 12, "cluster_size": 44, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}


def plate_item() -> dict:
    return make_item(
        crop_id="plate-1",
        image_id="img-1",
        class_id=1,
        class_name="license_plate",
        thumbnail_url="/curation/crops/plate-1/thumbnail",
    )


def test_plate_reject_then_z_calls_region_undo(stub, page, app_url):
    region_calls: list[dict] = []
    undo_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(_request, _match):
        return (200, {"items": [plate_item()], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/", review_handler)

    def region_handler(request, match):
        region_calls.append(request.post_data_json or {})
        return (200, {"item": plate_item()})

    stub.on("PUT", r"/crops/([^/]+)/region$", region_handler)

    meta_calls: list[dict] = []

    def meta_handler(request, match):
        meta_calls.append(request.post_data_json or {})
        return (200, {"item": plate_item()})

    stub.on("PATCH", r"/crops/([^/]+)/region_meta$", meta_handler)

    # Registered AFTER the bare PUT handler above (not that it matters —
    # different method/path) so it's unambiguous either way.
    def region_undo_handler(request, match):
        undo_calls.append(match.string)
        return (200, plate_item())

    stub.on("POST", r"/crops/([^/]+)/region/undo$", region_undo_handler)

    page.goto(f"{app_url}/review?tab=plates")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=15000)
    page.wait_for_timeout(500)

    # D = reject on the Plates tab -> ONE PATCH region_meta carrying the
    # served reject status (the backend clears the box), so one Z undoes it.
    page.keyboard.press("d")
    page.wait_for_timeout(400)

    assert len(meta_calls) == 1, f"D should PATCH region_meta exactly once: {meta_calls}"
    assert meta_calls[0].get("region_status"), f"reject must send a status: {meta_calls}"
    assert region_calls == [], f"reject must not also PUT region: {region_calls}"

    # Z must now reverse that write server-side.
    page.keyboard.press("z")
    page.wait_for_timeout(400)

    assert len(undo_calls) == 1, (
        f"Z on the Plates tab must POST {{API_PREFIX}}/crops/{{id}}/region/undo "
        f"exactly once: {undo_calls}"
    )
    assert "plate-1" in undo_calls[0], undo_calls

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the plate-reject-then-undo flow: {errors[:3]}"


def test_plate_reject_z_before_any_write_does_not_call_region_undo(stub, page, app_url):
    """Sanity check on the shared undoStore: Z with nothing recorded (no
    slot write yet this session) must not fire a region/undo request —
    the store's own 'Nothing to undo' early return, unchanged by this
    finding, still applies to the new `kind: 'region'` entries."""
    undo_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))
    stub.on("GET", r"/review/", lambda *_: (200, {"items": [plate_item()], "total": 1, "page": 1, "page_size": 30}))
    stub.on("POST", r"/crops/([^/]+)/region/undo$", lambda request, match: undo_calls.append(match.string) or (200, plate_item()))

    page.goto(f"{app_url}/review?tab=plates")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=15000)
    page.wait_for_timeout(500)

    page.keyboard.press("z")
    page.wait_for_timeout(300)

    assert undo_calls == [], f"Z with nothing recorded must not call region/undo: {undo_calls}"
