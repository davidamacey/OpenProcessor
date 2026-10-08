"""K2 (docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md):
the served keymap applies to real keypresses end to end.

  1. `test_keymap_absent_when_404` — the default stub (conftest.py's
     shared `GET {prefix}/keymap` 404) means `/settings` shows no
     "Keyboard shortcuts" section, and the default key (`d`) still
     discards on `/review`.
  2. `test_keymap_rebind_persists_and_applies_live` — a served keymap
     document is rebound on `/settings` (Discard: d -> x), saved through
     a real validate+PUT round trip with the exact request bodies
     asserted, then on `/review` pressing `x` discards the current item
     and `d` does nothing.
  3. `test_keymap_per_context_override` (K2b, plan §0 decision 4 + §5.4):
     detaching `cluster.discard` from the `discard` group on `/settings`
     — leaving `review.queue.discard` at its default `d` — writes only
     `cluster.discard` in the PUT body, then `/review` still discards on
     `d` while `/clusters/[id]` discards on the new key and ignores `d`.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint
from playwright.sync_api import expect

import copy
import json
import re

from fixtures.wire import make_item

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 10,
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
        class_name="widget",
        proposed_class_id=1,
        proposed_class_name="widget",
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )
    item["reason"] = "uncertainty"
    return item


def _base_keymap_actions() -> list[dict]:
    """A minimal-but-real subset of the served action list (plan §4.2) —
    just enough for `/settings`'s editor and `/review`'s discard/skip to
    round-trip; a real backend serves all 46."""
    return [
        {
            "id": "global.shortcuts_overlay",
            "context": "global",
            "group": None,
            "label": "Toggle this panel",
            "description": "",
            "default": ["~", "`", "shift+~"],
            "keys": ["~", "`", "shift+~"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "global.close_overlay",
            "context": "global",
            "group": "cancel",
            "label": "Close this panel / cancel",
            "description": "",
            "default": ["escape"],
            "keys": ["escape"],
            "modifiable": False,
            "available": True,
            "locked_keys": ["escape"],
        },
        {
            "id": "review.skip",
            "context": "review",
            "group": "skip",
            "label": "Skip",
            "description": "",
            "default": ["n"],
            "keys": ["n"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "review.undo",
            "context": "review",
            "group": "undo",
            "label": "Undo last",
            "description": "",
            "default": ["z"],
            "keys": ["z"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "review.queue.confirm",
            "context": "review.queue",
            "group": "confirm",
            "label": "Confirm proposed & advance (or search classes if blank)",
            "description": "",
            "default": ["enter"],
            "keys": ["enter"],
            "modifiable": True,
            "available": True,
            "locked_keys": ["enter"],
        },
        {
            "id": "review.queue.discard",
            "context": "review.queue",
            "group": "discard",
            "label": "Discard",
            "description": "",
            "default": ["d"],
            "keys": ["d"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "review.queue.class_picker",
            "context": "review.queue",
            "group": None,
            "label": "Search all classes…",
            "description": "",
            "default": ["/"],
            "keys": ["/"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "review.queue.prev",
            "context": "review.queue",
            "group": "prev",
            "label": "Previous item",
            "description": "",
            "default": ["arrowleft"],
            "keys": ["arrowleft"],
            "modifiable": True,
            "available": True,
            "locked_keys": ["arrowleft"],
        },
        {
            "id": "review.queue.next",
            "context": "review.queue",
            "group": "next",
            "label": "Next item",
            "description": "",
            "default": ["arrowright"],
            "keys": ["arrowright"],
            "modifiable": True,
            "available": True,
            "locked_keys": ["arrowright"],
        },
        {
            "id": "cluster.discard",
            "context": "cluster",
            "group": "discard",
            "label": "Discard selected",
            "description": "",
            "default": ["d"],
            "keys": ["d"],
            "modifiable": True,
            "available": True,
        },
        {
            "id": "cluster.confirm",
            "context": "cluster",
            "group": "confirm",
            "label": "Confirm selected & advance",
            "description": "",
            "default": ["enter"],
            "keys": ["enter"],
            "modifiable": True,
            "available": True,
            "locked_keys": ["enter"],
        },
    ]


def served_keymap_doc(revision: int = 1) -> dict:
    return {
        "scope": "project",
        "project": "default",
        "revision": revision,
        "etag": f"keymap:{revision}",
        "is_default": True,
        "updated_at": None,
        "grammar": {
            "locked_keys": ["escape", "enter", "arrowleft", "arrowright", "arrowup", "arrowdown"],
            "max_combos_per_action": 3,
        },
        "contexts": [
            {"id": "global", "label": "Everywhere", "description": "", "includes": [], "class_hotkeys_live": True},
            {"id": "review", "label": "Review (every tab)", "description": "", "includes": ["global"], "class_hotkeys_live": True},
            {"id": "review.queue", "label": "Review queue", "description": "", "includes": ["review"], "class_hotkeys_live": True},
            {"id": "cluster", "label": "Cluster", "description": "", "includes": ["global"], "class_hotkeys_live": True},
        ],
        "actions": _base_keymap_actions(),
        "overrides": {},
        "reserved_hotkeys": ["d", "n", "z", "/"],
        "issues": [],
    }


def register_common(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})


def test_keymap_absent_when_404(stub, page, app_url):
    register_common(stub)
    # The shared default stub (conftest.py) already serves a 404 for
    # `GET {prefix}/keymap` — no override needed to exercise that path.

    page.goto(f"{app_url}/p/default/settings", wait_until="domcontentloaded")
    page.wait_for_selector("text=Curation scores", timeout=ACTION_TIMEOUT_MS)
    # Real wait for the page to finish firing its on-mount requests
    # (including the 404'd GET /keymap this assertion depends on) instead
    # of an arbitrary settle sleep.
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert page.locator("text=Keyboard shortcuts").count() == 0

    dismiss_calls: list[str] = []

    def review_handler(_request, _match):
        return (200, {"items": [review_item(0)], "total": 1, "page": 1, "page_size": 30})

    def dismiss_handler(request, match):
        dismiss_calls.append(match.string)
        return (200, review_item(0))

    stub.on("GET", r"/review/all(\?|$)", review_handler)
    stub.on(
        "POST",
        r"/crops/[^/]+/review_dismiss$",
        dismiss_handler,
    )

    page.goto(f"{app_url}/p/default/review", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    # The hint strip renders from the same registrations the key handler
    # uses, so once it shows the discard hint the key is live.
    page.wait_for_selector("text=discard", timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/review_dismiss" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press("d")
    assert len(dismiss_calls) == 1

    stub.assert_fail_closed()


def test_keymap_rebind_persists_and_applies_live(stub, page, app_url):
    register_common(stub)

    doc_holder = {"doc": served_keymap_doc(revision=1)}
    put_calls: list[dict] = []
    validate_calls: list[dict] = []

    def keymap_get(_request, _match):
        return (200, doc_holder["doc"])

    def keymap_validate(request, _match):
        body = json.loads(request.post_data or "{}")
        validate_calls.append(body)
        return (200, {"ok": True, "errors": [], "warnings": [], "force_allowed": False})

    def keymap_put(request, _match):
        body = json.loads(request.post_data or "{}")
        put_calls.append(body)
        assert body["expected_revision"] == doc_holder["doc"]["revision"]
        new_doc = copy.deepcopy(doc_holder["doc"])
        new_doc["revision"] += 1
        new_doc["is_default"] = False
        overrides = body["overrides"]
        for action in new_doc["actions"]:
            if action["id"] in overrides:
                action["keys"] = overrides[action["id"]]
        doc_holder["doc"] = new_doc
        return (200, new_doc)

    stub.on("GET", r"/keymap(\?|$)", keymap_get)
    stub.on("POST", r"/keymap/validate(\?|$)", keymap_validate)
    stub.on("PUT", r"/keymap(\?|$)", keymap_put)

    page.goto(f"{app_url}/p/default/settings", wait_until="domcontentloaded")
    page.wait_for_selector("text=Keyboard shortcuts", timeout=ACTION_TIMEOUT_MS)

    # Find the Discard row, remove the default 'd' key, add 'x' instead.
    discard_row = page.locator("tr", has_text="Discard").first
    discard_row.get_by_role("button", name="remove d").click()
    expect(discard_row.get_by_role("button", name="remove d")).to_have_count(
        0, timeout=ACTION_TIMEOUT_MS
    )
    discard_row.get_by_role("button", name="Change").click()
    page.wait_for_selector("[data-capture]", timeout=ACTION_TIMEOUT_MS)
    # The validate call is debounced (~350ms) — wait for the real request
    # instead of guessing a duration past the debounce.
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/keymap/validate" in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press("x")

    assert any("review.queue.discard" in v.get("overrides", {}) for v in validate_calls)

    page.get_by_role("button", name="Save", exact=True).first.click()
    page.wait_for_selector("text=Save keyboard shortcuts?", timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith("/keymap"), timeout=ACTION_TIMEOUT_MS
    ):
        page.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()

    assert len(put_calls) == 1
    assert put_calls[0]["overrides"]["review.queue.discard"] == ["x"]
    assert put_calls[0]["expected_revision"] == 1

    # Now confirm the new key applies live on /review, and the old one
    # does nothing.
    dismiss_calls: list[str] = []

    def review_handler(_request, _match):
        return (200, {"items": [review_item(0)], "total": 1, "page": 1, "page_size": 30})

    def dismiss_handler(request, match):
        dismiss_calls.append(match.string)
        return (200, review_item(0))

    stub.on("GET", r"/review/all(\?|$)", review_handler)
    stub.on("POST", r"/crops/[^/]+/review_dismiss$", dismiss_handler)

    page.goto(f"{app_url}/p/default/review", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    # The hint strip's own key glyph proves the rebound keymap has loaded
    # before either key is tested below.
    page.wait_for_selector("kbd:has-text('X')", timeout=ACTION_TIMEOUT_MS)

    page.keyboard.press("d")
    # 'd' is a no-op now; let its (non-)handling settle onto the DOM/event
    # loop via a real paint tick rather than an arbitrary sleep before
    # checking the negative.
    wait_for_paint(page)
    assert len(dismiss_calls) == 0, "the old key must no longer discard"

    with page.expect_response(
        lambda r: r.request.method == "POST" and "/review_dismiss" in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press("x")
    assert len(dismiss_calls) == 1

    stub.assert_fail_closed()


CLUSTER_CLASSES = [
    {"id": 1, "name": "widget", "group": None, "hotkey_letter": None, "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

CLUSTER_LIST = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 3,
            "validated_count": 0,
            "dominant_class_id": 1,
            "dominant_class_name": "widget",
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


def cluster_crop(i: int) -> dict:
    return make_item(
        crop_id=f"ccrop-{i}",
        image_id=f"cimg-{i}",
        class_id=1,
        class_name="widget",
        cluster_id=1,
        label_validated=False,
        label_source="model_suggestion",
        thumbnail_url=f"/curation/crops/ccrop-{i}/thumbnail",
    )


def test_keymap_per_context_override(stub, page, app_url):
    """K2b: detach `cluster.discard` from the `discard` group on
    `/settings`, leaving `review.queue.discard` at its default `d`.
    Saving writes only `cluster.discard` in the PUT body; afterwards
    `/review` still discards on `d`, and `/clusters/[id]` discards on the
    new key and ignores `d`.
    """
    register_common(stub)

    doc_holder = {"doc": served_keymap_doc(revision=1)}
    put_calls: list[dict] = []

    def keymap_get(_request, _match):
        return (200, doc_holder["doc"])

    def keymap_validate(_request, _match):
        return (200, {"ok": True, "errors": [], "warnings": [], "force_allowed": False})

    def keymap_put(request, _match):
        body = json.loads(request.post_data or "{}")
        put_calls.append(body)
        new_doc = copy.deepcopy(doc_holder["doc"])
        new_doc["revision"] += 1
        new_doc["is_default"] = False
        overrides = body["overrides"]
        for action in new_doc["actions"]:
            if action["id"] in overrides:
                action["keys"] = overrides[action["id"]]
        doc_holder["doc"] = new_doc
        return (200, new_doc)

    stub.on("GET", r"/keymap(\?|$)", keymap_get)
    stub.on("POST", r"/keymap/validate(\?|$)", keymap_validate)
    stub.on("PUT", r"/keymap(\?|$)", keymap_put)

    page.goto(f"{app_url}/settings", wait_until="domcontentloaded")
    page.wait_for_selector("text=Keyboard shortcuts", timeout=ACTION_TIMEOUT_MS)

    # Expand the 'discard' verb group's "Customize per page" disclosure
    # and rebind only the cluster.discard member row.
    group_details = page.locator('[data-testid="group-discard"]')
    group_details.get_by_text("Customize per page").click()
    member_row = page.locator('[data-testid="member-cluster.discard"]')
    member_row.get_by_role("button", name="remove d").click()
    expect(member_row.get_by_role("button", name="remove d")).to_have_count(
        0, timeout=ACTION_TIMEOUT_MS
    )
    member_row.get_by_role("button", name="Change").click()
    page.wait_for_selector("[data-capture]", timeout=ACTION_TIMEOUT_MS)
    # The validate call is debounced (~350ms) — wait for the real request.
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/keymap/validate" in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press("x")

    # The detached marker shows on the cluster row, not the review row.
    assert page.locator('[data-testid="detached-cluster.discard"]').count() == 1
    assert page.locator('[data-testid="detached-review.queue.discard"]').count() == 0

    page.get_by_role("button", name="Save", exact=True).first.click()
    page.wait_for_selector("text=Save keyboard shortcuts?", timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith("/keymap"), timeout=ACTION_TIMEOUT_MS
    ):
        page.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()

    assert len(put_calls) == 1
    assert put_calls[0]["overrides"]["cluster.discard"] == ["x"]
    assert "review.queue.discard" not in put_calls[0]["overrides"]

    # /review still discards on the group's default 'd'.
    dismiss_calls: list[str] = []

    def review_handler(_request, _match):
        return (200, {"items": [review_item(0)], "total": 1, "page": 1, "page_size": 30})

    def dismiss_handler(request, match):
        dismiss_calls.append(match.string)
        return (200, review_item(0))

    stub.on("GET", r"/review/all(\?|$)", review_handler)
    stub.on("POST", r"/crops/[^/]+/review_dismiss$", dismiss_handler)

    page.goto(f"{app_url}/review", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/review_dismiss" in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press("d")
    assert len(dismiss_calls) == 1, "the untouched context must keep the group's default key"

    # /clusters/[id] discards on the new key ('x') and ignores 'd'.
    discard_calls: list[tuple[str, str]] = []

    def discard_handler(request, match):
        discard_calls.append((request.method, match.string))
        return (200, cluster_crop(1))

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLUSTER_CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTER_LIST)
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [cluster_crop(i) for i in range(3)]})
    stub.on("POST", r"/crops/([^/]+)/discard$", discard_handler)
    stub.on("POST", r"/crops/discard_batch$", lambda req, m: (200, {"items": [], "discarded": 0, "conflicts": [], "not_found": []}))

    page.goto(f"{app_url}/clusters/1", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.locator("img").nth(1).click()
    expect(page.get_by_text("1 selected").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    page.keyboard.press("d")
    # 'd' is a no-op in this detached context; let it settle via a real
    # paint tick, then prove the app is still live and listening with the
    # real key below rather than sleeping and hoping.
    wait_for_paint(page)
    assert len(discard_calls) == 0, "the detached context's old default key must no longer discard"

    with page.expect_response(
        lambda r: r.request.method == "POST" and "/discard" in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press("x")
    assert len(discard_calls) == 1

    stub.assert_fail_closed()
