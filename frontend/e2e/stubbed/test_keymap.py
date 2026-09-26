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
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import copy
import json
import re

from fixtures.wire import make_item

CLASSES = [
    {
        "id": 1,
        "name": "widget",
        "group": None,
        "hotkey_letter": None,
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
        ],
        "actions": _base_keymap_actions(),
        "overrides": {},
        "reserved_hotkeys": ["d", "n", "z", "/"],
        "issues": [],
    }


def register_common(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/scores/coverage(\?|$)", (404, {"detail": "not found"}))


def test_keymap_absent_when_404(stub, page, app_url):
    register_common(stub)
    # The shared default stub (conftest.py) already serves a 404 for
    # `GET {prefix}/keymap` — no override needed to exercise that path.

    page.goto(f"{app_url}/settings", wait_until="domcontentloaded")
    page.wait_for_selector("text=Curation scores", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(200)
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

    page.goto(f"{app_url}/review", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(200)
    page.keyboard.press("d")
    page.wait_for_timeout(200)
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

    page.goto(f"{app_url}/settings", wait_until="domcontentloaded")
    page.wait_for_selector("text=Keyboard shortcuts", timeout=ACTION_TIMEOUT_MS)

    # Find the Discard row, remove the default 'd' key, add 'x' instead.
    discard_row = page.locator("tr", has_text="Discard").first
    discard_row.get_by_role("button", name="remove d").click()
    page.wait_for_timeout(100)
    discard_row.get_by_role("button", name="Change").click()
    page.wait_for_selector("[data-capture]", timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press("x")
    page.wait_for_timeout(400)  # validate debounce

    assert any("review.queue.discard" in v.get("overrides", {}) for v in validate_calls)

    page.get_by_role("button", name="Save", exact=True).first.click()
    page.wait_for_selector("text=Save keyboard shortcuts?", timeout=ACTION_TIMEOUT_MS)
    page.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()
    page.wait_for_timeout(200)

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

    page.goto(f"{app_url}/review", wait_until="domcontentloaded")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(200)

    page.keyboard.press("d")
    page.wait_for_timeout(200)
    assert len(dismiss_calls) == 0, "the old key must no longer discard"

    page.wait_for_selector("kbd:has-text('X')", timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press("x")
    page.wait_for_timeout(200)
    assert len(dismiss_calls) == 1

    stub.assert_fail_closed()
