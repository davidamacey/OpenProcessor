"""OpenProcessor W3 (prompt-pack CRUD) and the pack half of W5
(test-on-crop), any_domain_plan.md §3, §5.1, §7.2, §7.5, §7.6;
docs/design/w3-pack-editor-ui-plan-2026-09-27.md.

The backend has not shipped W3 yet, so every payload below follows the
spec's own examples (in the neutral widget/tag domain). The fail-closed
stub means any pack request the page makes that a test did not expect
fails the test.

  1. `test_edit_validate_save_and_resolve_a_conflict` — a served issue
     lands under its field after the debounced validate; Save sends
     `expected_revision`; a 409 `revision_conflict` offers "Keep my edits",
     and the next Save claims the served revision.
  2. `test_activate_needs_force_only_when_the_server_allows_it` — a 422
     with `force_allowed` offers "Activate anyway"; the second body has
     `force: true` and the pinned revision.
  3. `test_rollback_from_the_list` — confirm, `expected_active` from the
     last read.
  4. `test_test_on_crop_sends_the_draft_and_shows_the_reply`.
  5. `test_clone_a_template_opens_the_new_pack`.
  6. `test_absent_when_prompt_packs_404` — conftest's default 404: no
     /settings card, the route says so, no other pack request fires.
"""

from __future__ import annotations

import copy
from typing import Any

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

CLEAN = {"ok": True, "errors": [], "warnings": [], "force_allowed": False}

SCHEMA: dict[str, Any] = {
    "fields": [
        {
            "field": "class_system",
            "label": "Classify: system message",
            "group": "classify",
            "kind": "text",
            "formatted": False,
            "required_placeholders": [],
            "allowed_placeholders": [],
            "expected_reply_keys": [],
            "optional_reply_keys": [],
            "used_by": ["auto_label_vlm_stage"],
            "help": "Sent verbatim as the system message.",
        },
        {
            "field": "class_user_template",
            "label": "Classify: user message",
            "group": "classify",
            "kind": "text",
            "formatted": True,
            "required_placeholders": ["class_names_csv"],
            "allowed_placeholders": ["class_names_csv"],
            "expected_reply_keys": ["img", "class", "confidence"],
            "optional_reply_keys": [],
            "used_by": ["auto_label_vlm_stage", "vlm_label_batch"],
            "help": "Asks for one class per image.",
        },
        {
            "field": "synonyms",
            "label": "Synonyms",
            "group": "vocabulary",
            "kind": "map",
            "formatted": False,
            "required_placeholders": [],
            "allowed_placeholders": [],
            "expected_reply_keys": [],
            "optional_reply_keys": [],
            "used_by": ["auto_label_vlm_stage"],
            "help": "Maps a word the VLM may answer to a registry class.",
        },
    ],
    "placeholders": [
        {
            "name": "class_names_csv",
            "meaning": "comma-separated class names from the registry",
            "example": "widget, gadget",
        }
    ],
    "calls": [
        {"id": "classify", "label": "Classify", "fields": ["class_system", "class_user_template"], "testable": True}
    ],
    "reply_key_contract": {"classify": {"required": ["img", "class", "confidence"], "optional": []}},
}

BODY = {
    "class_system": "You classify widgets.",
    "class_user_template": "Pick one of: {class_names_csv}",
    "synonyms": {"doohickey": "gadget"},
}


def doc(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "widget_tag",
        "source": "stored",
        "read_only": False,
        "revision": 2,
        "etag": "prompt_pack:widget_tag:2",
        "description": "Tags on widgets",
        "body": copy.deepcopy(BODY),
        "created_at": "2026-09-26T10:00:00Z",
        "updated_at": "2026-09-26T12:00:00Z",
        "updated_by": None,
        "cloned_from": "template:widget_tag@-",
        "active": True,
        "active_revision": 1,
        "validation": CLEAN,
    }
    base.update(over)
    return base


def active(name: str | None = "widget_tag", revision: int | None = 1, previous: Any = None) -> dict[str, Any]:
    return {
        "axis": "prompt_pack",
        "active": {"name": name, "revision": revision},
        "source": "stored",
        "activated_at": "2026-09-26T12:05:00Z",
        "previous": previous if previous is not None else {"name": "generic_item_v1", "revision": None},
        "config_revision": 17,
        "stale": False,
        "applied": [
            {
                "process": "detection_worker",
                "host": "worker-1",
                "applied_config_revision": 17,
                "profile": None,
                "pack": {"name": name, "revision": revision},
                "applied_at": "2026-09-26T12:05:02Z",
                "lagging": False,
            }
        ],
    }


LIST: dict[str, Any] = {
    "packs": [
        {
            "name": "generic_item_v1",
            "source": "builtin",
            "read_only": True,
            "revision": None,
            "etag": "prompt_pack:generic_item_v1:3f2a9c01be77",
            "description": "Built-in example",
            "asks_region_text": True,
            "active": False,
            "updated_at": None,
        },
        {
            "name": "widget_tag",
            "source": "stored",
            "read_only": False,
            "revision": 2,
            "etag": "prompt_pack:widget_tag:2",
            "description": "Tags on widgets",
            "asks_region_text": False,
            "active": True,
            "active_revision": 1,
            "updated_at": "2026-09-26T12:00:00Z",
        },
    ],
    "templates": [
        {
            "name": "widget_tag",
            "source": "template",
            "read_only": True,
            "path": "examples/prompt_packs/widget_tag.json",
        }
    ],
    "active": {"name": "widget_tag", "revision": 1},
    "config_revision": 17,
    "stale": False,
}

REVISIONS = {
    "name": "widget_tag",
    "revisions": [
        {"revision": 2, "saved_at": "2026-09-26T12:00:00Z", "cloned_from": None, "description": "Tags on widgets"},
        {"revision": 1, "saved_at": "2026-09-26T10:00:00Z", "cloned_from": "template:widget_tag@-", "description": "First cut"},
    ],
}


def serve_packs(stub: Any, active_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """The W3 reads every pack page makes. Returns the mutable state the
    write handlers in each test update."""
    state: dict[str, Any] = {"active": active_state or active()}
    # The project shell's class registry read.
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/prompt_packs(\?|$)", copy.deepcopy(LIST))
    stub.on("GET", r"/prompt_packs/schema$", copy.deepcopy(SCHEMA))
    stub.on("GET", r"/prompt_packs/active$", lambda _r, _m: (200, state["active"]))
    stub.on("GET", r"/prompt_packs/widget_tag$", doc())
    stub.on("GET", r"/prompt_packs/widget_tag/revisions$", copy.deepcopy(REVISIONS))
    return state


def open_editor(page: Any, app_url: str) -> None:
    page.goto(f"{app_url}/p/default/settings/prompt-packs/widget_tag")
    page.locator('[data-field="class_user_template"] textarea').wait_for(timeout=ACTION_TIMEOUT_MS)


def test_edit_validate_save_and_resolve_a_conflict(stub, page, app_url):
    serve_packs(stub)
    validated: list[Any] = []

    def validate(request: Any, _m: Any):
        validated.append(request.post_data_json)
        return (
            200,
            {
                "ok": False,
                "errors": [
                    {
                        "code": "pack_placeholder_missing",
                        "severity": "error",
                        "field": "class_user_template",
                        "message": "class_user_template must contain {class_names_csv}",
                        "detail": {"placeholder": "class_names_csv"},
                        "bypassable": False,
                    }
                ],
                "warnings": [],
                "force_allowed": False,
            },
        )

    stub.on("POST", r"/prompt_packs/validate$", validate)
    puts: list[Any] = []

    def put(request: Any, _m: Any):
        body = request.post_data_json
        puts.append(body)
        if len(puts) == 1:
            return (200, doc(revision=3, body=body["body"], description=body["description"]))
        if len(puts) == 2:
            return (
                409,
                {
                    "detail": {
                        "error": "revision_conflict",
                        "message": "Revision 4 was saved after you opened this pack.",
                        "current_revision": 4,
                    }
                },
            )
        return (200, doc(revision=5, body=body["body"], description=body["description"]))

    stub.on("PUT", r"/prompt_packs/widget_tag$", put)

    open_editor(page, app_url)
    field = page.locator('[data-field="class_user_template"]')
    expect(field.get_by_test_id("placeholder-chip")).to_have_text("{class_names_csv} required")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/prompt_packs/validate")):
        field.locator("textarea").fill("Pick one class.")
    expect(field.get_by_test_id("config-issue")).to_contain_text(
        "class_user_template must contain {class_names_csv}", timeout=ACTION_TIMEOUT_MS
    )
    assert validated[-1] == {"name": None, "body": {**BODY, "class_user_template": "Pick one class."}}

    with page.expect_request(lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    assert puts[0] == {
        "expected_revision": 2,
        "description": "Tags on widgets",
        "body": {**BODY, "class_user_template": "Pick one class."},
    }, puts[0]
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 3", timeout=ACTION_TIMEOUT_MS)

    page.locator('[data-field="class_system"] textarea').fill("You classify widgets and gadgets.")
    with page.expect_request(lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    conflict = page.get_by_test_id("save-conflict")
    expect(conflict).to_contain_text("Revision 4 was saved after you opened this pack.")
    assert puts[1]["expected_revision"] == 3
    conflict.get_by_test_id("keep-mine").click()
    with page.expect_request(lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    assert puts[2]["expected_revision"] == 4
    assert puts[2]["body"]["class_system"] == "You classify widgets and gadgets."
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 5", timeout=ACTION_TIMEOUT_MS)


def test_activate_needs_force_only_when_the_server_allows_it(stub, page, app_url):
    state = serve_packs(stub)
    bodies: list[Any] = []

    def activate(request: Any, _m: Any):
        body = request.post_data_json
        bodies.append(body)
        if not body["force"]:
            return (
                422,
                {
                    "detail": {
                        "error": "validation_failed",
                        "message": "widget_tag r2 has errors.",
                        "report": {
                            "ok": False,
                            "errors": [
                                {
                                    "code": "pack_description_class_unknown",
                                    "severity": "error",
                                    "field": "class_descriptions.sprocket",
                                    "message": "sprocket is not a registry class",
                                    "detail": {},
                                    "bypassable": True,
                                }
                            ],
                            "warnings": [],
                            "force_allowed": True,
                        },
                    }
                },
            )
        state["active"] = active("widget_tag", 2, previous={"name": "widget_tag", "revision": 1})
        return (200, state["active"])

    stub.on("POST", r"/prompt_packs/widget_tag/activate$", activate)

    open_editor(page, app_url)
    page.get_by_test_id("config-activate").click()
    dialog = page.get_by_role("dialog", name="Activate widget_tag revision 2")
    expect(dialog.get_by_test_id("activate-from-to")).to_contain_text("widget_tag r1")
    expect(dialog.get_by_test_id("activate-force")).to_have_count(0)
    with page.expect_request(lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate", exact=True).click()
    expect(dialog.get_by_test_id("activate-error")).to_have_text("widget_tag r2 has errors.")
    expect(dialog).to_contain_text("sprocket is not a registry class")
    dialog.get_by_test_id("activate-force").check()
    with page.expect_request(lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate anyway").click()
    expect(dialog).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert bodies == [
        {"revision": 2, "expected_active": {"name": "widget_tag", "revision": 1}, "force": False},
        {"revision": 2, "expected_active": {"name": "widget_tag", "revision": 1}, "force": True},
    ], bodies
    expect(page.get_by_test_id("active-ref")).to_have_text("widget_tag r2")


def test_rollback_from_the_list(stub, page, app_url):
    state = serve_packs(stub)
    bodies: list[Any] = []

    def rollback(request: Any, _m: Any):
        bodies.append(request.post_data_json)
        state["active"] = active("generic_item_v1", None, previous={"name": "widget_tag", "revision": 1})
        return (200, state["active"])

    stub.on("POST", r"/prompt_packs/active/rollback$", rollback)

    page.goto(f"{app_url}/p/default/settings/prompt-packs")
    expect(page.get_by_test_id("active-ref")).to_have_text("widget_tag r1", timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-name="widget_tag"]').get_by_test_id("pack-active-chip")).to_have_text("active r1")
    page.get_by_role("button", name="Roll back to generic_item_v1").click()
    dialog = page.get_by_role("dialog", name="Roll back the active pack")
    with page.expect_request(lambda r: r.url.endswith("/active/rollback")):
        dialog.get_by_role("button", name="Roll back", exact=True).click()
    assert bodies == [{"expected_active": {"name": "widget_tag", "revision": 1}}], bodies
    expect(page.get_by_test_id("active-ref")).to_have_text("generic_item_v1", timeout=ACTION_TIMEOUT_MS)


def test_test_on_crop_sends_the_draft_and_shows_the_reply(stub, page, app_url):
    serve_packs(stub)
    stub.on("POST", r"/prompt_packs/validate$", CLEAN)
    tests: list[Any] = []

    def run_test(request: Any, _m: Any):
        tests.append(request.post_data_json)
        return (
            200,
            {
                "call": "classify",
                "pack": {"name": None, "revision": None, "draft": True},
                "vlm": {
                    "name": "env",
                    "revision": None,
                    "draft": False,
                    "model": "local-vlm",
                    "resolved_model": "example/vision-model",
                    "sends_images_externally": False,
                },
                "prompt": {"system": "You classify widgets.", "user_text": "Pick one of: widget, gadget"},
                "raw_reply": '[{"img": 1, "class": "widget", "confidence": 0.91}]',
                "reasoning": None,
                "latency_ms": 812.4,
                "validation": CLEAN,
                "results": [
                    {
                        "crop_id": "c_123",
                        "parse_ok": True,
                        "parse_error": None,
                        "parsed_combined": None,
                        "parsed_region": None,
                        "parsed_class": {"class_name": "widget", "confidence": 0.91},
                        "parsed_visible": None,
                        "preview_item": {"crop_id": "c_123", "bbox_norm": [0.1, 0.1, 0.5, 0.5]},
                    }
                ],
            },
        )

    stub.on("POST", r"/prompt_packs/test$", run_test)

    open_editor(page, app_url)
    page.locator('[data-field="class_system"] textarea').fill("You sort widgets.")
    panel = page.get_by_test_id("pack-test-panel")
    panel.get_by_test_id("test-crop-ids").fill("c_123")
    with page.expect_request(lambda r: r.url.endswith("/prompt_packs/test")):
        panel.get_by_test_id("test-run").click()
    assert tests == [
        {"draft": {**BODY, "class_system": "You sort widgets."}, "call": "classify", "crop_ids": ["c_123"]}
    ], tests
    expect(panel.get_by_test_id("test-raw-reply")).to_contain_text('"class": "widget"', timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("test-parse-ok")).to_be_visible()
    expect(panel.get_by_test_id("test-parsed")).to_contain_text('"class_name": "widget"')
    expect(panel.get_by_test_id("pack-test-preview")).to_be_visible()


def test_clone_a_template_opens_the_new_pack(stub, page, app_url):
    serve_packs(stub)
    clones: list[Any] = []

    def clone(request: Any, _m: Any):
        clones.append(request.post_data_json)
        return (201, doc(name="widget_tag_v2", revision=1, active=False, active_revision=None))

    stub.on("POST", r"/prompt_packs/widget_tag/clone$", clone)
    stub.on("GET", r"/prompt_packs/widget_tag_v2$", doc(name="widget_tag_v2", revision=1, active=False, active_revision=None))
    stub.on("GET", r"/prompt_packs/widget_tag_v2/revisions$", {"name": "widget_tag_v2", "revisions": []})

    page.goto(f"{app_url}/p/default/settings/prompt-packs")
    page.get_by_test_id("template-row").get_by_role("button", name="Clone").click()
    dialog = page.get_by_role("dialog", name="Clone widget_tag (template)")
    dialog.get_by_test_id("clone-name").fill("widget_tag_v2")
    with page.expect_request(lambda r: r.url.endswith("/clone")):
        dialog.get_by_role("button", name="Clone", exact=True).click()
    assert clones == [
        {"new_name": "widget_tag_v2", "revision": None, "source": "template", "description": None}
    ], clones
    page.wait_for_url("**/p/default/settings/prompt-packs/widget_tag_v2", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role("heading", name="widget_tag_v2")).to_be_visible()


def test_absent_when_prompt_packs_404(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})

    with page.expect_response(lambda r: r.url.endswith("/prompt_packs") and r.status == 404):
        page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_role("heading", name="Deployment defaults")).to_be_visible()
    expect(page.get_by_test_id("prompt-packs-card")).to_have_count(0)

    page.goto(f"{app_url}/p/default/settings/prompt-packs")
    expect(page.get_by_test_id("packs-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.goto(f"{app_url}/p/default/settings/prompt-packs/widget_tag")
    expect(page.get_by_test_id("packs-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    pack_calls = [p for _m, p in stub.handled if "/prompt_packs" in p]
    assert pack_calls and all(p.endswith("/prompt_packs") for p in pack_calls), pack_calls
