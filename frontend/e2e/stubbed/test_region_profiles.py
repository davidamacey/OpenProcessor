"""OpenProcessor W4 (region-profile CRUD, activation, impact), any_domain_plan.md
§4, §7.3, §7.4, §7.6; docs/design/w4-profile-editor-ui-plan-2026-09-27.md §10.

The backend has not shipped W4 yet, so every payload below follows the spec's
own examples (neutral widget/tag domain). The fail-closed stub means any
profile request the page makes that a test did not expect fails the test.

  1. `test_edit_validate_save_and_resolve_a_conflict` — a model picked from the
     served vocabulary; a served issue lands under its field after the
     debounced validate; Save sends `expected_revision`; a 409
     `revision_conflict` offers "Keep my edits".
  2. `test_check_for_activation_shows_the_for_activation_report`.
  3. `test_activate_force_then_impact_and_rerun` — 422 with `force_allowed`
     offers "Activate anyway" (`force: true`, pinned revision); the served
     impact renders; Re-run sends the served request as a dry run, then for
     real.
  4. `test_rollback_and_turn_off_from_the_list`.
  5. `test_clone_a_template_opens_the_new_profile`.
  6. `test_absent_when_region_profiles_404` — conftest's default 404.
"""

from __future__ import annotations

import copy
from typing import Any

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

CLEAN = {"ok": True, "errors": [], "warnings": [], "force_allowed": False}

BODY: dict[str, Any] = {
    "display_name": "Tags",
    "display_name_singular": "Tag",
    "region_class_name": "tag",
    "parent_classes": ["widget"],
    "detector_model": "tag_detector_v1",
    "segmenter_text_prompt": "tag",
    "max_regions_per_item": 4,
    "segmenter_min_score": None,
    "region_nms_iou": 0.5,
    "text_reader": "none",
    "text_hint_enabled": False,
    "input_size": 640,
    "letterbox_fill": [114, 114, 114],
    "auto_confirm_area_frac": [0.1, 0.9],
    "ocr_det_model": "",
}


def _field(name: str, label: str, group: str, type_: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "field": name,
        "label": label,
        "group": group,
        "type": type_,
        "default": None,
        "min": None,
        "max": None,
        "enum": None,
        "advanced": False,
        "applies_when": None,
        "choices_from": None,
        "empty_choice": None,
        "help": f"Help for {label}.",
    }
    base.update(over)
    return base


SCHEMA: dict[str, Any] = {
    "groups": [
        {"id": "identity", "label": "Name and display"},
        {"id": "detector", "label": "Detector"},
        {"id": "segmenter", "label": "Segmenter"},
    ],
    "fields": [
        _field("display_name", "Display name (plural)", "identity", "string", default=""),
        _field(
            "detector_model",
            "Detector model",
            "detector",
            "string",
            default="",
            choices_from="detectors",
            empty_choice={"id": "", "label": "No detector leg"},
        ),
        _field(
            "max_regions_per_item",
            "Most regions per item",
            "segmenter",
            "int",
            default=1,
            min=1,
            max=64,
        ),
        _field(
            "region_nms_iou",
            "Overlap for duplicates",
            "segmenter",
            "float",
            default=0.5,
            min=0,
            max=1,
            advanced=True,
        ),
    ],
}


def _model(name: str, label: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "choice": {"id": name, "label": label or name},
        "source": "triton",
        "state": "READY",
        "ready": True,
        "versions": ["1"],
    }


VOCABULARY: dict[str, Any] = {
    "detectors": [_model("tag_detector_v1", "tag_detector_v1 (promoted)"), _model("item_detector_base")],
    "segmenters": [],
    "vlm": {"active": {"name": "env", "revision": None}, "endpoints": []},
    "ocr": {"available": False, "pipeline_models": [], "det_models": [], "rec_models": []},
    "model_choices": [],
    "text_reader_modes": [],
    "registry_classes": [
        {"class_id": 0, "class_name": "widget", "choice": {"id": "widget", "label": "widget"}}
    ],
    "prompt_pack_calls": [],
    "labels": {"scope": {}},
}


def doc(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "widget_tag",
        "source": "stored",
        "read_only": False,
        "revision": 3,
        "etag": "region_profile:widget_tag:3",
        "description": "Tags on widgets",
        "body": copy.deepcopy(BODY),
        "effective": {
            "reads_text": False,
            "text_hint_active": False,
            "legs": ["detector", "segmenter"],
            "segmenter_enabled": True,
        },
        "created_at": "2026-09-26T10:00:00Z",
        "updated_at": "2026-09-26T12:00:00Z",
        "updated_by": None,
        "cloned_from": "template:widget_tag@-",
        "active": True,
        "active_revision": 2,
        "validation": CLEAN,
    }
    base.update(over)
    return base


def active(
    name: str | None = "widget_tag", revision: int | None = 2, previous: Any = "default"
) -> dict[str, Any]:
    if previous == "default":
        previous = {"name": "env_tags", "revision": None}
    return {
        "axis": "detection_profile",
        "active": {"name": name, "revision": revision},
        "source": "stored",
        "activated_at": "2026-09-26T12:05:00Z",
        "previous": previous,
        "config_revision": 21,
        "stale": False,
        "applied": [],
    }


LIST: dict[str, Any] = {
    "profiles": [
        {
            "name": "env_tags",
            "source": "env",
            "read_only": True,
            "revision": None,
            "etag": "region_profile:env_tags:1a2b",
            "display_name": "Tags",
            "display_name_singular": "Tag",
            "region_class_name": "tag",
            "text_reader": "ocr",
            "reads_text": True,
            "detector_model": "tag_detector_v1",
            "segmenter_text_prompt": "",
            "parent_classes": ["widget"],
            "max_regions_per_item": 1,
            "active": False,
            "updated_at": None,
        },
        {
            "name": "widget_tag",
            "source": "stored",
            "read_only": False,
            "revision": 3,
            "etag": "region_profile:widget_tag:3",
            "display_name": "Tags",
            "display_name_singular": "Tag",
            "region_class_name": "tag",
            "text_reader": "none",
            "reads_text": False,
            "detector_model": "",
            "segmenter_text_prompt": "tag",
            "parent_classes": ["widget", "gadget"],
            "max_regions_per_item": 4,
            "active": True,
            "active_revision": 2,
            "updated_at": "2026-09-26T12:00:00Z",
        },
    ],
    "templates": [
        {
            "name": "widget_tag",
            "source": "template",
            "read_only": True,
            "path": "examples/region_profiles/widget_tag.json",
            "display_name": "Tags",
            "reads_text": False,
        }
    ],
    "active": {"name": "widget_tag", "revision": 2},
    "config_revision": 21,
    "stale": False,
}

REVISIONS = {
    "name": "widget_tag",
    "revisions": [
        {"revision": 3, "saved_at": "2026-09-26T12:00:00Z", "cloned_from": None, "description": "Tags on widgets"},
        {"revision": 2, "saved_at": "2026-09-26T11:00:00Z", "cloned_from": None, "description": "Second cut"},
    ],
}

SUGGESTED = {
    "targets": {"filter": {"profile_not": "widget_tag", "include_detected": True}},
    "scopes": ["region"],
    "region_mode": "redetect",
    "dry_run": True,
}

IMPACT: dict[str, Any] = {
    "items_total": 1840,
    "by_profile": [
        {"name": "env_tags", "revision": None, "count": 900},
        {"name": "widget_tag", "revision": 3, "count": 40},
    ],
    "validated_items": 12,
    "unseeded_items": 850,
    "pending_items": 38,
    "pending_not_matching": 0,
    "suggested_reprocess": SUGGESTED,
}

DATASET_FORMATS: dict[str, Any] = {
    "formats": [],
    "processing": [],
    "parents": [],
    "label_trust": [],
    "mapping_actions": [],
    "match_kinds": [],
    "issues": [],
    "upload": {"max_bytes": 1, "max_files": 1, "accepted": []},
    "status_labels": {"queued": "Queued"},
    "reprocess": {
        "scopes": [{"id": "region", "label": "Regions", "description": "Regenerate regions.", "unit": "item"}],
        "region_modes": [{"id": "redetect", "label": "Detect again", "description": "Remove machine boxes."}],
        "lock_rule": "Validated items are never changed.",
    },
}


def serve_profiles(stub: Any, active_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """The reads every profile page makes. Returns the mutable state the write
    handlers in each test update."""
    state: dict[str, Any] = {"active": active_state or active()}
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/region_profiles(\?|$)", copy.deepcopy(LIST))
    stub.on("GET", r"/region_profiles/schema$", copy.deepcopy(SCHEMA))
    stub.on("GET", r"/region_profiles/active$", lambda _r, _m: (200, state["active"]))
    stub.on("GET", r"/region_profiles/widget_tag$", doc())
    stub.on("GET", r"/region_profiles/widget_tag/revisions$", copy.deepcopy(REVISIONS))
    stub.on("GET", r"/config/vocabulary(\?|$)", copy.deepcopy(VOCABULARY))
    return state


def open_editor(page: Any, app_url: str, name: str = "widget_tag") -> None:
    page.goto(f"{app_url}/p/default/settings/region-profiles/{name}")
    page.locator('[data-field="max_regions_per_item"] input').wait_for(timeout=ACTION_TIMEOUT_MS)


def test_edit_validate_save_and_resolve_a_conflict(stub, page, app_url):
    serve_profiles(stub)
    validated: list[Any] = []

    def validate(request: Any, _m: Any):
        validated.append(request.post_data_json)
        if request.post_data_json["body"]["max_regions_per_item"] != 99:
            return (200, CLEAN)
        return (
            200,
            {
                "ok": False,
                "errors": [
                    {
                        "code": "profile_field_range",
                        "severity": "error",
                        "field": "max_regions_per_item",
                        "message": "max_regions_per_item must be at most 64",
                        "detail": {"max": 64},
                        "bypassable": False,
                    }
                ],
                "warnings": [],
                "force_allowed": False,
            },
        )

    stub.on("POST", r"/region_profiles/validate$", validate)
    puts: list[Any] = []

    def put(request: Any, _m: Any):
        body = request.post_data_json
        puts.append(body)
        if len(puts) == 1:
            return (409, {"detail": {
                "error": "revision_conflict",
                "message": "Revision 4 was saved after you opened this profile.",
                "current_revision": 4,
            }})
        return (200, doc(revision=5, body=body["body"], description=body["description"]))

    stub.on("PUT", r"/region_profiles/widget_tag$", put)

    open_editor(page, app_url)
    detector = page.locator('[data-field="detector_model"] [data-testid="field-choice"]')
    expect(detector.locator("option")).to_have_text(
        ["No detector leg", "tag_detector_v1 (promoted)", "item_detector_base"]
    )
    # The advanced field stays behind its toggle.
    expect(page.locator('[data-field="region_nms_iou"]')).to_have_count(0)

    detector.select_option("item_detector_base")
    field = page.locator('[data-field="max_regions_per_item"]')
    field.locator("input").fill("99")
    expect(field.get_by_test_id("config-issue")).to_contain_text(
        "max_regions_per_item must be at most 64", timeout=ACTION_TIMEOUT_MS
    )
    edited = {**BODY, "detector_model": "item_detector_base", "max_regions_per_item": 99}
    assert validated[-1] == {"name": None, "body": edited}, validated[-1]

    with page.expect_request(lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    conflict = page.get_by_test_id("save-conflict")
    expect(conflict).to_contain_text("Revision 4 was saved after you opened this profile.")
    assert puts[0] == {"expected_revision": 3, "description": "Tags on widgets", "body": edited}, puts[0]
    conflict.get_by_test_id("keep-mine").click()
    with page.expect_request(lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    assert puts[1]["expected_revision"] == 4
    assert puts[1]["body"] == edited
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 5", timeout=ACTION_TIMEOUT_MS)


def test_check_for_activation_shows_the_for_activation_report(stub, page, app_url):
    serve_profiles(stub)
    urls: list[str] = []

    def validate(request: Any, _m: Any):
        urls.append(request.url)
        if "for_activation=true" not in request.url:
            return (200, CLEAN)
        return (
            200,
            {
                "ok": False,
                "errors": [
                    {
                        "code": "detector_model_not_ready",
                        "severity": "error",
                        "field": "detector_model",
                        "message": "tag_detector_v1 is not loaded",
                        "detail": {},
                        "bypassable": True,
                    }
                ],
                "warnings": [],
                "force_allowed": True,
            },
        )

    stub.on("POST", r"/region_profiles/validate$", validate)

    open_editor(page, app_url)
    with page.expect_request(lambda r: "for_activation=true" in r.url):
        page.get_by_test_id("check-activation").click()
    result = page.get_by_test_id("activation-check")
    expect(result).to_contain_text("tag_detector_v1 is not loaded", timeout=ACTION_TIMEOUT_MS)
    expect(result).to_contain_text("1 error")
    expect(result).to_contain_text("the server allows overriding")
    assert len(urls) == 1 and urls[0].endswith("/region_profiles/validate?for_activation=true"), urls


def test_activate_force_then_impact_and_rerun(stub, page, app_url):
    state = serve_profiles(stub)
    stub.on("GET", r"/datasets/formats(\?|$)", copy.deepcopy(DATASET_FORMATS))
    stub.on("POST", r"/region_profiles/validate$", CLEAN)
    activations: list[Any] = []

    def activate(request: Any, _m: Any):
        body = request.post_data_json
        activations.append(body)
        if not body["force"]:
            return (
                422,
                {
                    "detail": {
                        "error": "validation_failed",
                        "message": "widget_tag r3 has errors.",
                        "report": {
                            "ok": False,
                            "errors": [
                                {
                                    "code": "detector_model_not_ready",
                                    "severity": "error",
                                    "field": "detector_model",
                                    "message": "tag_detector_v1 is not loaded",
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
        state["active"] = active("widget_tag", 3, previous={"name": "widget_tag", "revision": 2})
        return (200, {**state["active"], "impact": copy.deepcopy(IMPACT), "validation": CLEAN})

    stub.on("POST", r"/region_profiles/widget_tag/activate$", activate)
    reprocesses: list[Any] = []

    def reprocess(request: Any, _m: Any):
        body = request.post_data_json
        reprocesses.append(body)
        scopes = [{"scope": "region", "selected": 940, "locked_skipped": 12, "queued": 0 if body["dry_run"] else 928}]
        job = None if body["dry_run"] else {"job_id": "rp-1", "status": "queued", "poll_after_s": None}
        message = "940 items would be re-run." if body["dry_run"] else "928 items queued."
        return (200, {"dry_run": body["dry_run"], "scopes": scopes, "job": job, "items": [], "message": message})

    stub.on("POST", r"/reprocess$", reprocess)

    open_editor(page, app_url)
    page.get_by_test_id("config-activate").click()
    dialog = page.get_by_role("dialog", name="Activate widget_tag revision 3")
    expect(dialog.get_by_test_id("activate-from-to")).to_contain_text("widget_tag r2")
    expect(dialog.get_by_test_id("activate-force")).to_have_count(0)
    with page.expect_request(lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate", exact=True).click()
    expect(dialog.get_by_test_id("activate-error")).to_have_text("widget_tag r3 has errors.")
    expect(dialog).to_contain_text("tag_detector_v1 is not loaded")
    dialog.get_by_test_id("activate-force").check()
    with page.expect_request(lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate anyway").click()
    expect(dialog).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert activations == [
        {"revision": 3, "expected_active": {"name": "widget_tag", "revision": 2}, "force": False},
        {"revision": 3, "expected_active": {"name": "widget_tag", "revision": 2}, "force": True},
    ], activations

    impact = page.get_by_test_id("profile-impact")
    expect(impact.get_by_test_id("impact-items_total")).to_contain_text("1,840", timeout=ACTION_TIMEOUT_MS)
    expect(impact.get_by_test_id("impact-by-profile")).to_contain_text("env_tags")

    impact.get_by_test_id("rerun-open").click()
    expect(impact.get_by_test_id("rerun-dry-run")).to_contain_text("940", timeout=ACTION_TIMEOUT_MS)
    assert reprocesses == [{**SUGGESTED, "dry_run": True}], reprocesses
    impact.get_by_test_id("rerun-apply").click()
    confirm = page.get_by_role("dialog", name="Re-run items")
    with page.expect_request(lambda r: r.url.endswith("/reprocess")):
        confirm.get_by_role("button", name="Re-run", exact=True).click()
    expect(impact.get_by_test_id("rerun-result")).to_contain_text("928 items queued.", timeout=ACTION_TIMEOUT_MS)
    expect(impact.get_by_test_id("rerun-job")).to_contain_text("rp-1")
    assert reprocesses[1] == {**SUGGESTED, "dry_run": False}, reprocesses


def test_rollback_and_turn_off_from_the_list(stub, page, app_url):
    state = serve_profiles(stub)
    rollbacks: list[Any] = []
    deactivations: list[Any] = []

    def rollback(request: Any, _m: Any):
        rollbacks.append(request.post_data_json)
        state["active"] = active("env_tags", None, previous={"name": "widget_tag", "revision": 2})
        return (200, state["active"])

    def deactivate(request: Any, _m: Any):
        deactivations.append(request.post_data_json)
        state["active"] = active(None, None, previous={"name": "env_tags", "revision": None})
        return (200, state["active"])

    stub.on("POST", r"/region_profiles/active/rollback$", rollback)
    stub.on("POST", r"/region_profiles/deactivate$", deactivate)

    page.goto(f"{app_url}/p/default/settings/region-profiles")
    expect(page.get_by_test_id("active-ref")).to_have_text("widget_tag r2", timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-name="widget_tag"]').get_by_test_id("profile-active-chip")).to_contain_text("active r2")

    page.get_by_role("button", name="Roll back to env_tags").click()
    dialog = page.get_by_role("dialog", name="Roll back the active region profile")
    with page.expect_request(lambda r: r.url.endswith("/active/rollback")):
        dialog.get_by_role("button", name="Roll back", exact=True).click()
    assert rollbacks == [{"expected_active": {"name": "widget_tag", "revision": 2}}], rollbacks
    expect(page.get_by_test_id("active-ref")).to_have_text("env_tags", timeout=ACTION_TIMEOUT_MS)

    page.get_by_test_id("active-deactivate").click()
    off = page.get_by_role("dialog", name="Turn off region detection")
    with page.expect_request(lambda r: r.url.endswith("/region_profiles/deactivate")):
        off.get_by_role("button", name="Turn off", exact=True).click()
    assert deactivations == [{"expected_active": {"name": "env_tags", "revision": None}}], deactivations
    expect(page.get_by_test_id("active-ref")).to_have_text(
        "off: region detection is off", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("active-deactivate")).to_have_count(0)


def test_clone_a_template_opens_the_new_profile(stub, page, app_url):
    serve_profiles(stub)
    clones: list[Any] = []

    def clone(request: Any, _m: Any):
        clones.append(request.post_data_json)
        return (201, doc(name="widget_tag_v2", revision=1, active=False, active_revision=None))

    stub.on("POST", r"/region_profiles/widget_tag/clone$", clone)
    stub.on("GET", r"/region_profiles/widget_tag_v2$", doc(name="widget_tag_v2", revision=1, active=False, active_revision=None))
    stub.on("GET", r"/region_profiles/widget_tag_v2/revisions$", {"name": "widget_tag_v2", "revisions": []})

    page.goto(f"{app_url}/p/default/settings/region-profiles")
    page.get_by_test_id("template-row").get_by_role("button", name="Clone").click()
    dialog = page.get_by_role("dialog", name="Clone widget_tag (template)")
    dialog.get_by_test_id("clone-name").fill("widget_tag_v2")
    with page.expect_request(lambda r: r.url.endswith("/clone")):
        dialog.get_by_role("button", name="Clone", exact=True).click()
    assert clones == [
        {"new_name": "widget_tag_v2", "revision": None, "source": "template", "description": None}
    ], clones
    page.wait_for_url("**/p/default/settings/region-profiles/widget_tag_v2", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role("heading", name="widget_tag_v2")).to_be_visible()


def test_absent_when_region_profiles_404(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})

    with page.expect_response(lambda r: "/region_profiles" in r.url and r.status == 404):
        page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_role("heading", name="Deployment defaults")).to_be_visible()
    expect(page.get_by_test_id("region-profiles-card")).to_have_count(0)

    page.goto(f"{app_url}/p/default/settings/region-profiles")
    expect(page.get_by_test_id("profiles-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.goto(f"{app_url}/p/default/settings/region-profiles/widget_tag")
    expect(page.get_by_test_id("profiles-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    calls = [p for _m, p in stub.handled if "/region_profiles" in p]
    assert calls and all(p.endswith("/region_profiles") for p in calls), calls
