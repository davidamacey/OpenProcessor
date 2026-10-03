"""OpenProcessor v0.4.0 SAM 3 open-vocabulary sets,
docs/design/v040-backend-deltas-ui-plan-2026-10-03.md §6 (Track A). Payloads
follow the vendored contract (fce17771) in the neutral widget domain. The
fail-closed stub means any open-vocab request the page makes that a test did
not expect fails the test.

  1. `test_absent_when_open_vocab_404` — conftest's default 404: no /settings
     card, the routes say so, no other `/open_vocab` request fires.
  2. `test_list_new_set_and_clone_a_template`.
  3. `test_edit_a_target_validate_save_and_resolve_a_conflict` — a served
     issue lands under the cell its path names; Save sends
     `expected_revision`; a 409 offers "Keep my edits".
  4. `test_activation_refusal_offers_force_only_when_allowed`.
  5. `test_test_panel_draws_hits_and_words_a_dropped_one`.
  6. `test_rerun_sends_the_all_images_open_vocab_request` — also one re-run
     per served status, labelled by the served vocabulary.
  7. `test_segmenter_fact_is_shown_as_served_and_hides_nothing`.

Screenshots (1600 and 800 px) land in `artifacts_local/v040-ui/open-vocab/`
and are looked at by hand; the assertions here do not replace that.
"""

from __future__ import annotations

import copy
import struct
import zlib
from pathlib import Path
from typing import Any

from conftest import ACTION_TIMEOUT_MS, expect_handled
from fixtures.wire import make_item
from playwright.sync_api import expect

SHOT_DIR = Path(__file__).resolve().parents[2] / "artifacts_local" / "v040-ui" / "open-vocab"
CLEAN = {"ok": True, "errors": [], "warnings": [], "force_allowed": False}


def _row(scope: str, field: str, label: str, type_: str, default: Any, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scope": scope,
        "field": field,
        "label": label,
        "type": type_,
        "default": default,
        "advanced": False,
        "help": "",
        "min": None,
        "max": None,
    }
    base.update(over)
    return base


VOCABULARY: dict[str, Any] = {
    "statuses": [
        {"value": "pending", "label": "Waiting for a pass"},
        {"value": "done", "label": "Pass finished"},
        {"value": "skipped_gate", "label": "Skipped by the gate"},
        {"value": "failed", "label": "Pass failed"},
    ],
    "drop_reasons": [
        {"value": "below_min_score", "label": "Scored below the target minimum"},
        {"value": "nms", "label": "Overlapped a better hit"},
        {"value": "agree_existing", "label": "Matches an item already there"},
    ],
    "gate_reasons": [
        {"value": "no_parent_class", "label": "No parent class on the image"},
        {"value": "vlm_no", "label": "The VLM pre-check said no"},
    ],
}

SCHEMA: dict[str, Any] = {
    "max_enabled_targets_ceiling": 16,
    "vocabulary": VOCABULARY,
    "fields": [
        _row("set", "display_name", "Display name", "string", "", help="What this set is called."),
        _row("set", "run_on_ingest", "Run on ingest", "bool", False, help="Run on every new image."),
        _row("set", "image_max_side", "Longest image side", "int", 1024, advanced=True, min=256, max=2048),
        _row("target", "prompt", "Prompt", "string", "", help="What to find, in words."),
        _row("target", "class_name", "Class name", "string", "", help="The class hits are stored as."),
        _row("target", "min_score", "Minimum score", "float", 0.5, min=0, max=1),
        _row("target", "enabled", "Enabled", "bool", True),
        _row("target", "max_instances", "Most instances", "int", 20, advanced=True),
        _row("gating", "tier2_vlm_precheck", "VLM pre-check", "bool", False),
        _row("tier3_hit_rate", "window", "Window", "int", 20, advanced=True),
    ],
}

BODY: dict[str, Any] = {
    "display_name": "Widgets",
    "run_on_ingest": False,
    "image_max_side": 1024,
    "dedup_iou": 0.5,
    "max_enabled_targets": 8,
    "targets": [
        {"prompt": "blue widget", "class_name": "widget", "enabled": True, "min_score": 0.5},
        {"prompt": "cracked widget", "class_name": "", "enabled": True, "min_score": 0.4},
    ],
    "gating": {"tier2_vlm_precheck": False, "tier3_hit_rate": {"enabled": False, "window": 20}},
}


def png(width: int, height: int) -> bytes:
    """A plain mid-grey PNG, so the hit overlay has a real image to sit on."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x70" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def doc(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "widgets",
        "source": "stored",
        "read_only": False,
        "revision": 3,
        "etag": "ov:widgets:3",
        "description": "Widget finder",
        "body": copy.deepcopy(BODY),
        "created_at": "2026-10-01T09:00:00Z",
        "updated_at": "2026-10-02T09:00:00Z",
        "cloned_from": None,
        "active": False,
        "active_revision": None,
        "validation": CLEAN,
    }
    base.update(over)
    return base


def active(name: str | None = None, revision: int | None = None, previous: Any = None) -> dict[str, Any]:
    return {
        "axis": "open_vocab",
        "active": {"name": name, "revision": revision},
        "source": "stored" if name else "env",
        "activated_at": "2026-10-02T10:00:00Z" if name else None,
        "previous": previous,
        "config_revision": 7,
        "stale": False,
        "applied": [],
    }


def summary(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "widgets",
        "source": "stored",
        "read_only": False,
        "revision": 3,
        "etag": "ov:widgets:3",
        "display_name": "Widgets",
        "n_targets": 2,
        "n_enabled_targets": 2,
        "run_on_ingest": False,
        "active": False,
        "active_revision": None,
        "updated_at": "2026-10-02T09:00:00Z",
    }
    base.update(over)
    return base


LIST: dict[str, Any] = {
    "sets": [summary()],
    "templates": [
        {
            "name": "starter_widgets",
            "path": "templates/open_vocab/starter_widgets.json",
            "n_targets": 1,
            "display_name": "Starter widgets",
            "read_only": True,
            "source": "template",
        }
    ],
    "active": {"name": None, "revision": None},
    "config_revision": 7,
    "stale": False,
    "segmenter": {"configured": True, "reachable": True},
}

REVISIONS = {
    "name": "widgets",
    "revisions": [
        {"revision": 3, "saved_at": "2026-10-02T09:00:00Z", "cloned_from": None, "description": "Widget finder"},
        {"revision": 2, "saved_at": "2026-10-01T09:00:00Z", "cloned_from": None, "description": "First cut"},
    ],
}


def serve_open_vocab(stub: Any, list_body: dict[str, Any] | None = None, active_state: dict[str, Any] | None = None):
    state: dict[str, Any] = {"active": active_state or active()}
    # The project shell's class registry read.
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/open_vocab(\?|$)", lambda _r, _m: (200, copy.deepcopy(list_body or LIST)))
    stub.on("GET", r"/open_vocab/schema$", copy.deepcopy(SCHEMA))
    stub.on("GET", r"/open_vocab/active$", lambda _r, _m: (200, state["active"]))
    stub.on("GET", r"/open_vocab/widgets$", doc())
    stub.on("GET", r"/open_vocab/widgets/revisions$", copy.deepcopy(REVISIONS))
    return state


def open_editor(page: Any, app_url: str) -> None:
    page.goto(f"{app_url}/p/default/settings/open-vocab/widgets")
    page.get_by_test_id("target-row").first.wait_for(timeout=ACTION_TIMEOUT_MS)


def shoot(page: Any, locator: Any, name: str) -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    original = page.viewport_size
    for w in (1600, 800):
        page.set_viewport_size({"width": w, "height": 1400})
        page.wait_for_timeout(200)
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 1, f"{name}@{w}: horizontal overflow {overflow}px"
        locator.screenshot(path=str(SHOT_DIR / f"{name}-{w}.png"))
    if original:
        page.set_viewport_size(original)


def test_absent_when_open_vocab_404(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})

    with page.expect_response(lambda r: "/open_vocab" in r.url and r.status == 404):
        page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_role("heading", name="Deployment defaults")).to_be_visible()
    expect(page.get_by_test_id("open-vocab-card")).to_have_count(0)

    page.goto(f"{app_url}/p/default/settings/open-vocab")
    expect(page.get_by_test_id("open-vocab-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.goto(f"{app_url}/p/default/settings/open-vocab/widgets")
    expect(page.get_by_test_id("open-vocab-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    calls = [p for _m, p in stub.handled if "/open_vocab" in p]
    assert calls and all("/open_vocab" in p and p.split("/open_vocab")[1] == "" for p in calls), calls


def test_list_new_set_and_clone_a_template(stub, page, app_url):
    served = copy.deepcopy(LIST)
    served["sets"] = [summary(active=True, active_revision=3)]
    served["active"] = {"name": "widgets", "revision": 3}
    serve_open_vocab(stub, served, active("widgets", 3, previous={"name": "widgets", "revision": 2}))
    posts: list[Any] = []

    def create(request: Any, _m: Any):
        posts.append(request.post_data_json)
        return (201, doc(name="fresh", revision=1))

    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})
    stub.on("POST", r"/open_vocab$", create)
    clones: list[Any] = []

    def clone(request: Any, _m: Any):
        clones.append(request.post_data_json)
        return (201, doc(name="mine", revision=1))

    stub.on("POST", r"/open_vocab/starter_widgets/clone$", clone)
    stub.on("GET", r"/open_vocab/fresh$", doc(name="fresh", revision=1))
    stub.on("GET", r"/open_vocab/fresh/revisions$", {"name": "fresh", "revisions": []})
    stub.on("GET", r"/open_vocab/mine$", doc(name="mine", revision=1))
    stub.on("GET", r"/open_vocab/mine/revisions$", {"name": "mine", "revisions": []})

    page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_test_id("open-vocab-card")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("open-vocab-card").get_by_role("link").click()
    expect(page.get_by_test_id("active-ref")).to_have_text("widgets r3", timeout=ACTION_TIMEOUT_MS)
    row = page.locator('[data-name="widgets"]')
    expect(row.get_by_test_id("open-vocab-active-chip")).to_have_text("active r3")
    expect(page.get_by_test_id("template-row")).to_have_count(1)
    shoot(page, page.locator("main, body").first, "list")

    page.get_by_test_id("open-vocab-new").click()
    page.get_by_test_id("open-vocab-new-name").fill("fresh")
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/open_vocab")):
        page.get_by_role("dialog").get_by_role("button", name="Create", exact=True).click()
    assert posts == [{"name": "fresh", "body": {}}], posts
    page.wait_for_url("**/settings/open-vocab/fresh", timeout=ACTION_TIMEOUT_MS)

    page.goto(f"{app_url}/p/default/settings/open-vocab")
    page.get_by_test_id("template-row").get_by_role("button", name="Clone").click()
    dialog = page.get_by_role("dialog")
    dialog.locator("input").first.fill("mine")
    with expect_handled(page, lambda r: r.url.endswith("/clone")):
        dialog.get_by_role("button", name="Clone", exact=True).click()
    assert clones[0]["new_name"] == "mine" and clones[0]["source"] == "template", clones
    page.wait_for_url("**/settings/open-vocab/mine", timeout=ACTION_TIMEOUT_MS)


def test_segmenter_fact_is_shown_as_served_and_hides_nothing(stub, page, app_url):
    served = copy.deepcopy(LIST)
    served["segmenter"] = {"configured": False, "reachable": False}
    serve_open_vocab(stub, served)
    page.goto(f"{app_url}/p/default/settings/open-vocab")
    notice = page.get_by_test_id("segmenter-notice")
    expect(notice).to_have_attribute("data-state", "not_configured", timeout=ACTION_TIMEOUT_MS)
    expect(notice).to_contain_text("No segmenter is configured")
    expect(page.get_by_test_id("open-vocab-row")).to_have_count(1)
    open_editor(page, app_url)
    expect(page.get_by_test_id("segmenter-notice")).to_have_attribute("data-state", "not_configured")
    expect(page.get_by_test_id("target-row")).to_have_count(2)


def test_edit_a_target_validate_save_and_resolve_a_conflict(stub, page, app_url):
    serve_open_vocab(stub)
    validated: list[Any] = []

    def validate(request: Any, _m: Any):
        validated.append(request.post_data_json)
        return (
            200,
            {
                "ok": False,
                "errors": [
                    {
                        "code": "open_vocab_empty_prompt",
                        "severity": "error",
                        "field": "targets[1].prompt",
                        "message": "A target needs a prompt.",
                        "detail": {},
                        "bypassable": False,
                    }
                ],
                "warnings": [],
                "force_allowed": False,
            },
        )

    stub.on("POST", r"/open_vocab/validate$", validate)
    puts: list[Any] = []

    def put(request: Any, _m: Any):
        body = request.post_data_json
        puts.append(body)
        if len(puts) == 1:
            return (200, doc(revision=4, body=body["body"]))
        if len(puts) == 2:
            return (
                409,
                {
                    "detail": {
                        "error": "revision_conflict",
                        "message": "Revision 5 was saved after you opened this set.",
                        "current_revision": 5,
                    }
                },
            )
        return (200, doc(revision=6, body=body["body"]))

    stub.on("PUT", r"/open_vocab/widgets$", put)

    open_editor(page, app_url)
    cell = page.locator('[data-field="targets[1].prompt"]')
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/open_vocab/validate")):
        cell.locator("input").fill("")
    row = page.get_by_test_id("target-row").nth(1)
    expect(row.get_by_test_id("config-issue")).to_contain_text("A target needs a prompt.", timeout=ACTION_TIMEOUT_MS)
    assert validated[-1]["name"] is None
    assert validated[-1]["body"]["targets"][1]["prompt"] == ""
    expect(page.get_by_test_id("target-row").nth(0).get_by_test_id("config-issue")).to_have_count(0)
    shoot(page, page.locator("body"), "editor-issue")

    cell.locator("input").fill("chipped widget")
    with expect_handled(page, lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    assert puts[0]["expected_revision"] == 3, puts[0]
    assert puts[0]["body"]["targets"][1]["prompt"] == "chipped widget"
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 4", timeout=ACTION_TIMEOUT_MS)

    page.locator('[data-field="targets[0].min_score"] input').fill("0.7")
    with expect_handled(page, lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    conflict = page.get_by_test_id("save-conflict")
    expect(conflict).to_contain_text("Revision 5 was saved after you opened this set.")
    conflict.get_by_test_id("keep-mine").click()
    with expect_handled(page, lambda r: r.method == "PUT"):
        page.get_by_test_id("config-save").click()
    assert puts[2]["expected_revision"] == 5
    assert puts[2]["body"]["targets"][0]["min_score"] == 0.7
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 6", timeout=ACTION_TIMEOUT_MS)


def test_activation_refusal_offers_force_only_when_allowed(stub, page, app_url):
    state = serve_open_vocab(stub)
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
                        "message": "widgets r3 has errors.",
                        "report": {
                            "ok": False,
                            "errors": [
                                {
                                    "code": "segmenter_unreachable",
                                    "severity": "error",
                                    "field": None,
                                    "message": "The segmenter did not answer.",
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
        state["active"] = active("widgets", 3)
        return (200, {**state["active"], "validation": CLEAN})

    stub.on("POST", r"/open_vocab/widgets/activate$", activate)

    open_editor(page, app_url)
    page.get_by_test_id("config-activate").click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_test_id("activate-force")).to_have_count(0)
    with expect_handled(page, lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate", exact=True).click()
    expect(dialog.get_by_test_id("activate-error")).to_have_text("widgets r3 has errors.")
    expect(dialog).to_contain_text("The segmenter did not answer.")
    shoot(page, dialog, "activate-refused")
    dialog.get_by_test_id("activate-force").check()
    with expect_handled(page, lambda r: r.url.endswith("/activate")):
        dialog.get_by_role("button", name="Activate anyway").click()
    expect(dialog).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert bodies == [
        {"revision": 3, "expected_active": {"name": None, "revision": None}, "force": False},
        {"revision": 3, "expected_active": {"name": None, "revision": None}, "force": True},
    ], bodies
    expect(page.get_by_test_id("active-ref")).to_have_text("widgets r3")


def test_test_panel_draws_hits_and_words_a_dropped_one(stub, page, app_url):
    serve_open_vocab(stub)
    stub.on("POST", r"/open_vocab/validate$", CLEAN)
    stub.on("GET", r"/crops/c_123$", make_item(crop_id="c_123", image_id="img_9"))
    tests: list[Any] = []

    def run_test(request: Any, _m: Any):
        tests.append(request.post_data_json)
        return (
            200,
            {
                "image": {"width": 800, "height": 600},
                "prompt": "blue widget",
                "class_name": "widget",
                "gate": {"run": True, "tier": None, "reason": None},
                "hits": [
                    {
                        "bbox_norm": [0.1, 0.1, 0.4, 0.5],
                        "score": 0.91,
                        "selected": True,
                        "drop_reason": None,
                        "mask_polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5]],
                    },
                    {
                        "bbox_norm": [0.5, 0.5, 0.8, 0.9],
                        "score": 0.62,
                        "selected": False,
                        "drop_reason": "agree_existing",
                        "mask_polygon": None,
                    },
                ],
                "elapsed_ms": 412.5,
                "validation": CLEAN,
            },
        )

    stub.on("POST", r"/open_vocab/test$", run_test)
    stub.on("GET", r"/crops/[^/]+/image$", (200, png(400, 300), "image/png"))
    stub.on(
        "GET",
        r"/crops/c_123/context$",
        {
            "image": {"image_id": "img_9", "image_path": "/fixtures/stub.jpg", "width": 400, "height": 300},
            "items": [],
        },
    )

    open_editor(page, app_url)
    panel = page.get_by_test_id("open-vocab-test-panel")
    panel.get_by_test_id("ov-test-crop-id").fill("c_123")
    with expect_handled(page, lambda r: r.url.endswith("/open_vocab/test")):
        panel.get_by_test_id("ov-test-run").click()
    expect(panel.get_by_test_id("ov-test-hit")).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    assert tests == [
        {
            "image_id": "img_9",
            "target": BODY["targets"][0],
            "image_max_side": 1024,
            "dedup_iou": 0.5,
        }
    ], tests
    expect(panel.get_by_test_id("ov-test-hit").nth(1)).to_contain_text("Matches an item already there")
    expect(panel.get_by_test_id("overlay-extra-box")).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    assert panel.get_by_test_id("overlay-extra-box").nth(1).get_attribute("data-dimmed") == "true"
    shoot(page, panel, "test-panel")


def test_a_segmenter_error_is_an_error_not_no_hits(stub, page, app_url):
    serve_open_vocab(stub)
    stub.on("POST", r"/open_vocab/validate$", CLEAN)
    stub.on("GET", r"/crops/c_123$", make_item(crop_id="c_123", image_id="img_9"))
    stub.on(
        "POST",
        r"/open_vocab/test$",
        (502, {"detail": {"error": "segmenter_error", "message": "The segmenter timed out."}}),
    )
    open_editor(page, app_url)
    panel = page.get_by_test_id("open-vocab-test-panel")
    panel.get_by_test_id("ov-test-crop-id").fill("c_123")
    panel.get_by_test_id("ov-test-run").click()
    expect(panel.get_by_test_id("ov-test-error")).to_have_text(
        "Segmenter error: The segmenter timed out.", timeout=ACTION_TIMEOUT_MS
    )
    expect(panel.get_by_test_id("ov-test-no-hits")).to_have_count(0)


def test_rerun_sends_the_all_images_open_vocab_request(stub, page, app_url):
    served = copy.deepcopy(LIST)
    served["sets"] = [summary(active=True, active_revision=3)]
    served["active"] = {"name": "widgets", "revision": 3}
    serve_open_vocab(stub, served, active("widgets", 3))
    stub.on(
        "GET",
        r"/datasets/formats$",
        {
            "formats": [{"format": "yolo", "label": "YOLO"}],
            "processing_modes": [],
            "parents_modes": [],
            "trust_levels": [],
            "mapping_actions": [],
            "match_kinds": [],
            "upload_limits": {"max_bytes": 1000000},
        },
    )
    posts: list[Any] = []

    def reprocess(request: Any, _m: Any):
        posts.append(request.post_data_json)
        return (
            200,
            {
                "dry_run": request.post_data_json["dry_run"],
                "scopes": [
                    {
                        "scope": "open_vocab",
                        "selected": 12,
                        "queued": 0,
                        "detail": {"estimated_calls": 24, "segmenter_reachable": True},
                    }
                ],
            },
        )

    stub.on("POST", r"/reprocess$", reprocess)
    page.goto(f"{app_url}/p/default/settings/open-vocab")
    panel = page.get_by_test_id("open-vocab-rerun")
    expect(panel).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    shoot(page, page.locator("body"), "list-rerun")
    panel.get_by_test_id("reprocess-open").first.click()
    dialog = page.get_by_role("dialog", name="Reprocess")
    with expect_handled(page, lambda r: r.url.endswith("/reprocess")):
        dialog.get_by_role("button", name="Check what would run").click()
    expect(dialog.get_by_test_id("reprocess-dry-run")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert posts == [
        {"targets": {"filter": {"all_images": True}}, "scopes": ["open_vocab"], "dry_run": True}
    ], posts
    dialog.get_by_role("button", name="Cancel").click()

    labels = [b.inner_text().strip() for b in panel.get_by_test_id("reprocess-open").all()]
    assert labels == [
        "Run on all images…",
        "Re-run: Waiting for a pass…",
        "Re-run: Skipped by the gate…",
        "Re-run: Pass failed…",
    ], labels
    panel.get_by_test_id("reprocess-open").nth(2).click()
    with expect_handled(page, lambda r: r.url.endswith("/reprocess")):
        page.get_by_role("dialog", name="Reprocess").get_by_role("button", name="Check what would run").click()
    expect(page.get_by_role("dialog", name="Reprocess").get_by_test_id("reprocess-dry-run")).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert posts[1] == {
        "targets": {"filter": {"open_vocab_status": ["skipped_gate"]}},
        "scopes": ["open_vocab"],
        "dry_run": True,
    }, posts
