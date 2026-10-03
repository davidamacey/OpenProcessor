"""OpenProcessor W9 (VLM endpoint registry, local model, per-project
activation), any_domain_plan.md §7.8; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.

Every payload follows the f582aa05 contract (neutral names). The fail-closed
stub means any VLM request the page makes that a test did not expect fails
the test.

  1. `test_create_an_endpoint_validate_and_test_connection` — a served name
     issue shows after the debounced validate; Test connection posts the
     draft with `probe=true` and renders the served probe; Create posts
     `{name, description, body}` and opens the new endpoint's editor.
  2. `test_activate_an_external_endpoint_needs_the_acknowledgement` — the
     served warning and the checkbox; the first POST has no
     `acknowledge_external` and is refused (422); checked, the second body
     carries `acknowledge_external: true` and the active ref moves.
  3. `test_rollback_and_turn_off_send_expected_active`.
  4. `test_switch_the_local_model_shows_the_restart_then_follows_the_server`.
  5. `test_absent_when_the_registry_404s` — conftest's default 404: no
     Models card, the route says so, no other `/vlm/*` request fires.
"""

from __future__ import annotations

import copy
from typing import Any

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

CLEAN = {"ok": True, "errors": [], "warnings": [], "force_allowed": False}
WARNING = "Crops are sent to a service outside this deployment (api.example.com)."

LABELS = {
    "status": {
        "ready": "Ready",
        "unprobed": "Not probed yet",
        "probe_failed": "Probe failed",
        "unreachable": "Unreachable",
    },
    "locality": {
        "compose": "Same stack",
        "host": "This host",
        "private": "Private network",
        "external": "Outside this deployment",
        "unknown": "Unknown",
    },
    "source": {"env": "Set by the deployment", "stored": "Saved here"},
}


def summary(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "local_vlm",
        "source": "stored",
        "read_only": False,
        "revision": 3,
        "etag": "vlm:local_vlm:3",
        "description": "The GPU box",
        "base_url": "http://vlm.internal:8000/v1",
        "model": "example/vision-7b",
        "catalog_id": None,
        "locality": "private",
        "sends_images_externally": False,
        "warning": None,
        "api_key_ref": None,
        "api_key_present": False,
        "status": "ready",
        "last_probe_at": "2026-10-01T09:00:00Z",
        "active_in": ["default"],
    }
    base.update(over)
    return base


def listing() -> dict[str, Any]:
    return {
        "endpoints": [
            summary(),
            summary(
                name="cloud_vlm",
                revision=1,
                etag="vlm:cloud_vlm:1",
                description="A hosted model",
                base_url="https://api.example.com/v1",
                model="vendor/vision-large",
                locality="external",
                sends_images_externally=True,
                warning=WARNING,
                api_key_ref="CLOUD_VLM_KEY",
                api_key_present=True,
                status="unprobed",
                last_probe_at=None,
                active_in=[],
            ),
            summary(
                name="env_default",
                source="env",
                read_only=True,
                revision=None,
                etag="vlm:env_default",
                description="Set by the deployment",
                active_in=[],
            ),
        ],
        "config_revision": 12,
        "external_policy": "ack",
        "secret_refs": [
            {"ref": "CLOUD_VLM_KEY", "present": True, "choice": {"id": "CLOUD_VLM_KEY", "label": "CLOUD_VLM_KEY"}}
        ],
        "labels": LABELS,
    }


SCHEMA = {
    "groups": [{"id": "connection", "label": "Connection"}, {"id": "limits", "label": "Limits"}],
    "fields": [
        {"field": "base_url", "label": "Base URL", "group": "connection", "type": "string", "default": "", "help": "The OpenAI-compatible base URL."},
        {"field": "model", "label": "Model", "group": "connection", "type": "string", "default": "", "help": ""},
        {
            "field": "api_key_ref",
            "label": "API key",
            "group": "connection",
            "type": "string",
            "default": None,
            "choices_from": "secret_refs",
            "empty_choice": {"id": None, "label": "No key"},
            "help": "Names a secret on the host.",
        },
        {"field": "allow_external", "label": "Allow crops to leave the deployment", "group": "connection", "type": "bool", "default": False, "help": ""},
        {"field": "max_images_per_call", "label": "Images per call", "group": "limits", "type": "int", "default": 8, "min": 1, "max": 32, "help": ""},
    ],
}

BODY = {
    "base_url": "http://vlm.internal:8000/v1",
    "model": "example/vision-7b",
    "api_key_ref": None,
    "catalog_id": None,
    "allow_external": False,
    "json_mode": "auto",
    "max_images_per_call": 8,
    "open_images_per_call": 3,
    "requests_per_second": 500.0,
    "timeout_s": 240.0,
}


def doc(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "local_vlm",
        "source": "stored",
        "read_only": False,
        "revision": 3,
        "etag": "vlm:local_vlm:3",
        "description": "The GPU box",
        "body": copy.deepcopy(BODY),
        "created_at": "2026-09-30T10:00:00Z",
        "updated_at": "2026-10-01T08:00:00Z",
        "updated_by": None,
        "cloned_from": None,
        "validation": CLEAN,
        "api_key_present": False,
        "locality": "private",
        "sends_images_externally": False,
        "warning": None,
        "active_in": ["default"],
        "last_probe": None,
    }
    base.update(over)
    return base


def active(name: str | None = "local_vlm", revision: int | None = 3, previous: Any = None) -> dict[str, Any]:
    return {
        "axis": "vlm",
        "active": {"name": name, "revision": revision},
        "source": "stored" if name else "off",
        "activated_at": "2026-10-01T08:30:00Z",
        "previous": previous if previous is not None else {"name": "env_default", "revision": None},
        "config_revision": 12,
        "stale": False,
        "applied": [],
    }


def local(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "configured": True,
        "endpoint": "local_vlm",
        "served": {"model": "example/vision-7b", "root": "example/vision-7b", "catalog_id": "vision-7b", "max_model_len": 32768},
        "desired": None,
        "restart_required": False,
        "poll_after_s": None,
        "gpu_total_gb": 48,
        "can_restart_from_api": False,
        "reason": "The local model server is serving the selected model.",
    }
    base.update(over)
    return base


def catalog_entry(cid: str, label: str, serving: bool, desired: bool = False) -> dict[str, Any]:
    return {
        "id": cid,
        "choice": {"id": cid, "label": label},
        "hf_repo": f"example/{cid}",
        "family": "vision",
        "license": "Apache-2.0",
        "license_url": "https://example.com/license",
        "gated": False,
        "params_b": 7,
        "quantization": None,
        "context_max": 32768,
        "max_model_len": 16384,
        "max_images": 8,
        "vram_gb": 24,
        "disk_gb": 16,
        "status": "tested",
        "rank": 1,
        "multi_box_verified": True,
        "text_reading_verified": None,
        "fits": True,
        "serving": serving,
        "desired": desired,
    }


def vocabulary() -> dict[str, Any]:
    return {
        "detectors": [],
        "segmenters": [],
        "vlm": {"active": {"name": "local_vlm"}, "endpoints": []},
        "ocr": {"available": False, "pipeline_models": [], "det_models": [], "rec_models": []},
        "text_reader_modes": [],
        "registry_classes": [],
        "model_choices": [
            {
                "role": "vlm",
                "label": "VLM endpoint",
                "scope": "per_run",
                "current": "local_vlm",
                "dims": None,
                "choices": [{"id": "local_vlm", "label": "Local VLM"}],
                "settable": True,
                "settable_via": "Settings → Models",
            }
        ],
        "labels": {"scope": {"per_run": "Chosen per run"}},
    }


def serve_registry(stub: Any) -> dict[str, Any]:
    """The W9 reads every page makes. Returns the mutable state the write
    handlers in each test update."""
    state: dict[str, Any] = {"active": active(), "local": local()}
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/config/vocabulary(\?|$)", vocabulary())
    stub.on("GET", r"/vlm/endpoints(\?|$)", listing())
    stub.on("GET", r"/vlm/endpoints/schema$", copy.deepcopy(SCHEMA))
    stub.on("GET", r"/vlm/endpoints/active$", lambda _r, _m: (200, state["active"]))
    stub.on(
        "GET",
        r"/vlm/catalog$",
        lambda _r, _m: (
            200,
            {
                "entries": [
                    catalog_entry("vision-7b", "Vision 7B", state["local"]["served"]["catalog_id"] == "vision-7b"),
                    catalog_entry(
                        "vision-30b",
                        "Vision 30B",
                        state["local"]["served"]["catalog_id"] == "vision-30b",
                        bool(state["local"]["desired"]),
                    ),
                ],
                "local": state["local"],
                "labels": {"status": {"tested": "Tested", "to_verify": "To verify"}},
            },
        ),
    )
    stub.on("GET", r"/vlm/local$", lambda _r, _m: (200, state["local"]))
    return state


def row(page: Any, name: str) -> Any:
    return page.locator(f'[data-testid="vlm-endpoint-row"][data-name="{name}"]')


def test_create_an_endpoint_validate_and_test_connection(stub, page, app_url):
    serve_registry(stub)
    validated: list[Any] = []

    def validate(request: Any, _m: Any):
        body = request.post_data_json
        validated.append((request.url, body))
        issues = []
        if body["name"] == "local_vlm":
            issues = [
                {
                    "code": "name_conflict",
                    "id": "name_conflict",
                    "severity": "error",
                    "field": "name",
                    "message": "An endpoint named local_vlm already exists.",
                    "detail": {},
                    "bypassable": False,
                }
            ]
        probe = None
        if request.url.endswith("probe=true"):
            probe = {
                "ok": True,
                "probed_at": "2026-10-01T09:30:00Z",
                "latency_ms": 84,
                "models_listed": ["example/vision-7b"],
                "model_listed": True,
                "root": "example/vision-7b",
                "max_model_len": 32768,
                "vision_ok": True,
                "json_mode_supported": True,
                "reasoning_channel": False,
                "image_tokens": 256,
                "max_images_ok": True,
                "issues": [],
            }
        return (
            200,
            {
                "validation": {"ok": not issues, "errors": issues, "warnings": [], "force_allowed": False},
                "locality": "private",
                "sends_images_externally": False,
                "probe": probe,
            },
        )

    stub.on("POST", r"/vlm/endpoints/validate$", validate)
    created: list[Any] = []

    def create(request: Any, _m: Any):
        body = request.post_data_json
        created.append(body)
        return (201, doc(name=body["name"], revision=1, body=body["body"], description=body["description"], active_in=[]))

    stub.on("POST", r"/vlm/endpoints$", create)
    stub.on("GET", r"/vlm/endpoints/new_vlm$", doc(name="new_vlm", revision=1, description="Fresh", active_in=[]))
    stub.on(
        "GET",
        r"/vlm/endpoints/new_vlm/revisions$",
        {"name": "new_vlm", "revisions": [{"revision": 1, "saved_at": "2026-10-01T10:00:00Z", "cloned_from": None, "description": "Fresh"}]},
    )

    page.goto(f"{app_url}/p/default/settings/models/new-endpoint")
    name = page.get_by_test_id("vlm-new-name")
    name.wait_for(timeout=ACTION_TIMEOUT_MS)

    # The name is checked with the draft: the served issue shows under it.
    with page.expect_request(lambda r: r.method == "POST" and "/vlm/endpoints/validate" in r.url):
        name.fill("local_vlm")
    expect(page.get_by_test_id("config-issue").first).to_contain_text(
        "An endpoint named local_vlm already exists.", timeout=ACTION_TIMEOUT_MS
    )
    assert validated[-1][1]["name"] == "local_vlm"
    assert validated[-1][1]["body"]["api_key_ref"] is None

    name.fill("new_vlm")
    page.locator('[data-field="base_url"] input').fill("http://vlm.internal:8000/v1")
    page.locator('[data-field="model"] input').fill("example/vision-7b")

    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/vlm/endpoints/validate?probe=true")):
        page.get_by_test_id("vlm-test-connection").click()
    probe = page.get_by_test_id("vlm-probe")
    expect(probe).to_contain_text("probe ok", timeout=ACTION_TIMEOUT_MS)
    expect(probe).to_contain_text("example/vision-7b")
    assert validated[-1][1]["name"] == "new_vlm"
    assert validated[-1][1]["body"]["model"] == "example/vision-7b"

    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/vlm/endpoints")):
        page.get_by_test_id("vlm-create").click()
    assert created[0]["name"] == "new_vlm"
    assert created[0]["description"] == ""
    assert created[0]["body"]["base_url"] == "http://vlm.internal:8000/v1"
    assert created[0]["body"]["model"] == "example/vision-7b"
    assert set(created[0]) == {"name", "description", "body"}
    # The new endpoint's editor opens.
    expect(page.get_by_test_id("config-meta")).to_contain_text("revision 1", timeout=ACTION_TIMEOUT_MS)
    assert page.url.endswith("/settings/models/vlm/new_vlm")


def test_activate_an_external_endpoint_needs_the_acknowledgement(stub, page, app_url):
    state = serve_registry(stub)
    bodies: list[Any] = []

    def activate(request: Any, _m: Any):
        body = request.post_data_json
        bodies.append(body)
        if not body.get("acknowledge_external"):
            return (
                422,
                {
                    "detail": {
                        "error": "vlm_external_not_acknowledged",
                        "message": "cloud_vlm sends crops outside this deployment; acknowledge it first.",
                        "endpoint": "cloud_vlm",
                        "activate_via": "settings/models",
                    }
                },
            )
        state["active"] = active("cloud_vlm", 1, previous={"name": "local_vlm", "revision": 3})
        return (200, state["active"])

    stub.on("POST", r"/vlm/endpoints/cloud_vlm/activate$", activate)

    page.goto(f"{app_url}/p/default/settings/models")
    row(page, "cloud_vlm").wait_for(timeout=ACTION_TIMEOUT_MS)
    # The served warning and the key reference (never a key) are on the row.
    expect(row(page, "cloud_vlm").get_by_test_id("vlm-external-warning")).to_have_text(WARNING)
    expect(row(page, "cloud_vlm").get_by_test_id("vlm-key-ref")).to_contain_text("CLOUD_VLM_KEY")
    expect(row(page, "local_vlm").get_by_test_id("vlm-external-warning")).to_have_count(0)

    row(page, "cloud_vlm").get_by_test_id("vlm-activate").click()
    dialog = page.get_by_role("dialog", name="Activate cloud_vlm revision 1")
    expect(dialog.get_by_test_id("activate-external-warning")).to_have_text(WARNING)
    expect(dialog.get_by_test_id("activate-ack")).to_be_visible()

    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/cloud_vlm/activate")):
        dialog.get_by_role("button", name="Activate", exact=True).click()
    expect(dialog.get_by_test_id("activate-error")).to_contain_text(
        "acknowledge it first", timeout=ACTION_TIMEOUT_MS
    )
    assert "acknowledge_external" not in bodies[0]
    assert bodies[0] == {"revision": 1, "expected_active": {"name": "local_vlm", "revision": 3}, "force": False}

    dialog.get_by_test_id("activate-ack").check()
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/cloud_vlm/activate")):
        dialog.get_by_role("button", name="Activate", exact=True).click()
    assert bodies[1] == {
        "revision": 1,
        "expected_active": {"name": "local_vlm", "revision": 3},
        "force": False,
        "acknowledge_external": True,
    }
    expect(page.get_by_test_id("active-ref")).to_have_text("cloud_vlm r1", timeout=ACTION_TIMEOUT_MS)


def test_rollback_and_turn_off_send_expected_active(stub, page, app_url):
    state = serve_registry(stub)
    rollbacks: list[Any] = []
    deactivations: list[Any] = []

    def rollback(request: Any, _m: Any):
        rollbacks.append(request.post_data_json)
        state["active"] = active("env_default", None, previous={"name": "local_vlm", "revision": 3})
        return (200, state["active"])

    def deactivate(request: Any, _m: Any):
        deactivations.append(request.post_data_json)
        state["active"] = active(None, None, previous={"name": "env_default", "revision": None})
        return (200, state["active"])

    stub.on("POST", r"/vlm/endpoints/active/rollback$", rollback)
    stub.on("POST", r"/vlm/endpoints/deactivate$", deactivate)

    page.goto(f"{app_url}/p/default/settings/models")
    expect(page.get_by_test_id("active-ref")).to_have_text("local_vlm r3", timeout=ACTION_TIMEOUT_MS)

    page.get_by_role("button", name="Roll back to env_default").click()
    dialog = page.get_by_role("dialog", name="Roll back the VLM endpoint")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/active/rollback")):
        dialog.get_by_role("button", name="Roll back").click()
    assert rollbacks == [{"expected_active": {"name": "local_vlm", "revision": 3}}]
    expect(page.get_by_test_id("active-ref")).to_have_text("env_default", timeout=ACTION_TIMEOUT_MS)

    page.get_by_test_id("active-deactivate").click()
    dialog = page.get_by_role("dialog", name="Turn off the VLM for this project")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/vlm/endpoints/deactivate")):
        dialog.get_by_role("button", name="Turn off").click()
    assert deactivations == [{"expected_active": {"name": "env_default", "revision": None}}]
    expect(page.get_by_test_id("active-ref")).to_contain_text("off", timeout=ACTION_TIMEOUT_MS)


def test_switch_the_local_model_shows_the_restart_then_follows_the_server(stub, page, app_url):
    state = serve_registry(stub)
    selects: list[Any] = []

    def select(request: Any, _m: Any):
        selects.append(request.post_data_json)
        state["local"] = local(
            restart_required=True,
            poll_after_s=1,
            reason="A restart is needed to serve vision-30b.",
            desired={
                "catalog_id": "vision-30b",
                "requested_at": "2026-10-01T10:00:00Z",
                "command": "docker compose up -d vlm",
            },
        )
        return (202, state["local"])

    stub.on("POST", r"/vlm/local/select$", select)

    page.goto(f"{app_url}/p/default/settings/models")
    switch = page.locator('[data-testid="local-vlm-row"][data-id="vision-30b"]').get_by_test_id("local-vlm-switch")
    switch.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("local-vlm-restart")).to_have_count(0)

    switch.click()
    dialog = page.get_by_role("dialog", name="Switch the local model to Vision 30B")
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/vlm/local/select")):
        dialog.get_by_role("button", name="Switch", exact=True).click()
    assert selects == [{"catalog_id": "vision-30b"}]

    banner = page.get_by_test_id("local-vlm-restart")
    expect(banner).to_contain_text("A restart is needed to serve vision-30b.", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("local-vlm-command")).to_have_text("docker compose up -d vlm")
    # Nothing says it switched: the row is requested, not serving.
    target_row = page.locator('[data-testid="local-vlm-row"][data-id="vision-30b"]')
    expect(target_row.get_by_test_id("local-vlm-desired")).to_be_visible()
    expect(target_row.get_by_test_id("local-vlm-serving")).to_have_count(0)

    # The server restarts into the requested model; the next poll shows it.
    state["local"] = local(
        served={"model": "example/vision-30b", "root": "example/vision-30b", "catalog_id": "vision-30b", "max_model_len": 32768},
        reason="The local model server is serving the selected model.",
    )
    expect(target_row.get_by_test_id("local-vlm-serving")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("local-vlm-restart")).to_have_count(0)


def test_absent_when_the_registry_404s(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})

    with page.expect_response(lambda r: r.url.endswith("/vlm/endpoints") and r.status == 404):
        page.goto(f"{app_url}/p/default/settings")
    expect(page.get_by_role("heading", name="Deployment defaults")).to_be_visible()
    expect(page.get_by_test_id("vlm-models-card")).to_have_count(0)

    page.goto(f"{app_url}/p/default/settings/models")
    expect(page.get_by_test_id("vlm-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.goto(f"{app_url}/p/default/settings/models/vlm/local_vlm")
    expect(page.get_by_test_id("vlm-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.goto(f"{app_url}/p/default/settings/models/new-endpoint")
    expect(page.get_by_test_id("vlm-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    vlm_calls = [p for _m, p in stub.handled if "/vlm/" in p]
    assert vlm_calls and all(p.endswith("/vlm/endpoints") for p in vlm_calls), vlm_calls
