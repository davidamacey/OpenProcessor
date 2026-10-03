"""OpenProcessor v0.4.0 ingest policy page (`/settings/ingest-policy`,
docs/design/v040-backend-deltas-ui-plan-2026-10-03.md §7.2).

  1. `test_load_edit_preview_and_save_behind_a_confirm` -- the served policy
     loads, an edit triggers the cost preview (stored detections only),
     nothing is PUT until the confirm, the PUT carries `expected_revision`,
     and a served `unknown_names` list is shown as a warning.
  2. `test_a_409_offers_reload_and_keep_my_edits`.
  3. `test_settings_card_links_to_the_page`.
"""

from __future__ import annotations

from pathlib import Path

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

SHOTS = Path(__file__).resolve().parents[2] / "artifacts_local" / "v040-ui" / "detector"

DETECTOR = {
    "model": "widget_detector_v1",
    "version": "2",
    "input_size": 640,
    "assigns_class": False,
    "confidence_floor_applies": False,
    "n_labels": 2,
    "labels": [
        {"class_id": 0, "name": "widget", "slug": "widget"},
        {"class_id": 1, "name": "tag plate", "slug": "tag_plate"},
    ],
}

POLICY = {
    "detect": {
        "min_confidence": None,
        "min_box_area_frac": None,
        "max_per_image": None,
        "classes": None,
        "exclude_classes": [],
        "class_resolution": "proposal",
    },
    "embedding": {
        "min_confidence": None,
        "min_box_area_frac": None,
        "max_per_image": None,
        "mode": "all",
        "classes": [],
    },
    "detector": None,
    "revision": 4,
}

PREVIEW = {
    "total_items": 1000,
    "scanned": 800,
    "truncated": True,
    "would_embed": 400,
    "would_not_embed": 600,
    "estimated_vector_mb": 1.6,
    "by_class": [
        {"name": "widget", "would_embed": 300, "would_not_embed": 100},
        {"name": "tag plate", "would_embed": 100, "would_not_embed": 500},
    ],
}


def _ingest_config(policy):
    return {
        "upload": {
            "enabled": True,
            "max_images_per_request": 128,
            "max_bytes_per_request": 268435456,
            "accepted_extensions": [".jpg"],
            "persists_bytes": True,
        },
        "batch": {"enabled": True, "max_items": 256, "source_roots": []},
        "region_drain": {"poll_interval_s": 10, "stable_polls": 3},
        "detector": DETECTOR,
        "policy": policy,
    }


def _shots(page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for width in (1600, 800):
        page.set_viewport_size({"width": width, "height": 1000})
        page.screenshot(path=str(SHOTS / f"{name}-{width}.png"), full_page=True)
    page.set_viewport_size({"width": 1280, "height": 720})


def _stub_policy(stub, put_handler):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/ingest/config(\?|$)", _ingest_config(POLICY))
    stub.on("GET", r"/ingest/policy(\?|$)", POLICY)
    stub.on("POST", r"/ingest/policy/preview(\?|$)", PREVIEW)
    stub.on("PUT", r"/ingest/policy(\?|$)", put_handler)


def test_load_edit_preview_and_save_behind_a_confirm(stub, page, app_url):
    _stub_policy(
        stub,
        lambda _req, _m: {**POLICY, "revision": 5, "unknown_names": ["gizmo"]},
    )
    page.goto(f"{app_url}/p/default/settings/ingest-policy")
    page.get_by_test_id("ingest-policy-form").wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("policy-mode")).to_have_value("all")

    page.get_by_test_id("policy-mode").select_option("selected")
    expect(page.get_by_test_id("policy-preview-summary")).to_contain_text(
        "400 of 1000 stored detections would be embedded, about 1.6 MB",
        timeout=ACTION_TIMEOUT_MS,
    )
    expect(page.get_by_test_id("policy-preview-truncated")).to_contain_text(
        "Estimated from 800 detections"
    )
    expect(page.get_by_test_id("policy-preview")).to_contain_text(
        "changes only future ingests"
    )
    _shots(page, "ingest-policy-preview")

    puts = [c for c in stub.calls if c[0] == "PUT"]
    assert not puts, "nothing may be saved before the confirm"

    page.get_by_test_id("policy-save").click()
    page.get_by_role("dialog").get_by_role("button", name="Save policy").click()
    expect(page.get_by_test_id("policy-unknown-names")).to_contain_text(
        "gizmo", timeout=ACTION_TIMEOUT_MS
    )
    _shots(page, "ingest-policy-unknown-names")

    puts = [c for c in stub.calls if c[0] == "PUT"]
    assert len(puts) == 1
    body = puts[0][2]
    assert body["expected_revision"] == 4
    assert body["embedding"]["mode"] == "selected"


def test_a_409_offers_reload_and_keep_my_edits(stub, page, app_url):
    _stub_policy(
        stub,
        lambda _req, _m: (
            409,
            {
                "detail": {
                    "error": "revision_conflict",
                    "message": "The ingest policy is at revision 6.",
                }
            },
        ),
    )
    page.goto(f"{app_url}/p/default/settings/ingest-policy")
    page.get_by_test_id("ingest-policy-form").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("policy-mode").select_option("lazy")
    page.get_by_test_id("policy-save").click()
    page.get_by_role("dialog").get_by_role("button", name="Save policy").click()
    conflict = page.get_by_test_id("policy-conflict")
    expect(conflict).to_contain_text("The ingest policy is at revision 6.", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("policy-reload")).to_be_visible()
    expect(page.get_by_test_id("policy-keep")).to_be_visible()
    _shots(page, "ingest-policy-conflict")

    page.get_by_test_id("policy-reload").click()
    expect(page.get_by_test_id("policy-mode")).to_have_value("all")
    expect(conflict).to_have_count(0)


def test_settings_card_links_to_the_page(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/ingest/policy(\?|$)", POLICY)
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})
    page.goto(f"{app_url}/p/default/settings")
    card = page.get_by_test_id("ingest-policy-card")
    card.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("ingest-policy-card-mode")).to_contain_text("Embedding: all")
    expect(card.get_by_role("link", name="Open ingest policy")).to_have_attribute(
        "href", "/p/default/settings/ingest-policy"
    )
