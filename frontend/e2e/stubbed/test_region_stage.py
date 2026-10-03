"""OpenProcessor v0.4.0 region stage panel on /ingest,
docs/design/v040-backend-deltas-ui-plan-2026-10-03.md §6.6.

  1. `test_panel_only_with_a_region_profile` — no profile: no panel and no
     `/region_stage` request.
  2. `test_pause_needs_a_confirm_and_resume_follows` — nothing is sent until
     the confirm; the served state is adopted.
  3. `test_rerun_gate_skipped_sends_the_served_request`.

Screenshots (1600 and 800 px) land in `artifacts_local/v040-ui/open-vocab/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conftest import ACTION_TIMEOUT_MS, expect_handled
from playwright.sync_api import expect

from test_ingest import _base_ingest_stubs as _serve_ingest_baseline

SHOT_DIR = Path(__file__).resolve().parents[2] / "artifacts_local" / "v040-ui" / "open-vocab"

RERUN = {
    "targets": {"filter": {"all_images": False, "region_gate_skipped": True}},
    "scopes": ["region"],
    "dry_run": True,
}


def stage(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "project": "default",
        "paused": False,
        "paused_since": None,
        "pipeline_paused": False,
        "counts": {"pending_detection": 4, "pending_verification": 1, "gate_skipped": 7},
        "rerun_skipped": RERUN,
    }
    base.update(over)
    return base


def shoot(page: Any, name: str) -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    original = page.viewport_size
    for w in (1600, 800):
        page.set_viewport_size({"width": w, "height": 1400})
        page.wait_for_timeout(200)
        overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 1, f"{name}@{w}: horizontal overflow {overflow}px"
        page.get_by_test_id("region-stage-panel").screenshot(path=str(SHOT_DIR / f"{name}-{w}.png"))
    if original:
        page.set_viewport_size(original)


def test_panel_only_with_a_region_profile(stub, page, app_url):
    _serve_ingest_baseline(stub)
    stub.on("GET", r"/health$", {"status": "ok", "region_profile": None})
    page.goto(f"{app_url}/p/default/ingest")
    expect(page.get_by_text("Ingest").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_selector('[data-testid="ingest-status-table"], table, h2', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("region-stage-panel")).to_have_count(0)
    assert not [p for _m, p in stub.handled if "/region_stage" in p]


def test_pause_needs_a_confirm_and_resume_follows(stub, page, app_url):
    _serve_ingest_baseline(stub)
    state = {"paused": False}

    def current(_r: Any, _m: Any):
        return (200, stage(paused=state["paused"], paused_since="2026-10-03T08:00:00Z" if state["paused"] else None))

    stub.on("GET", r"/region_stage$", current)
    posts: list[str] = []

    def pause(_r: Any, _m: Any):
        posts.append("pause")
        state["paused"] = True
        return current(_r, _m)

    def resume(_r: Any, _m: Any):
        posts.append("resume")
        state["paused"] = False
        return current(_r, _m)

    stub.on("POST", r"/region_stage/pause$", pause)
    stub.on("POST", r"/region_stage/resume$", resume)

    page.goto(f"{app_url}/p/default/ingest")
    panel = page.get_by_test_id("region-stage-panel")
    expect(panel.get_by_test_id("region-stage-state")).to_have_text("running", timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("region-stage-gate-skipped")).to_have_text("7")
    shoot(page, "region-stage-running")

    panel.get_by_test_id("region-stage-toggle").click()
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("Queued items stay pending; nothing is lost.")
    assert posts == []
    with expect_handled(page, lambda r: r.url.endswith("/region_stage/pause")):
        dialog.get_by_role("button", name="Pause", exact=True).click()
    expect(panel.get_by_test_id("region-stage-state")).to_have_text("paused", timeout=ACTION_TIMEOUT_MS)
    expect(panel.get_by_test_id("region-stage-since")).to_be_visible()
    shoot(page, "region-stage-paused")

    panel.get_by_test_id("region-stage-toggle").click()
    with expect_handled(page, lambda r: r.url.endswith("/region_stage/resume")):
        page.get_by_role("dialog").get_by_role("button", name="Resume", exact=True).click()
    expect(panel.get_by_test_id("region-stage-state")).to_have_text("running", timeout=ACTION_TIMEOUT_MS)
    assert posts == ["pause", "resume"], posts


def test_rerun_gate_skipped_sends_the_served_request(stub, page, app_url):
    _serve_ingest_baseline(stub)
    stub.on("GET", r"/region_stage$", stage())
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
        return (200, {"dry_run": True, "scopes": [{"scope": "region", "selected": 7, "queued": 0}]})

    stub.on("POST", r"/reprocess$", reprocess)
    page.goto(f"{app_url}/p/default/ingest")
    button = page.get_by_test_id("region-stage-panel").get_by_test_id("reprocess-open")
    expect(button).to_have_text("Re-run gate-skipped (7)…", timeout=ACTION_TIMEOUT_MS)
    button.click()
    with expect_handled(page, lambda r: r.url.endswith("/reprocess")):
        page.get_by_role("dialog", name="Reprocess").get_by_role("button", name="Check what would run").click()
    assert posts == [RERUN], posts
