"""OpenProcessor v0.4.0 "Create classes from the detector" on `/classes`
(docs/design/v040-backend-deltas-ui-plan-2026-10-03.md §7.3).

A dry run first (served created / skipped / conflicts, reasons shown), the
real call only after the confirm; absent when the deployment reports no
detector.
"""

from __future__ import annotations

import copy
from pathlib import Path

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_ingest_policy import DETECTOR, POLICY, _ingest_config

SHOTS = Path(__file__).resolve().parents[2] / "artifacts_local" / "v040-ui" / "detector"

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 3,
        "deprecated": False,
    },
]

DRY = {
    "dry_run": True,
    "detector_model": "widget_detector_v1",
    "created": [{"class_id": None, "name": "tag_plate", "detector_label": "tag plate"}],
    "skipped": [{"name": "widget", "detector_label": "widget", "reason": "exists"}],
    "conflicts": [
        {"detector_label": "??", "class_id_in_detector": 7, "reason": "unnamed_label"}
    ],
}

SUMMARY_EMPTY = {
    "total_pending": 0,
    "without_term": 0,
    "top_terms": [],
    "flagged_terms": [],
    "term_rules": {},
}


def _stubs(stub, detector):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", SUMMARY_EMPTY)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    config = _ingest_config(POLICY)
    config["detector"] = detector
    stub.on("GET", r"/ingest/config(\?|$)", config)


def _shots(page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for width in (1600, 800):
        page.set_viewport_size({"width": width, "height": 1000})
        page.screenshot(path=str(SHOTS / f"{name}-{width}.png"), full_page=True)
    page.set_viewport_size({"width": 1280, "height": 720})


def test_dry_run_then_create_behind_a_confirm(stub, page, app_url):
    _stubs(stub, DETECTOR)
    seeds: list[dict] = []

    def seed(request, _match):
        body = request.post_data_json
        seeds.append(body)
        out = copy.deepcopy(DRY)
        out["dry_run"] = body["dry_run"]
        if not body["dry_run"]:
            out["created"][0]["class_id"] = 11
        return out

    stub.on("POST", r"/classes/seed_from_detector$", seed)

    page.goto(f"{app_url}/p/default/classes")
    page.get_by_test_id("seed-panel").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("seed-panel").locator("summary").click()
    page.get_by_test_id("seed-preview").click()

    created = page.get_by_test_id("seed-created")
    expect(created).to_contain_text("would create", timeout=ACTION_TIMEOUT_MS)
    expect(created).to_contain_text("tag_plate")
    expect(created).to_contain_text("tag plate")
    expect(page.get_by_test_id("seed-skipped")).to_contain_text("Exists")
    expect(page.get_by_test_id("seed-conflicts")).to_contain_text("Unnamed label")
    _shots(page, "seed-dry-run")

    assert seeds == [{"dry_run": True}], "the preview is a dry run with no names"

    page.get_by_test_id("seed-create").click()
    assert len(seeds) == 1, "nothing is created before the confirm"
    page.get_by_role("dialog").get_by_role("button", name="Create 1 classes").click()
    page.wait_for_function("document.querySelector('[data-testid=seed-result]') === null")
    assert seeds[-1] == {"dry_run": False}


def test_a_served_refusal_is_shown_verbatim(stub, page, app_url):
    _stubs(stub, DETECTOR)
    stub.on(
        "POST",
        r"/classes/seed_from_detector$",
        (
            503,
            {
                "detail": {
                    "error": "detector_unavailable",
                    "message": "The detector service is not reachable.",
                }
            },
        ),
    )
    page.goto(f"{app_url}/p/default/classes")
    page.get_by_test_id("seed-panel").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("seed-panel").locator("summary").click()
    page.get_by_test_id("seed-preview").click()
    expect(page.get_by_test_id("seed-error")).to_contain_text(
        "The detector service is not reachable.", timeout=10000
    )


def test_absent_without_a_detector(stub, page, app_url):
    _stubs(stub, None)
    page.goto(f"{app_url}/p/default/classes")
    page.get_by_text("Classes").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)
    expect(page.get_by_test_id("seed-panel")).to_have_count(0)
