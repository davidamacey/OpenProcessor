"""OpenProcessor v0.4.0 embedding state (docs/design/
v040-backend-deltas-ui-plan-2026-10-03.md §7.3).

  1. `test_failed_badge_on_a_card` -- `/clusters/[id]` shows the badge for a
     served `failed` embedding state, none for `embedded`.
  2. `test_dashboard_summary_embed_n_opens_the_served_request` -- the
     detections summary renders served counts; "Embed N" sends the served
     `suggested_reprocess` as served, dry run first.
  3. `test_reprocess_embed_options_and_served_detail` -- embed options are
     sent only once touched; the served per-scope detail is shown.
  4. `test_unembedded_banner_on_an_ordered_cluster_view`.
  5. `test_failed_auto_label_job_shows_its_error_and_stage`.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from fixtures.wire import make_item
from test_dataset_import import FORMATS

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

CLUSTERS = {
    "items": [
        {
            "cluster_id": 1,
            "cluster_kind": "class",
            "size": 2,
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

SUGGESTED = {
    "targets": {"filter": {"embedding_state": ["not_selected", "failed"]}},
    "scopes": ["embed"],
    "embed": {"only_missing": True},
    "dry_run": True,
}

SUMMARY = {
    "total": 120,
    "embedding": {
        "embedded": 100,
        "not_embedded": 20,
        "by_state": {
            "embedded": 100,
            "not_selected": 15,
            "deferred": 0,
            "failed": 5,
            "unknown": 0,
        },
    },
    "by_label": [
        {
            "name": "widget",
            "count": 80,
            "embedding": {
                "embedded": 70,
                "not_embedded": 10,
                "by_state": {
                    "embedded": 70,
                    "not_selected": 10,
                    "deferred": 0,
                    "failed": 0,
                    "unknown": 0,
                },
            },
        }
    ],
    "labels_truncated": True,
    "suggested_reprocess": SUGGESTED,
}

IDLE_JOB = {
    "job_id": "job-0",
    "status": "idle",
    "stage": "",
    "processed": 0,
    "total": 0,
    "started_at": 0,
    "finished_at": 0,
    "error": None,
    "result": {},
    "args": {},
    "eta_seconds": None,
    "elapsed_seconds": 0,
}


DATASET_STATS = {
    "as_of": "2026-10-03T00:00:00Z",
    "total_crops": 120,
    "validated": 0,
    "test_holdout": 0,
    "by_source": [],
    "labeled": {"by_human": 0, "by_vlm": 0, "by_classifier": 0, "other": 0},
    "regions": {},
    "unlabeled": {
        "pending_detection": 0,
        "pending_verification": 0,
        "no_label_source": 0,
        "vlm_no_class": 0,
        "by_proposal": 0,
    },
    "in_progress": {"region_drain_total_unfinished": 0, "region_stall_reason": None},
    "clusters": {
        "last_run_at": None,
        "cluster_count": 0,
        "residual_count": 0,
        "noise_count": 0,
        "method": None,
    },
    "embedding": {
        "embedded": 100,
        "not_embedded": 20,
        "by_state": {"embedded": 100, "not_selected": 15, "failed": 5, "unknown": 0},
    },
}


def _shots(page, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    for width in (1600, 800):
        page.set_viewport_size({"width": width, "height": 1000})
        page.screenshot(path=str(SHOTS / f"{name}-{width}.png"), full_page=True)
    page.set_viewport_size({"width": 1280, "height": 720})


def _crop(i: int, state: str | None) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=1,
        class_name="widget",
        cluster_id=1,
        label_validated=False,
        label_source="model_suggestion",
        embedding_state=state,
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def _cluster_stubs(stub, page_body: dict[str, Any]) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", page_body)


def test_failed_badge_on_a_card(stub, page, app_url):
    _cluster_stubs(
        stub,
        {
            "total": 2,
            "page": 1,
            "page_size": 60,
            "crops": [_crop(0, "failed"), _crop(1, "embedded")],
        },
    )
    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    badges = page.get_by_test_id("embedding-state-badge")
    expect(badges).to_have_count(1)
    expect(badges.first).to_have_text("Embed failed")
    expect(badges.first).to_have_attribute("title", re.compile("No vector: encoder failed"))
    _shots(page, "failed-badge")


def test_unembedded_banner_on_an_ordered_cluster_view(stub, page, app_url):
    stub.on("GET", r"/datasets/formats(\?|$)", copy.deepcopy(FORMATS))
    _cluster_stubs(
        stub,
        {
            "total": 2,
            "page": 1,
            "page_size": 60,
            "crops": [_crop(0, "embedded"), _crop(1, "deferred")],
            "n_unembedded": 1,
            "suggested_reprocess": SUGGESTED,
        },
    )
    page.goto(f"{app_url}/p/default/clusters/1")
    banner = page.get_by_test_id("unembedded-banner")
    banner.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(banner).to_contain_text("1 items in scope have no vector and are not ranked")
    expect(banner.get_by_test_id("reprocess-open")).to_have_text("Embed them")
    _shots(page, "unembedded-banner")


def test_dashboard_summary_embed_n_opens_the_served_request(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})
    stub.on("GET", r"/detections/summary(\?|$)", copy.deepcopy(SUMMARY))
    stub.on("GET", r"/stats/dataset(\?|$)", copy.deepcopy(DATASET_STATS))
    stub.on("GET", r"/datasets/formats(\?|$)", copy.deepcopy(FORMATS))
    reprocess_bodies: list[dict] = []

    def reprocess(request, _match):
        body = request.post_data_json
        reprocess_bodies.append(body)
        return {
            "dry_run": True,
            "scopes": [
                {
                    "scope": "embed",
                    "selected": 20,
                    "detail": {
                        "items": 120,
                        "without_vector": 20,
                        "to_embed": 20,
                        "estimated_cost": 80.0,
                        "region_boxes_to_embed": False,
                    },
                }
            ],
            "job": None,
            "items": [],
        }

    stub.on("POST", r"/reprocess$", reprocess)

    page.goto(f"{app_url}/p/default/dashboard")
    summary = page.get_by_test_id("detections-summary")
    summary.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("detections-total")).to_contain_text("120 detections")
    expect(summary).to_contain_text("No vector: encoder failed 5")
    expect(page.get_by_test_id("detections-truncated")).to_be_visible()
    _shots(page, "dashboard-summary")

    page.get_by_test_id("reprocess-open").click()
    dialog = page.get_by_role("dialog", name="Reprocess")
    assert not reprocess_bodies, "opening the dialog must not send anything"
    dialog.get_by_role("button", name="Check what would run").click()
    detail = dialog.get_by_test_id("reprocess-detail")
    expect(detail).to_contain_text("To embed", timeout=ACTION_TIMEOUT_MS)
    expect(detail).to_contain_text("Estimated cost")
    expect(detail).to_contain_text("no")
    _shots(page, "reprocess-detail")
    assert reprocess_bodies == [SUGGESTED], "the served request is sent as served"


def test_reprocess_embed_options_and_served_detail(stub, page, app_url):
    stub.on("GET", r"/datasets/formats(\?|$)", copy.deepcopy(FORMATS))
    _cluster_stubs(
        stub,
        {
            "total": 2,
            "page": 1,
            "page_size": 60,
            "crops": [_crop(0, "failed"), _crop(1, "deferred")],
        },
    )
    bodies: list[dict] = []

    def reprocess(request, _match):
        bodies.append(request.post_data_json)
        return {
            "dry_run": True,
            "scopes": [{"scope": "embed", "selected": 2}],
            "job": None,
            "items": [],
        }

    stub.on("POST", r"/reprocess$", reprocess)
    page.goto(f"{app_url}/p/default/clusters/1")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press("a")
    page.get_by_test_id("reprocess-open").first.click()
    dialog = page.get_by_role("dialog", name="Reprocess")
    dialog.get_by_label("Embed", exact=True).check()
    options = dialog.get_by_test_id("reprocess-embed-options")
    expect(options).to_be_visible()
    dialog.get_by_role("button", name="Check what would run").click()
    expect(dialog.get_by_test_id("reprocess-dry-run")).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert "embed" not in bodies[-1], "untouched options send no embed key"

    options.get_by_label("Only items without a vector").check()
    options.get_by_label("Frame").check()
    _shots(page, "reprocess-embed-options")
    dialog.get_by_role("button", name="Check what would run").click()
    page.wait_for_function("true")
    expect(dialog.get_by_test_id("reprocess-dry-run")).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert bodies[-1]["embed"] == {"only_missing": True, "parts": ["frame"]}


def test_failed_auto_label_job_shows_its_error_and_stage(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})
    failed = {
        **IDLE_JOB,
        "job_id": "j1",
        "status": "failed",
        "stage": "embed_missing",
        "finished_at": 1,
        "error": "embedding service unavailable",
        "args": {"embed_missing": True},
        "result": {"stages": {"embed_missing": {"status": "error", "reason": "encoder down"}}},
    }
    stub.on("GET", r"/pipeline/auto_label/status", failed)
    stub.on("GET", r"/stats/dataset(\?|$)", copy.deepcopy(DATASET_STATS))
    stub.on("GET", r"/detections/summary(\?|$)", copy.deepcopy(SUMMARY))
    page.goto(f"{app_url}/p/default/dashboard")
    expect(page.get_by_text("Error: embedding service unavailable")).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    row = page.get_by_test_id("last-run-stages").locator("tr[data-stage-status=error]")
    expect(row).to_contain_text("embedding items without a vector")
    expect(row).to_contain_text("encoder down")
    _shots(page, "failed-auto-label")
