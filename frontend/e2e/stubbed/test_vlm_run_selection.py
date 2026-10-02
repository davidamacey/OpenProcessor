"""OpenProcessor W9 per-run VLM selection and the `/settings` dropdown,
any_domain_plan.md §7.8; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §3.3.

  1. `test_dashboard_auto_label_with_a_picked_external_endpoint` — the
     assist bar's picker shows the served warning and the acknowledgement;
     the start request carries `vlm=` and `acknowledge_external=true`.
  2. `test_settings_disables_an_unacknowledged_external_entry`.
  3. `test_no_picker_without_a_vlm_axis` — today's `/methods`: no picker,
     and the unscoped start request carries neither param.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

WARNING = "Crops are sent to a service outside this deployment (api.example.com)."

BASE_STRATEGIES = [
    {"id": "ivf", "axis": "cluster", "label": "FAISS IVF-512 (production)", "status": "stable", "default": True},
    {"id": "recent", "axis": "sort", "label": "Recent first", "status": "stable", "default": True},
]

VLM_STRATEGIES = [
    {
        "id": "local_vlm",
        "axis": "vlm",
        "label": "Local VLM",
        "status": "stable",
        "default": True,
        "settable": True,
        "endpoint_status": "ready",
        "endpoint_status_label": "Ready",
        "sends_images_externally": False,
        "warning": None,
        "default_ack_recorded": None,
        "per_run_ack_required": False,
    },
    {
        "id": "cloud_vlm",
        "axis": "vlm",
        "label": "Cloud VLM",
        "status": "stable",
        "settable": True,
        "endpoint_status": "unprobed",
        "endpoint_status_label": "Not probed yet",
        "sends_images_externally": True,
        "warning": WARNING,
        "default_ack_recorded": False,
        "per_run_ack_required": True,
    },
    {"id": "off", "axis": "vlm", "label": "Off", "status": "stable", "settable": True},
]

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
    "as_of": "2026-09-24T00:00:00Z",
    "total_crops": 3,
    "validated": 0,
    "test_holdout": 0,
    "by_source": [],
    "labeled": {"by_human": 0, "by_vlm": 0, "by_classifier": 0, "other": 0},
    "regions": {"total_detected": 0, "by_detector": 0, "by_segmenter": 0, "by_human": 0},
    "unlabeled": {"pending_detection": 0, "no_label_source": 0},
    "in_progress": {"region_drain_total_unfinished": 0},
    "clusters": {"last_run_at": None, "cluster_count": 0, "residual_count": 0, "noise_count": 0, "method": None},
}


def register_dashboard(stub: Any, strategies: list[dict[str, Any]], refuse_start: bool = False) -> list[str]:
    starts: list[str] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/methods(\?|$)", {"strategies": strategies, "flags": {}})
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)
    stub.on("GET", r"/stats/dataset(\?|$)", DATASET_STATS)
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})
    snapshot = {"state": {}, "stats": DATASET_STATS}
    stub.on("GET", r"/pipeline/events", (200, f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n", "text/event-stream"))

    def start(request: Any, _m: Any):
        starts.append(request.url)
        return (200, IDLE_JOB)

    stub.on("POST", r"/pipeline/auto_label/start", start)
    return starts


def test_dashboard_auto_label_with_a_picked_external_endpoint(stub, page, app_url):
    starts = register_dashboard(stub, [*BASE_STRATEGIES, *VLM_STRATEGIES])

    page.goto(f"{app_url}/p/default/dashboard")
    chip = page.get_by_text(re.compile(r"^assist:\s*whole dataset"))
    chip.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    chip.first.click()

    picker = page.get_by_test_id("vlm-run-picker")
    expect(picker).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    options = picker.locator("option").all_inner_texts()
    assert options == ["Project default", "Local VLM · Ready", "Cloud VLM · Not probed yet", "Off"], options

    page.get_by_test_id("vlm-run-select").select_option("cloud_vlm")
    expect(page.get_by_test_id("vlm-run-ack")).to_contain_text(WARNING)
    page.get_by_test_id("vlm-run-ack-checkbox").check()

    page.get_by_role("button", name=re.compile(r"^Recluster")).first.click()
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/pipeline/auto_label/start" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_role("button", name="Start").first.click()
    assert len(starts) == 1, starts
    query = parse_qs(urlsplit(starts[0]).query)
    assert query["vlm"] == ["cloud_vlm"], query
    assert query["acknowledge_external"] == ["true"], query
    # A VLM pick implies the VLM stage, exactly as a class or pack does.
    assert query["run_vlm"] == ["true"], query


def test_settings_disables_an_unacknowledged_external_entry(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})
    stub.on(
        "GET",
        r"/methods(\?|$)",
        {
            "strategies": [*BASE_STRATEGIES, *VLM_STRATEGIES],
            "flags": {},
            "axes": [{"axis": "vlm", "label": "VLM endpoint", "description": "Which endpoint reads crops."}],
        },
    )
    page.goto(f"{app_url}/p/default/settings")
    select = page.locator("select").filter(has=page.locator('option[value="cloud_vlm"]'))
    select.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text("VLM endpoint").first).to_be_visible()
    expect(select.locator('option[value="cloud_vlm"]')).to_be_disabled()
    expect(select.locator('option[value="local_vlm"]')).to_be_enabled()
    expect(select.locator('option[value="off"]')).to_be_enabled()
    assert page.get_by_test_id("settings-ack-hint").count() == 1
    link = page.get_by_test_id("settings-ack-hint").get_by_role("link", name="Settings → Models")
    assert link.get_attribute("href").endswith("/settings/models")


def test_no_picker_without_a_vlm_axis(stub, page, app_url):
    starts = register_dashboard(stub, BASE_STRATEGIES)
    page.goto(f"{app_url}/p/default/dashboard")
    start_btn = page.get_by_role("button", name="Recluster now")
    start_btn.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("vlm-run-picker").count() == 0
    assert page.get_by_text(re.compile(r"^assist:")).count() == 0

    start_btn.first.click()
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/pipeline/auto_label/start" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_role("button", name="Start").first.click()
    query = parse_qs(urlsplit(starts[0]).query)
    assert "vlm" not in query and "acknowledge_external" not in query, query
