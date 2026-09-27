"""Ported from scripts/playwright_assist_scope.py (deleted; see
docs/design/test-audit-2026-09-24.md recommendation 5 / P1-1).

Rewritten against current `AssistScopeBar.svelte`: the original script
assumed a detection-profile `<select>` alongside the prompt-pack one.
That control was deliberately removed (CLAUDE.md: "There is deliberately
no detection-profile control: region detection is the backend's startup
config") — `AssistScopeBar` renders only a class-search list and a single
optional prompt-pack `<select>`, gated by `isScopedAssistAvailable`
(prompt_pack axis only).

Pass 1 — `/curation/methods` answers with today's real shape (no
prompt_pack axis). The assist bar must be entirely absent, and the
unscoped start request must carry none of class_id/prompt_pack.

Pass 2 — `/curation/methods` advertises the prompt_pack axis (plus one
disabled pack, to prove the status filter). The bar appears, the class
picker narrows to "pallets", a non-default prompt pack can be picked, and
the recorded start request carries class_id + prompt_pack plus the
pre-existing params.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint

import json
import re
from urllib.parse import urlsplit

from playwright.sync_api import expect

CLASSES = [
    {
        "id": 1,
        "name": "pallets",
        "group": "warehouse",
        "hotkey_letter": "p",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "forklift",
        "group": "warehouse",
        "hotkey_letter": "f",
        "count": 20,
        "validated_count": 5,
        "cluster_size": 22,
        "deprecated": False,
    },
]

METHODS_TODAY = {
    "strategies": [
        {"id": "ivf", "axis": "cluster", "label": "FAISS IVF-512 (production)", "status": "stable", "default": True},
        {"id": "default", "axis": "sort", "label": "Recent first", "status": "stable", "default": True},
    ],
    "flags": {},
}

METHODS_WITH_PROMPT_PACK = {
    "strategies": [
        *METHODS_TODAY["strategies"],
        {"id": "warehouse_v1", "axis": "prompt_pack", "label": "Warehouse vocabulary", "status": "stable", "default": True},
        {"id": "retired_pack", "axis": "prompt_pack", "label": "Retired prompt pack", "status": "disabled"},
    ],
    "flags": {},
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

# Full DatasetStats shape (src/lib/api.ts) — DatasetStats.svelte reads
# nested fields (e.g. in_progress.region_drain_total_unfinished) directly, so
# a partial fixture throws a real pageerror instead of degrading via
# resolveStatsUpdate's error-envelope guard (that guard only checks
# top-level total_crops).
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


def register_base(stub, methods_body):
    starts: list[str] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    # D2 (visual audit 2026-09-24): the dashboard balance chart reads the
    # served per-class test holdout.
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/methods(\?|$)", methods_body)
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)
    stub.on("GET", r"/stats/dataset(\?|$)", DATASET_STATS)
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})

    snapshot = {"state": {}, "stats": DATASET_STATS}
    sse_body = f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
    stub.on("GET", r"/pipeline/events", (200, sse_body, "text/event-stream"))

    def start_handler(request, _match):
        starts.append(request.url)
        return (200, IDLE_JOB)

    stub.on("POST", r"/pipeline/auto_label/start", start_handler)
    return starts


def query_of(url: str) -> str:
    return urlsplit(url).query


def test_assist_scope(stub, page, app_url):
    # ================================================================
    # Pass 1 — today's real backend: no prompt_pack axis advertised.
    # ================================================================
    starts1 = register_base(stub, METHODS_TODAY)

    page.goto(f"{app_url}/dashboard")
    start_btn = page.get_by_role("button", name="Recluster now")
    start_btn.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    # Real wait for the dashboard's on-mount requests to finish before the
    # negative "no assist chip" assertion below.
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")], "dashboard should render with no pageerror"
    assert page.get_by_text(re.compile(r"^assist:")).count() == 0, "no 'assist:' chip should render — absent, not disabled"
    assert start_btn.count() > 0, "primary button should read exactly 'Recluster now'"

    starts1.clear()
    start_btn.first.click()
    # p2 (2026-09-24 interactive pass): "Recluster now" opens an in-page
    # confirm dialog before it actually starts the run.
    confirm_btn = page.get_by_role("button", name="Start")
    confirm_btn.first.wait_for(timeout=5000)
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/pipeline/auto_label/start" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        confirm_btn.first.click()
    assert len(starts1) == 1, f"exactly one POST to auto_label/start expected: {starts1}"
    qs = query_of(starts1[0])
    assert "class_id" not in qs, qs
    assert "prompt_pack" not in qs, qs

    # ================================================================
    # Pass 2 — backend advertises the prompt_pack axis.
    # ================================================================
    starts2 = register_base(stub, METHODS_WITH_PROMPT_PACK)

    stub.console_errors.clear()
    page.goto(f"{app_url}/dashboard")
    chip = page.get_by_text(re.compile(r"^assist:\s*whole dataset"))
    chip.first.wait_for(timeout=ACTION_TIMEOUT_MS)

    chip.first.click()
    class_search = page.locator("input[type=search]")
    expect(class_search.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("Server default").count() >= 1, "the prompt-pack <select> should appear"

    options_text = page.locator("select option").all_inner_texts()
    assert "Retired prompt pack" not in options_text, f"the disabled pack must not be offered: {options_text}"
    assert any("Warehouse vocabulary" in t for t in options_text), f"the usable pack should be offered: {options_text}"

    class_search.first.fill("pall")
    pallets_row = page.get_by_text("pallets", exact=False)
    expect(pallets_row.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    pallets_row.first.click()
    # Clicking a class-search result closes the list synchronously — no
    # network involved, just let the DOM update settle via a real paint.
    wait_for_paint(page)

    selects = page.locator("select")
    assert selects.count() == 1, f"exactly one <select> expected (prompt pack only, no detection profile): {selects.count()}"
    selects.first.select_option("warehouse_v1")

    collapse_btn = page.get_by_role("button", name="×")
    if collapse_btn.count() > 0:
        collapse_btn.first.click()

    scoped_btn = page.get_by_role("button", name=re.compile(r"^Recluster · VLM: pallets$"))
    expect(scoped_btn.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text(re.compile(r"^assist:\s*pallets")).first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )

    starts2.clear()
    scoped_btn.first.click()
    confirm_btn2 = page.get_by_role("button", name="Start")
    confirm_btn2.first.wait_for(timeout=5000)
    with page.expect_response(
        lambda r: r.request.method == "POST" and "/pipeline/auto_label/start" in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        confirm_btn2.first.click()
    assert len(starts2) == 1, f"exactly one scoped POST to auto_label/start expected: {starts2}"
    qs = query_of(starts2[0])
    assert "class_id=1" in qs, qs
    assert "prompt_pack=warehouse_v1" in qs, qs
    assert all(
        p in qs for p in ["train_clusters", "vlm_concurrency", "max_vlm_crops", "recluster_unvalidated"]
    ), qs

    # Re-expand, reset — chip and button return to the unscoped defaults.
    page.goto(f"{app_url}/dashboard")
    chip2 = page.get_by_text(re.compile(r"^assist:"))
    chip2.first.wait_for(timeout=ACTION_TIMEOUT_MS)
    chip2.first.click()
    # Expanding is a synchronous DOM toggle; the reset button is only
    # conditionally present (state may already be back to default after
    # the reload above), so there's no selector to wait FOR here — just
    # let the toggle's own DOM update settle via a real paint tick.
    wait_for_paint(page)
    reset_btn = page.get_by_role("button", name="reset")
    if reset_btn.count() > 0:
        reset_btn.first.click()
    collapse_btn2 = page.get_by_role("button", name="×")
    if collapse_btn2.count() > 0:
        collapse_btn2.first.click()
    expect(page.get_by_text(re.compile(r"^assist:\s*whole dataset")).first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    ), "reset should return the chip to 'assist: whole dataset'"

    pass2_errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not pass2_errors, f"no pageerror expected across pass 2: {pass2_errors[:3]}"
