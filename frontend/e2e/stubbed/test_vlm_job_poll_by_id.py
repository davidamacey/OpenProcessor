"""M7 (docs/design/interactive-pass-2026-09-24.md §6, BOTH — now FIXED):
`pollAutoLabelJob`'s `expectedJobId` path must poll
`GET {API_PREFIX}/pipeline/auto_label/status/{job_id}` for the job the
`POST /vlm/label_cluster/{id}` response returned, not the old
client-side "skip any status whose job_id doesn't match" workaround over
the single "current job" `GET {API_PREFIX}/pipeline/auto_label/status`
slot.

Repro this proves is fixed: a stale, already-completed job sits in the
"current job" slot (simulating the exact live race the finding
describes — a previous run's terminal status still answering
`.../status` when a brand-new job is started) with DIFFERENT numbers
than the job actually just started. If the frontend ever fell back to
reading the stale slot instead of the per-job endpoint, the dashboard's
"VLM labeled N crops" toast would show the stale numbers.

`/dashboard`'s `runVlm` is the target here (simpler than `/clusters/[id]`'s
equivalent — no native `window.confirm` dialog to drive) — both call
sites share the same `pollAutoLabelJob` helper in `src/lib/api.ts`, so
this exercises the fix for both.
"""

from __future__ import annotations

import json
import re

CLASSES = [
    {"id": 1, "name": "ducati", "group": "moto", "hotkey_letter": "k", "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
]

METHODS = {"strategies": [], "flags": {}}

DATASET_STATS = {
    "as_of": "2026-09-24T00:00:00Z",
    "total_crops": 3,
    "validated": 0,
    "test_holdout": 0,
    "by_source": [],
    "labeled": {"by_human": 0, "by_vlm": 0, "by_classifier": 0, "by_proposal": 0, "other": 0},
    "regions": {"total_detected": 0, "by_detector": 0, "by_segmenter": 0, "by_human": 0},
    "unlabeled": {"pending_detection": 0, "no_label_source": 0},
    "in_progress": {"sam_drain_total_unfinished": 0},
    "clusters": {"last_run_at": None, "cluster_count": 0, "residual_count": 0, "noise_count": 0, "method": None},
}

# The stale "current job" slot — a DIFFERENT, already-terminal job with
# numbers that must never leak into the toast for the job started below.
STALE_CURRENT_JOB = {
    "job_id": "job-stale-previous-run",
    "status": "completed",
    "stage": "finalize",
    "processed": 99,
    "total": 99,
    "started_at": 0,
    "finished_at": 1,
    "error": None,
    "result": {"stages": {"vlm": {"predicted": 99, "updated": 99}}},
    "args": {},
    "eta_seconds": None,
    "elapsed_seconds": 0,
}


def make_real_job_status(status: str) -> dict:
    return {
        "job_id": "job-real-cluster-26",
        "status": status,
        "stage": "vlm",
        "processed": 1 if status == "running" else 7,
        "total": 7,
        "started_at": 0,
        "finished_at": 0 if status == "running" else 2,
        "error": None,
        "result": {} if status == "running" else {"stages": {"vlm": {"predicted": 7, "updated": 7}}},
        "args": {"cluster_id": 26},
        "eta_seconds": None,
        "elapsed_seconds": 0,
    }


def test_dashboard_run_vlm_polls_the_per_job_status_endpoint(stub, page, app_url):
    real_job_status_calls: list[str] = []
    bare_status_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/methods(\?|$)", METHODS)
    stub.on("GET", r"/stats/dataset(\?|$)", DATASET_STATS)
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})

    snapshot = {"state": {}, "stats": DATASET_STATS}
    sse_body = f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
    stub.on("GET", r"/pipeline/events", (200, sse_body, "text/event-stream"))

    # Registered FIRST so the more specific per-job pattern (registered
    # below) is tried before it — conftest.py's Stub tries handlers in
    # reverse registration order.
    def bare_status_handler(request, _match):
        bare_status_calls.append(request.url)
        return (200, STALE_CURRENT_JOB)

    stub.on("GET", r"/pipeline/auto_label/status$", bare_status_handler)

    def start_handler(_request, _match):
        return (200, make_real_job_status("running"))

    stub.on("POST", r"/vlm/label_cluster/26$", start_handler)

    # First poll: running. Second+: completed. Proves pollAutoLabelJob
    # actually hits THIS route (not the bare one) and keeps polling until
    # terminal.
    poll_count = {"n": 0}

    def job_status_handler(request, _match):
        real_job_status_calls.append(request.url)
        poll_count["n"] += 1
        status = "running" if poll_count["n"] == 1 else "completed"
        return (200, make_real_job_status(status))

    stub.on("GET", r"/pipeline/auto_label/status/job-real-cluster-26$", job_status_handler)

    page.goto(f"{app_url}/dashboard")
    open_btn = page.get_by_role("button", name="Run VLM Labeling")
    open_btn.first.wait_for(timeout=15000)
    page.wait_for_timeout(300)

    open_btn.first.click()
    page.get_by_placeholder("e.g. 42").fill("26")
    page.get_by_role("button", name="Run", exact=True).click()

    # The real job's toast (predicted 7 / updated 7) must appear — the
    # stale slot's numbers (99/99) must never appear.
    page.get_by_text("VLM labeled 7 crops (7 updated).", exact=False).first.wait_for(
        timeout=15000
    )
    page.wait_for_timeout(300)

    assert len(real_job_status_calls) >= 1, (
        "pollAutoLabelJob(expectedJobId) must call GET "
        ".../status/{job_id} for the job just started"
    )
    assert all("job-real-cluster-26" in u for u in real_job_status_calls), real_job_status_calls
    assert page.get_by_text("VLM labeled 99 crops", exact=False).count() == 0, (
        "the stale 'current job' slot's numbers must never leak into the toast"
    )
    assert page.get_by_text(re.compile(r"VLM labeled 0 crops \(0 updated\)")).count() == 0, (
        "the exact live-repro regression this finding described"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the run-VLM flow: {errors[:3]}"
