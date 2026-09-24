"""New coverage (docs/design/test-audit-2026-09-24.md recommendation 5):
`/dashboard`'s stats come over SSE (`GET {API_PREFIX}/pipeline/events`,
`subscribePipelineEvents`, src/lib/sse.ts). A `snapshot` frame carrying
`{stats: {error: "..."}}` (the live-backend G1 failure mode) must render
the literal "Stats unavailable: {error}" text (DatasetStats.svelte)
without ever throwing a page error — resolveStatsUpdate
(src/lib/datasetStats.ts) guards it.
"""

from __future__ import annotations

import json

CLASSES = [
    {"id": 1, "name": "ducati", "group": "moto", "hotkey_letter": "k", "count": 10, "validated_count": 5, "cluster_size": 12, "deprecated": False},
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


def test_dashboard_stats_unavailable(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/methods(\?|$)", {"strategies": [], "flags": {}})
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)
    # A degraded SSE connection makes DatasetStats fall back to polling
    # these directly — stub them even though this test's assertion is
    # about the SSE-carried error, not these calls.
    stub.on("GET", r"/stats/dataset(\?|$)", {"total_crops": 3, "validated": 0, "test_holdout": 0, "by_source": {}})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 30, "crops": []})

    snapshot = {"state": {}, "stats": {"error": "stats unavailable (503)"}}
    sse_body = f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
    stub.on("GET", r"/pipeline/events", (200, sse_body, "text/event-stream"))

    page.goto(f"{app_url}/dashboard")
    page.get_by_text("Stats unavailable:", exact=False).first.wait_for(timeout=15000)
    page.wait_for_timeout(300)

    assert page.get_by_text("Stats unavailable: stats unavailable (503)", exact=False).count() > 0, (
        "the exact server error text should render inline"
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"a stats 503 must not throw a page error: {errors[:3]}"
