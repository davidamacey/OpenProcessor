"""Promote is a background job (OpenProcessor #87): POST /train/promote/{id}
answers 202 with a PromoteJobStatus; the modal follows the served phases
through GET .../jobs/{promote_id}, shows the served failure, tolerates a
409 promote_in_progress, and re-attaches after a reload via the `promote`
object on GET /train/status/{id}. The fake below walks one phase per poll.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

from test_train_gpus import register_train_mount

JOB_ID = "2026-09-24T23-47-55_yolo26n"
RUN = {
    "job_id": JOB_ID,
    "campaign_id": None,
    "state": "finished",
    "started_at": "2026-09-24T23:47:56Z",
    "finished_at": "2026-09-24T23:51:09Z",
    "current_epoch": 20,
    "total_epochs": 20,
    "eval": None,
}
RESULT = {
    "job_id": JOB_ID,
    "triton_name": "widgets",
    "onnx_path": "/m/model.onnx",
    "config_path": "/m/config.pbtxt",
    "labels_path": "/m/labels.txt",
    "triton_loaded": True,
    "cold_start_expected_on_first_inference": False,
}


def job(status, **extra):
    active = status not in ("done", "failed")
    return {
        "promote_id": "p1",
        "job_id": JOB_ID,
        "triton_name": "widgets",
        "status": status,
        "error": None,
        "error_status": None,
        "result": None,
        "started_at": "2026-10-04T10:00:00Z",
        "updated_at": "2026-10-04T10:00:01Z",
        "finished_at": None,
        "poll_after_s": 1 if active else None,
        **extra,
    }


class PhaseWalker:
    """Serves the next scripted phase per poll; the last one repeats."""

    def __init__(self, phases):
        self.phases = list(phases)
        self.polls = 0

    def __call__(self, _request, _match):
        i = min(self.polls, len(self.phases) - 1)
        self.polls += 1
        return (200, self.phases[i])


def open_modal(stub, page, app_url, status_body=None):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [RUN], "total": 1})
    stub.on(
        "GET",
        rf"/train/status/{JOB_ID}(\?|$)",
        status_body if status_body is not None else {**RUN, "promote": None},
    )
    page.goto(f"{app_url}/p/default/train")
    page.get_by_role("button", name="Promote ↑").first.click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_role("dialog", name="Promote to Triton").wait_for(timeout=ACTION_TIMEOUT_MS)


def current_phase(page):
    return page.locator('[data-testid="promote-progress"] [aria-current="step"]')


def test_promote_walks_phases_to_done(stub, page, app_url):
    walker = PhaseWalker(
        [job("exporting"), job("loading"), job("building"), job("warming"), job("done", result=RESULT)]
    )
    posts = []

    def start(request, _match):
        posts.append(request.url)
        return (202, job("queued"))

    stub.on("POST", rf"/train/promote/{JOB_ID}(\?|$)", start)
    stub.on("GET", rf"/train/promote/{JOB_ID}/jobs/p1(\?|$)", walker)
    open_modal(stub, page, app_url)

    page.get_by_role("button", name="Promote", exact=True).click()
    current_phase(page).wait_for(timeout=ACTION_TIMEOUT_MS)
    seen = []
    for expected in ("queued", "exporting", "loading", "building", "warming"):
        page.locator(f'[data-testid="promote-phase-{expected}"][aria-current="step"]').wait_for(
            timeout=ACTION_TIMEOUT_MS
        )
        seen.append(expected)
    assert seen == ["queued", "exporting", "loading", "building", "warming"]
    page.get_by_role("dialog", name="Promote to Triton").wait_for(state="hidden", timeout=ACTION_TIMEOUT_MS)
    assert len(posts) == 1 and "wait=true" not in posts[0]
    assert walker.polls == 5


def test_promote_failed_shows_served_error(stub, page, app_url):
    stub.on("POST", rf"/train/promote/{JOB_ID}(\?|$)", (202, job("queued")))
    stub.on(
        "GET",
        rf"/train/promote/{JOB_ID}/jobs/p1(\?|$)",
        PhaseWalker([job("loading"), job("failed", error="Triton refused the load", error_status=502)]),
    )
    open_modal(stub, page, app_url)
    page.get_by_role("button", name="Promote", exact=True).click()
    failed = page.locator('[data-testid="promote-failed"]')
    failed.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert "Triton refused the load (502)" in failed.inner_text()
    page.get_by_role("button", name="Edit and retry").click()
    page.get_by_role("button", name="Promote", exact=True).wait_for(timeout=ACTION_TIMEOUT_MS)


def test_promote_409_in_progress_attaches_to_active_job(stub, page, app_url):
    stub.on(
        "POST",
        rf"/train/promote/{JOB_ID}(\?|$)",
        (409, {"detail": {"code": "promote_in_progress", "message": "already running", "promote_id": "p1"}}),
    )
    stub.on(
        "GET",
        rf"/train/promote/{JOB_ID}/jobs/p1(\?|$)",
        PhaseWalker([job("building"), job("done", result=RESULT)]),
    )
    open_modal(stub, page, app_url)
    page.get_by_role("button", name="Promote", exact=True).click()
    page.locator('[data-testid="promote-phase-building"][aria-current="step"]').wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    assert page.locator('[data-testid="promote-failed"]').count() == 0
    page.get_by_role("dialog", name="Promote to Triton").wait_for(state="hidden", timeout=ACTION_TIMEOUT_MS)


def test_reload_resumes_an_active_promote(stub, page, app_url):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [RUN], "total": 1})
    stub.on("GET", rf"/train/status/{JOB_ID}(\?|$)", {**RUN, "promote": job("loading")})
    stub.on(
        "GET",
        rf"/train/promote/{JOB_ID}/jobs/p1(\?|$)",
        PhaseWalker([job("building"), job("done", result=RESULT)]),
    )
    page.goto(f"{app_url}/p/default/train")
    # No click: the page re-opens the modal on the running promote.
    page.get_by_role("dialog", name="Promote to Triton").wait_for(timeout=ACTION_TIMEOUT_MS)
    page.locator('[data-testid="promote-phase-loading"][aria-current="step"]').wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    page.locator('[data-testid="promote-phase-building"][aria-current="step"]').wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    page.get_by_role("dialog", name="Promote to Triton").wait_for(state="hidden", timeout=ACTION_TIMEOUT_MS)
    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]
