"""OpenProcessor #36 item 8 — the /train "Run probe predictions" control
(ProbeControl.svelte, embedded in RunResults for a finished run).
Populates probe_pred_* on items, the one prerequisite item 9's
empty_state.has_probe_predictions checks for.
"""

from __future__ import annotations

from test_train_gpus import register_train_mount

JOB_ID = "2026-09-24T23-47-55_yolo26n"

FINISHED_STATUS = {
    "job_id": JOB_ID,
    "campaign_id": None,
    "state": "finished",
    "started_at": "2026-09-24T23:47:56Z",
    "finished_at": "2026-09-24T23:51:09Z",
    "current_epoch": 20,
    "total_epochs": 20,
    # Live regression (2026-09-25): GET {API_PREFIX}/train/status/{job_id}
    # serves checkpoint_path on a finished run but NOT checkpoint_sha256
    # (that only appears inside the manifest) — the control gates on
    # checkpoint_path, matching what this endpoint actually serves.
    "checkpoint_path": "/var/lib/openprocessor/training_runs/{}/weights/best.pt".format(
        JOB_ID
    ),
}

NO_CHECKPOINT_STATUS = {
    **FINISHED_STATUS,
    "job_id": "run-no-checkpoint",
    "checkpoint_path": None,
}


def _open_results(page, app_url, job_id: str):
    page.goto(f"{app_url}/train")
    page.get_by_text(job_id, exact=False).first.wait_for(timeout=15000)
    results_button = page.get_by_role("button", name="Results")
    results_button.wait_for(timeout=10000)
    results_button.click()


def test_run_probe_button_offered_for_a_finished_run_with_a_checkpoint(
    stub, page, app_url
):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [FINISHED_STATUS], "total": 1})
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", {})
    stub.on("GET", r"/probe/status(\?|$)", {"status": "idle"})

    _open_results(page, app_url, JOB_ID)

    control = page.get_by_test_id("probe-control")
    control.wait_for(timeout=10000)
    assert "Run probe predictions" in control.inner_text()

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_run_probe_button_absent_for_a_run_with_no_checkpoint(stub, page, app_url):
    register_train_mount(stub)
    stub.on(
        "GET", r"/train/runs(\?|$)", {"items": [NO_CHECKPOINT_STATUS], "total": 1}
    )
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", {})
    stub.on("GET", r"/probe/status(\?|$)", {"status": "idle"})

    _open_results(page, app_url, "run-no-checkpoint")

    page.get_by_text("Loading manifest", exact=False).wait_for(
        state="hidden", timeout=10000
    )
    assert page.get_by_test_id("probe-control").count() == 0


def test_confirm_then_run_probe_posts_this_runs_job_id_and_polls_status(
    stub, page, app_url
):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [FINISHED_STATUS], "total": 1})
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", {})

    posted = {}
    poll_count = {"n": 0}

    def run_handler(request, _match):
        posted["body"] = request.post_data_json
        return (200, {"status": "running", "job_id": "probe-1", "train_job_id": JOB_ID})

    def status_handler(request, _match):
        poll_count["n"] += 1
        if poll_count["n"] == 1:
            return (200, {"status": "idle"})
        return (
            200,
            {
                "status": "completed",
                "job_id": "probe-1",
                "train_job_id": JOB_ID,
                "updated_count": 42,
            },
        )

    stub.on("POST", r"/probe/run(\?|$)", run_handler)
    stub.on("GET", r"/probe/status(\?|$)", status_handler)

    _open_results(page, app_url, JOB_ID)

    control = page.get_by_test_id("probe-control")
    control.wait_for(timeout=10000)
    control.get_by_role("button", name="Run probe predictions").click()

    dialog = page.get_by_role("dialog", name="Confirm run probe predictions")
    dialog.wait_for(timeout=5000)
    dialog.get_by_role("button", name="Confirm").click()

    page.wait_for_function(
        "document.querySelector('[data-testid=\"probe-control\"]')?.textContent.includes('Running')",
        timeout=10000,
    )
    assert posted["body"] == {"job_id": JOB_ID}

    page.wait_for_function(
        "document.querySelector('[data-testid=\"probe-control\"]')?.textContent.includes('42 items updated')",
        timeout=15000,
    )

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_probe_run_409_shows_the_served_detail_verbatim(stub, page, app_url):
    register_train_mount(stub)
    stub.on("GET", r"/train/runs(\?|$)", {"items": [FINISHED_STATUS], "total": 1})
    stub.on("GET", r"/train/manifest/[^/]+(\?|$)", {})
    stub.on("GET", r"/probe/status(\?|$)", {"status": "idle"})
    stub.on(
        "POST",
        r"/probe/run(\?|$)",
        (409, {"detail": "a probe is already running"}),
    )

    _open_results(page, app_url, JOB_ID)

    control = page.get_by_test_id("probe-control")
    control.wait_for(timeout=10000)
    control.get_by_role("button", name="Run probe predictions").click()
    page.get_by_role("dialog", name="Confirm run probe predictions").get_by_role(
        "button", name="Confirm"
    ).click()

    page.wait_for_function(
        "document.querySelector('[data-testid=\"probe-control\"]')?.textContent.includes('a probe is already running')",
        timeout=10000,
    )
