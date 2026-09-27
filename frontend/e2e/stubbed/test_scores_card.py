"""`/settings` "Curation scores" card (docs/design/
frontend-coverage-audit-2026-09-24.md §G10): the compute flow end to end
against the stub — coverage render, the confirm-before-compute dialog,
the request body sourced from served scorer ids, polling `/scores/status`
to completion with a coverage reload, cancel, and the card's absence on a
pre-`/scores/*` (404) backend.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import re

from playwright.sync_api import expect

COVERAGE_BEFORE = {
    "coverage": {
        "uniqueness": {"field": "uniqueness_score", "n_scored": 0, "total": 7961, "pct": 0.0},
        "near_dup": {"field": "dup_group_id", "n_scored": 0, "total": 7961, "pct": 0.0},
    }
}

COVERAGE_AFTER = {
    "coverage": {
        "uniqueness": {
            "field": "uniqueness_score",
            "n_scored": 7961,
            "total": 7961,
            "pct": 100.0,
        },
        "near_dup": {"field": "dup_group_id", "n_scored": 0, "total": 7961, "pct": 0.0},
    }
}


def register(stub, *, coverage_status=200, coverage_body=None):
    compute_calls: list[tuple[str, str]] = []
    status_calls = {"n": 0}
    coverage_state = {"body": coverage_body if coverage_body is not None else COVERAGE_BEFORE}

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None})

    def coverage_get(_request, _match):
        if coverage_status != 200:
            return (coverage_status, {"detail": "not found"})
        return (200, coverage_state["body"])

    def compute_post(request, _match):
        compute_calls.append((request.url, request.post_data or ""))
        status_calls["n"] = 0
        return (
            200,
            {
                "job_id": "job-1",
                "status": "running",
                "scorers": ["uniqueness", "near_dup"],
                "processed": 0,
                "total": 7961,
                "started_at": 1000,
                "finished_at": 0,
                "error": None,
                "results": {},
            },
        )

    def status_get(_request, _match):
        status_calls["n"] += 1
        # First poll (and the mount-time adoption check) reports running;
        # the second poll reports completed — this exercises a real
        # poll-until-terminal loop, not a same-tick resolve.
        if status_calls["n"] < 2:
            return (
                200,
                {
                    "job_id": "job-1",
                    "status": "running",
                    "scorers": ["uniqueness", "near_dup"],
                    "processed": 3000,
                    "total": 7961,
                    "started_at": 1000,
                    "finished_at": 0,
                    "error": None,
                    "results": {},
                },
            )
        coverage_state["body"] = COVERAGE_AFTER
        return (
            200,
            {
                "job_id": "job-1",
                "status": "completed",
                "scorers": ["uniqueness", "near_dup"],
                "processed": 7961,
                "total": 7961,
                "started_at": 1000,
                "finished_at": 1050,
                "error": None,
                "results": {"uniqueness": {"n_scored": 7961}, "near_dup": {"n_scored": 0}},
            },
        )

    def cancel_post(_request, _match):
        return (
            200,
            {
                "cancelled": True,
                "job_id": "job-1",
                "status": "cancelled",
                "scorers": ["uniqueness"],
                "processed": 10,
                "total": 7961,
                "started_at": 1000,
                "finished_at": 1005,
                "error": None,
                "results": {},
            },
        )

    stub.on("GET", r"/scores/coverage(\?|$)", coverage_get)
    stub.on("POST", r"/scores/compute(\?|$)", compute_post)
    stub.on("GET", r"/scores/status(\?|$)", status_get)
    stub.on("POST", r"/scores/cancel(\?|$)", cancel_post)
    return compute_calls


def test_scores_card_absent_on_404(stub, page, app_url):
    register(stub, coverage_status=404)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Deployment defaults").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")]
    assert page.get_by_text("Curation scores").count() == 0, (
        "the scores card must be entirely absent on a pre-/scores/* backend, not "
        "rendered broken/empty"
    )


def test_scores_card_compute_all_flow(stub, page, app_url):
    compute_calls = register(stub)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Curation scores").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")]
    assert page.get_by_text("uniqueness").count() > 0
    assert page.get_by_text("near_dup").count() > 0

    page.get_by_role("button", name="Compute all").click()

    dialog = page.get_by_role("dialog", name="Confirm compute curation scores")
    expect(dialog).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "POST" and r.url.endswith("/scores/compute"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        dialog.get_by_role("button", name="Confirm").click()

    assert len(compute_calls) == 1, f"exactly one compute POST expected: {compute_calls}"
    _url, body = compute_calls[0]
    assert body == '{"scorers":null}', body

    page.get_by_text(re.compile(r"Computing")).first.wait_for(timeout=5000)

    # Poll interval is 3s; two live polls are needed before the job
    # resolves to 'completed'. `expect(...)` retries on its own schedule
    # until the real text shows up (or the shared action-timeout budget
    # is exhausted), instead of guessing a fixed multiple of the interval.
    expect(page.get_by_text(re.compile(r"7,961\s*/\s*7,961")).first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    # the in-progress indicator should clear once the job is done
    expect(page.get_by_text(re.compile(r"Computing"))).to_have_count(0, timeout=ACTION_TIMEOUT_MS)


def test_scores_card_compute_selected_and_cancel(stub, page, app_url):
    compute_calls = register(stub)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Curation scores").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    page.get_by_label("Select near_dup").check()
    compute_selected_btn = page.get_by_role("button", name=re.compile(r"^Compute selected"))
    expect(compute_selected_btn).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    compute_selected_btn.click()

    dialog = page.get_by_role("dialog", name="Confirm compute curation scores")
    expect(dialog).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert dialog.get_by_text("near_dup").count() > 0
    with page.expect_response(
        lambda r: r.request.method == "POST" and r.url.endswith("/scores/compute"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        dialog.get_by_role("button", name="Confirm").click()

    assert len(compute_calls) == 1
    _url, body = compute_calls[0]
    assert body == '{"scorers":["near_dup"]}', body

    cancel_btn = page.get_by_role("button", name="Cancel")
    cancel_btn.first.wait_for(timeout=5000)
    with page.expect_response(
        lambda r: r.request.method == "POST" and r.url.endswith("/scores/cancel"),
        timeout=ACTION_TIMEOUT_MS,
    ):
        cancel_btn.click()

    expect(page.get_by_role("button", name="Cancel")).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
