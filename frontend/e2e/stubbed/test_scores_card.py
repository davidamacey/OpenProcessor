"""`/settings` "Curation scores" card (docs/design/
frontend-coverage-audit-2026-09-24.md §G10): the compute flow end to end
against the stub — coverage render, the confirm-before-compute dialog,
the request body sourced from served scorer ids, polling `/scores/status`
to completion with a coverage reload, and cancel.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import re

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


def register(stub, *, coverage_body=None):
    compute_calls: list[tuple[str, str]] = []
    status_calls = {"n": 0}
    coverage_state = {"body": coverage_body if coverage_body is not None else COVERAGE_BEFORE}

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/settings(\?|$)", {"defaults": {}, "updated_at": None, "updated_by": None})

    def coverage_get(_request, _match):
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


def test_scores_card_compute_all_flow(stub, page, app_url):
    compute_calls = register(stub)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Curation scores").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")]
    assert page.get_by_text("uniqueness").count() > 0
    assert page.get_by_text("near_dup").count() > 0

    page.get_by_role("button", name="Compute all").click()
    page.wait_for_timeout(200)

    dialog = page.get_by_role("dialog", name="Confirm compute curation scores")
    assert dialog.count() > 0, "confirm dialog should appear before any write"
    dialog.get_by_role("button", name="Confirm").click()
    page.wait_for_timeout(300)

    assert len(compute_calls) == 1, f"exactly one compute POST expected: {compute_calls}"
    _url, body = compute_calls[0]
    assert body == '{"scorers":null}', body

    page.get_by_text(re.compile(r"Computing")).first.wait_for(timeout=5000)

    # Poll interval is 3s; wait past two ticks so the job resolves to
    # 'completed' and the coverage reload lands.
    page.wait_for_timeout(6500)

    assert page.get_by_text(re.compile(r"7,961\s*/\s*7,961")).count() > 0, (
        "coverage should reload with the post-compute numbers once the job completes"
    )
    assert page.get_by_text(re.compile(r"Computing")).count() == 0, (
        "the in-progress indicator should clear once the job is done"
    )


def test_scores_card_compute_selected_and_cancel(stub, page, app_url):
    compute_calls = register(stub)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Curation scores").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)

    page.get_by_label("Select near_dup").check()
    page.wait_for_timeout(150)
    page.get_by_role("button", name=re.compile(r"^Compute selected")).click()
    page.wait_for_timeout(200)

    dialog = page.get_by_role("dialog", name="Confirm compute curation scores")
    assert dialog.get_by_text("near_dup").count() > 0
    dialog.get_by_role("button", name="Confirm").click()
    page.wait_for_timeout(300)

    assert len(compute_calls) == 1
    _url, body = compute_calls[0]
    assert body == '{"scorers":["near_dup"]}', body

    page.get_by_role("button", name="Cancel").first.wait_for(timeout=5000)
    page.get_by_role("button", name="Cancel").click()
    page.wait_for_timeout(300)

    assert page.get_by_role("button", name="Cancel").count() == 0, (
        "cancel should clear the in-progress state (no more Cancel button) without "
        "waiting for the next poll"
    )
