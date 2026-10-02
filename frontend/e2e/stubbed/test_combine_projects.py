"""OpenProcessor P4 (combine projects), projects_plan.md §6, §8;
docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md §4.

The wire follows the vendored contract (f582aa05) and `plan.py` /
`execute.py`. The fail-closed stub means any combine request the page makes
that a test did not expect fails the test.

  1. `test_two_sources_preview_mapping_and_start` — pick two sources, the
     served suggestions fill the mapping, a touched row survives, Start
     (behind a confirm) sends the exact body incl. `expected_preview_sha`.
  2. `test_job_completes_and_offers_next_steps` — the job view polls to
     completed; the Open-project / conflict-review hrefs; the served next
     step runs against the target's prefix.
  3. `test_preview_stale_re_previews` — a 409 `preview_stale` shows the
     served message and fires a fresh preview; the retry carries the new sha.
  4. `test_failed_job_undo_opens_the_delete_dry_run`.
  5. `test_feature_absent_on_a_plain_404` — conftest's default 404: no
     button, the route says so, no other combine request fires.
"""

from __future__ import annotations

import copy
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from fixtures.wire import project, projects_response

API_PREFIX = "/curation"
JOB_ID = "cmb_20261001T120000_1a2b3c4d"

NOT_FOUND = {"detail": {"error": "combine_not_found", "message": "no combine job"}}

MAPPING_ACTIONS = [
    {"value": "map", "label": "Map to class", "description": "Point at a class a create row defines."},
    {"value": "create", "label": "Create class", "description": "Define a target class."},
    {"value": "skip", "label": "Skip this class", "description": ""},
    {"value": "region", "label": "Region boxes", "description": ""},
]


def source(slug: str, classes: list[tuple[str, int, str | None]]) -> dict[str, Any]:
    return {
        "project": slug,
        "images": 10,
        "items": sum(c[1] for c in classes),
        "labeled_items": 4,
        "holdout_images": 1,
        "classes": [{"name": n, "count": c, "mapped_to": m} for n, c, m in classes],
    }


def preview(sha: str = "sha-1", **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [
            {
                "code": "label_conflicts",
                "severity": "warning",
                "project": None,
                "message": "1 boxes disagree between sources; the first source wins",
                "detail": {"count": 1},
            }
        ],
        "preview_sha": sha,
        "suggested_mapping": {
            "widgets-a": [
                {"dataset_class": "widget", "action": "create", "new_class_name": "widget"},
                {"dataset_class": "gadget", "action": "create", "new_class_name": "gadget"},
            ],
            "widgets-b": [{"dataset_class": "widget", "action": "map", "new_class_name": "widget"}],
        },
        "sources": [
            source("widgets-a", [("widget", 8, "widget"), ("gadget", 2, "gadget")]),
            source("widgets-b", [("widget", 5, "widget")]),
        ],
        "target": {
            "slug": "merged",
            "slug_available": True,
            "classes": [
                {
                    "id": 0,
                    "name": "widget",
                    "count": 13,
                    "from": [
                        {"project": "widgets-a", "class": "widget"},
                        {"project": "widgets-b", "class": "widget"},
                    ],
                },
                {"id": 1, "name": "gadget", "count": 2, "from": [{"project": "widgets-a", "class": "gadget"}]},
            ],
            "images": 18,
            "items": 15,
            "holdout_images": 1,
        },
        "dedup": {
            "identical_images": 2,
            "merged_items": 1,
            "conflicts": 1,
            "conflict_samples": [],
            "near_duplicate_pairs_estimate": None,
        },
        "bytes": {"to_link": 2048, "to_copy": 0},
    }
    base.update(over)
    return base


def job(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "job_id": JOB_ID,
        "status": "running",
        "phase": "images",
        "done": 8,
        "total": 20,
        "started_at": "2026-10-01T12:00:00Z",
        "finished_at": None,
        "sources": ["widgets-a", "widgets-b"],
        "target": "merged",
        "error": None,
        "report": {},
        "next_steps": [],
    }
    base.update(over)
    return base


def serve_projects(stub: Any, *, merged: dict[str, Any] | None = None) -> None:
    rows = [
        project(API_PREFIX, "default", is_default=True, deletable=False),
        project(API_PREFIX, "widgets-a", display_name="Widgets A"),
        project(API_PREFIX, "widgets-b", display_name="Widgets B"),
    ]
    if merged is not None:
        rows.append(merged)
    stub.on("GET", rf"^{re.escape(API_PREFIX)}/projects$", (200, projects_response(API_PREFIX, rows)))


def serve_combine(stub: Any, *, jobs: list[dict[str, Any]] | None = None) -> list[int]:
    """The combine router: the probe's structured 404, and the job reads."""
    reads = [0]

    def handler(request: Any, m: Any):
        job_id = m.group(1)
        if job_id != JOB_ID or not jobs:
            return (404, NOT_FOUND)
        i = min(reads[0], len(jobs) - 1)
        reads[0] += 1
        return (200, jobs[i])

    stub.on("GET", r"/projects/combine/([^/?]+)$", handler)
    return reads


def serve_source_formats(stub: Any) -> None:
    stub.on("GET", r"/projects/widgets-a/datasets/formats$", {"mapping_actions": copy.deepcopy(MAPPING_ACTIONS)})


def open_wizard(page: Any, app_url: str) -> None:
    page.goto(f"{app_url}/projects/combine")
    page.get_by_test_id("combine-add-source").wait_for(timeout=ACTION_TIMEOUT_MS)


def fill_wizard(page: Any) -> None:
    page.get_by_test_id("combine-add-source").select_option("widgets-a")
    page.get_by_test_id("combine-add-source").select_option("widgets-b")
    page.get_by_test_id("combine-target-slug").fill("merged")
    page.get_by_test_id("combine-target-name").fill("Merged")


def test_two_sources_preview_mapping_and_start(stub, page, app_url):
    serve_projects(stub)
    serve_combine(stub)
    serve_source_formats(stub)
    previews: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []

    def preview_handler(request: Any, _m: Any):
        previews.append(request.post_data_json)
        return (200, preview())

    def start_handler(request: Any, _m: Any):
        starts.append(request.post_data_json)
        return (202, {"job_id": JOB_ID, "target": "merged"})

    stub.on("POST", r"/projects/combine/preview$", preview_handler)
    stub.on("POST", r"/projects/combine$", start_handler)

    open_wizard(page, app_url)
    # Start is not offered before there is a preview.
    expect(page.get_by_test_id("combine-start")).to_be_disabled()
    fill_wizard(page)

    # The served preview renders: counts, warning, target classes.
    page.get_by_test_id("combine-preview").wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-warnings")).to_contain_text("1 boxes disagree")
    expect(page.get_by_test_id("combine-preview-target")).to_contain_text("18 images")
    # Served mapping-action labels (read from the first source's formats).
    expect(page.get_by_test_id("combine-mapping-widgets-a").locator("option", has_text="Skip this class").first).to_be_attached(
        timeout=ACTION_TIMEOUT_MS
    )

    # The suggestions fill the untouched rows.
    row_b = page.get_by_test_id("combine-mapping-widgets-b").locator('[data-class="widget"]')
    expect(row_b.get_by_test_id("combine-map-action")).to_have_value("map", timeout=ACTION_TIMEOUT_MS)
    expect(row_b.get_by_test_id("combine-map-target")).to_have_value("widget")

    # A touched row survives the re-previews that follow.
    row_a_gadget = page.get_by_test_id("combine-mapping-widgets-a").locator('[data-class="gadget"]')
    row_a_gadget.get_by_test_id("combine-map-action").select_option("skip")

    with page.expect_request(
        lambda r: r.method == "POST" and r.url.endswith("/projects/combine")
    ) as req_info:
        page.get_by_test_id("combine-start").click()
        expect(page.get_by_test_id("combine-confirm-body")).to_contain_text("18 images")
        page.get_by_role("button", name="Start combine", exact=True).click()
    assert req_info.value.post_data_json["expected_preview_sha"] == "sha-1"

    page.wait_for_url(re.compile(rf"/projects/combine/{JOB_ID}$"), timeout=ACTION_TIMEOUT_MS)
    assert len(starts) == 1
    body = starts[0]
    assert body["target"] == {"slug": "merged", "display_name": "Merged"}
    assert body["sources"] == [{"project": "widgets-a"}, {"project": "widgets-b"}]
    assert body["class_mapping"] == {
        "widgets-a": [
            {"dataset_class": "widget", "action": "create", "new_class_name": "widget"},
            {"dataset_class": "gadget", "action": "skip"},
        ],
        "widgets-b": [{"dataset_class": "widget", "action": "map", "new_class_name": "widget"}],
    }
    # Nothing the operator never touched is sent.
    for key in ("dedup", "dedup_iou", "holdout", "settings_from", "target_classes"):
        assert key not in body
    # The very first preview had no mapping at all; the last carried the touched row.
    assert "class_mapping" not in previews[0]
    assert previews[-1]["class_mapping"]["widgets-a"][1] == {"dataset_class": "gadget", "action": "skip"}


def test_job_completes_and_offers_next_steps(stub, page, app_url):
    serve_projects(
        stub,
        merged=project(
            API_PREFIX,
            "merged",
            display_name="Merged",
            origin={"kind": "combine", "job_id": JOB_ID, "sources": ["widgets-a", "widgets-b"]},
        ),
    )
    completed = job(
        status="completed",
        phase="done",
        done=20,
        finished_at="2026-10-01T12:05:00Z",
        report={"images_copied": 18, "images_duplicate": 2, "items_copied": 15},
        next_steps=[
            {
                "action": "recluster",
                "method": "POST",
                "path": "/cluster/umap/rebuild",
                "reason": "Source clusters were not copied; recluster the combined items.",
            }
        ],
    )
    serve_combine(stub, jobs=[job(status="running"), completed])
    ran: list[tuple[str, str]] = []

    def next_step(request: Any, _m: Any):
        ran.append((request.method, request.post_data or ""))
        return (200, {"status": "no_residuals", "n_residuals": 0, "refit": False})

    stub.on("POST", r"/projects/merged/cluster/umap/rebuild$", next_step)

    page.goto(f"{app_url}/projects/combine/{JOB_ID}")
    expect(page.get_by_test_id("combine-job-status")).to_have_text("Running", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-job-progress")).to_contain_text("8 of 20")
    expect(page.get_by_test_id("combine-cancel")).to_be_visible()
    # The next poll (2 s) sees the completed job.
    expect(page.get_by_test_id("combine-job-status")).to_have_text("Completed", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-cancel")).to_have_count(0)
    assert page.get_by_test_id("combine-open-project").get_attribute("href") == "/p/merged/dashboard"
    assert (
        page.get_by_test_id("combine-review-conflicts").get_attribute("href")
        == "/p/merged/review?tab=all&combine_conflict=true"
    )
    expect(page.get_by_test_id("combine-job-report")).to_contain_text("15")

    page.get_by_test_id("combine-next-step-recluster").click()
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/projects/merged/cluster/umap/rebuild")):
        page.get_by_role("button", name="Run", exact=True).click()
    expect(page.get_by_role("button", name="Run", exact=True)).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert ran == [("POST", "")]
    result = page.get_by_test_id("combine-step-result")
    expect(result).to_contain_text("Recluster")
    expect(page.get_by_test_id("combine-step-result-status")).to_have_text("No residuals")
    expect(result.locator('[data-result-key="n_residuals"]')).to_have_text("0")
    expect(page.get_by_text("Ran Recluster")).to_be_visible()


def test_preview_stale_re_previews(stub, page, app_url):
    serve_projects(stub)
    serve_combine(stub)
    previews: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    state = {"sha": "sha-1"}

    def preview_handler(request: Any, _m: Any):
        previews.append(request.post_data_json)
        return (200, preview(state["sha"]))

    def start_handler(request: Any, _m: Any):
        body = request.post_data_json
        starts.append(body)
        if body["expected_preview_sha"] == "sha-1":
            state["sha"] = "sha-2"  # a source changed behind our back
            return (
                409,
                {"detail": {"error": "preview_stale", "message": "a source changed since the preview; preview again"}},
            )
        return (202, {"job_id": JOB_ID, "target": "merged"})

    stub.on("POST", r"/projects/combine/preview$", preview_handler)
    stub.on("POST", r"/projects/combine$", start_handler)

    open_wizard(page, app_url)
    fill_wizard(page)
    page.get_by_test_id("combine-preview").wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-start")).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    settled = len(previews)

    # The 409 triggers a fresh preview on its own.
    with page.expect_request(lambda r: r.method == "POST" and r.url.endswith("/projects/combine/preview")):
        page.get_by_test_id("combine-start").click()
        page.get_by_role("button", name="Start combine", exact=True).click()
    refusal = page.get_by_test_id("combine-start-refusal")
    expect(refusal).to_contain_text("a source changed since the preview; preview again", timeout=ACTION_TIMEOUT_MS)
    assert refusal.get_attribute("data-code") == "preview_stale"
    assert len(previews) > settled

    # Start again, now with the new sha (the refusal clears on the next edit
    # or preview; the button re-enables once the fresh preview is on screen).
    expect(page.get_by_test_id("combine-start")).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("combine-start").click()
    page.get_by_role("button", name="Start combine", exact=True).click()
    page.wait_for_url(re.compile(rf"/projects/combine/{JOB_ID}$"), timeout=ACTION_TIMEOUT_MS)
    assert [s["expected_preview_sha"] for s in starts] == ["sha-1", "sha-2"]


def test_failed_job_undo_opens_the_delete_dry_run(stub, page, app_url):
    merged = project(
        API_PREFIX,
        "merged",
        display_name="Merged",
        status="failed",
        writable=False,
        selectable=False,
        origin={"kind": "combine", "job_id": JOB_ID, "sources": ["widgets-a", "widgets-b"]},
    )
    serve_projects(stub, merged=merged)
    serve_combine(
        stub,
        jobs=[job(status="failed", phase="images", error="combine failed: disk full", report={"images_copied": 3})],
    )
    dry_runs: list[str] = []

    def delete(request: Any, m: Any):
        qs = parse_qs(urlparse(request.url).query)
        if qs.get("dry_run") == ["true"]:
            dry_runs.append(m.group(1))
            return (
                200,
                {
                    "indexes": [{"name": "op_merged_items", "docs": 3, "store_bytes": None}],
                    "dirs": [],
                    "promoted_models": [],
                    "mlflow_experiment": None,
                    "running_jobs": [],
                    "referenced_by": [],
                    "blocking": [],
                    "blocking_detail": [],
                },
            )
        return (500, {"detail": "unexpected real delete"})

    stub.on("DELETE", rf"^{re.escape(API_PREFIX)}/projects/([^/]+)$", delete)

    page.goto(f"{app_url}/projects/combine/{JOB_ID}")
    expect(page.get_by_test_id("combine-job-served-error")).to_have_text(
        "combine failed: disk full", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("combine-open-project")).to_have_count(0)
    page.get_by_test_id("combine-undo").click()
    expect(page.get_by_test_id("delete-project-title")).to_contain_text("Undo combine", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("delete-project-report")).to_contain_text("op_merged_items", timeout=ACTION_TIMEOUT_MS)
    assert dry_runs == ["merged"]

    # /projects shows the same project's job link and the renamed action.
    page.goto(f"{app_url}/projects")
    page.get_by_test_id("project-row-merged").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_test_id("project-combine-job-merged").get_attribute("href") == f"/projects/combine/{JOB_ID}"
    expect(page.get_by_test_id("project-delete-merged")).to_have_text("Undo combine")
    expect(page.get_by_test_id("projects-combine")).to_be_visible()


def test_feature_absent_on_a_plain_404(stub, page, app_url):
    # conftest's default: GET {prefix}/projects/combine/<id> -> plain 404.
    serve_projects(stub)

    with page.expect_response(lambda r: "/projects/combine/" in r.url):
        page.goto(f"{app_url}/projects")
    page.get_by_test_id("project-row-widgets-a").wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("projects-combine")).to_have_count(0)

    page.goto(f"{app_url}/projects/combine")
    expect(page.get_by_test_id("combine-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-add-source")).to_have_count(0)

    page.goto(f"{app_url}/projects/combine/{JOB_ID}")
    expect(page.get_by_test_id("combine-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    seen = stub.handled + stub.unhandled
    combine = [(m, p) for (m, p) in seen if "/projects/combine" in p]
    assert combine and all(m == "GET" and p.endswith("/projects/combine/__probe__") for (m, p) in combine), combine


def test_conflict_review_link_leads_to_a_review_that_sends_the_filter(stub, page, app_url):
    """Cross-feature walk (plan §6 I-2): the finished job's "Review flagged
    conflicts" link opens `/review` with `combine_conflict=true` sent to the
    served queue, and a combined item's Details show its origin rows and the
    VLM provenance rows."""
    from test_import_leftovers import review_item, review_requests, serve_review, tabs_with_imported

    completed = job(status="completed", phase="done", done=20, finished_at="2026-10-01T12:05:00Z", report={})
    serve_combine(stub, jobs=[completed])
    merged = project(API_PREFIX, "merged")
    stub.on(
        "GET",
        r"^/curation/projects$",
        projects_response(API_PREFIX, [project(API_PREFIX, "default", is_default=True, deletable=False), merged]),
    )
    stub.on("GET", r"/projects/merged$", merged)
    seen = serve_review(stub, tabs_with_imported())

    def combined_items(request: Any, _m: Any):
        seen.append({"path": [urlparse(request.url).path], **parse_qs(urlparse(request.url).query)})
        item = review_item(
            0,
            origin_project="widgets-a",
            origin_item_id="it_77",
            combine_conflict=True,
            vlm_endpoint="env@abc123",
            vlm_model="example/vision-model",
        )
        return (200, {"items": [item], "total": 1, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", combined_items)
    stub.on("GET", r"/crops/[^/]+/history$", {"crop_id": "crop-0", "entries": []})

    page.goto(f"{app_url}/projects/combine/{JOB_ID}")
    link = page.get_by_test_id("combine-review-conflicts")
    expect(link).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(lambda r: "/review/all" in r.url, timeout=ACTION_TIMEOUT_MS):
        link.click()
    reqs = review_requests(seen, "all")
    assert reqs and reqs[0]["combine_conflict"] == ["true"], (seen, page.url)
    expect(page.get_by_test_id("filter-chip-combine-conflict")).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    page.get_by_role("button", name="Details").click()
    expect(page.get_by_test_id("combine-origin-project")).to_contain_text("widgets-a", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("combine-origin-conflict")).to_be_visible()
    expect(page.get_by_test_id("combine-origin-item")).to_have_text("it_77")
    expect(page.get_by_test_id("vlm-provenance-endpoint")).to_have_text("env@abc123")
    expect(page.get_by_test_id("vlm-provenance-model")).to_have_text("example/vision-model")
