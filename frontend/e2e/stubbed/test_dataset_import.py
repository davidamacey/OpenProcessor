"""OpenProcessor W10 (labeled-dataset import + Reprocess), any_domain_plan.md
§7.12; docs/design/w10-import-reprocess-ui-plan-2026-09-27.md.

The backend has not shipped W10 yet, so every payload below follows the
spec's own examples. The fail-closed stub means any W10 request the page
makes that a test did not expect fails the test.

  1. `test_import_with_mapping_starts_and_follows_the_job` — /ingest links
     to the wizard; a preview, an explicit name-based mapping (one `map`,
     one `create`), Start behind a confirm, the exact POST body, then the
     job view.
  2. `test_job_progresses_to_done` — the job view follows the served
     `poll_after_s` from running to completed and shows the served labels
     and counts.
  3. `test_failed_job_shows_its_served_error`.
  4. `test_reprocess_confirm_on_cluster_selection` — a served dry run
     precedes the apply; both bodies asserted.
  5. `test_feature_absent_when_formats_404` — conftest's default 404:
     no ingest link, no Reprocess button, the wizard route says so, and no
     other W10 request fires.
"""

from __future__ import annotations

import copy
from typing import Any

from conftest import ACTION_TIMEOUT_MS, expect_handled
from playwright.sync_api import expect

from fixtures.wire import make_item
from test_ingest import _base_ingest_stubs

IMPORT_ID = "imp_20260927T120000_1a2b3c4d"
IMPORT_KEY = "k" * 64

CLASSES = [
    {
        "class_id": 2,
        "class_name": "widget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 3,
        "deprecated": False,
    },
    {
        "class_id": 5,
        "class_name": "gadget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 4,
        "validated_count": 1,
        "cluster_size": 4,
        "deprecated": False,
    },
]

FORMATS: dict[str, Any] = {
    "formats": [
        {"format": "auto", "label": "Detect automatically"},
        {"format": "yolo", "label": "YOLO (data.yaml)"},
        {"format": "coco", "label": "COCO JSON"},
        {"format": "openprocessor_export", "label": "OpenProcessor export"},
    ],
    "processing_modes": [
        {
            "value": "none",
            "label": "Import as-is",
            "description": "Index the images and import the labels. No detector or VLM runs.",
        },
        {
            "value": "propose",
            "label": "Import and find missed objects",
            "description": "Also run the detector, region and VLM pipeline. Anything the labels missed arrives as a proposal for review; imported labels are never changed.",
        },
    ],
    "parents_modes": [
        {"value": "auto", "label": "Automatic", "description": ""},
        {"value": "labels", "label": "From the dataset's labels", "description": ""},
        {"value": "detect", "label": "Detect them", "description": ""},
    ],
    "trust_levels": [
        {"value": "validated", "label": "Trusted (validated)", "description": ""},
        {"value": "suggestion", "label": "Suggestions to review", "description": ""},
    ],
    "mapping_actions": [
        {"value": "map", "label": "Map to class", "description": ""},
        {"value": "create", "label": "Create class", "description": ""},
        {"value": "skip", "label": "Skip", "description": ""},
        {"value": "region", "label": "Region boxes", "description": ""},
    ],
    "match_kinds": [
        {"value": "exact", "label": "Same name", "description": ""},
        {"value": "case_insensitive", "label": "Same name, different case", "description": ""},
        {"value": "synonym", "label": "Synonym", "description": ""},
        {"value": "none", "label": "No match", "description": ""},
    ],
    "issues": [
        {
            "code": "label_file_missing",
            "id": "label_file_missing",
            "severity": "warning",
            "blocking": False,
            "bypassable": False,
            "label": "Images without a label file",
        },
        {
            "code": "class_index_name_mismatch",
            "id": "class_index_name_mismatch",
            "severity": "info",
            "blocking": False,
            "bypassable": False,
            "label": "The old index path would have mislabeled a class",
        },
    ],
    "upload_limits": {
        "max_bytes": 2147483648,
        "max_files": 200000,
        "ttl_hours": 24,
        "preview_max_files": 5000,
    },
    "status_labels": {
        "queued": "Queued",
        "running": "Importing",
        "paused_backpressure": "Waiting for the region worker",
        "completed": "Done",
        "completed_with_errors": "Done with errors",
        "failed": "Failed",
        "cancelled": "Cancelled",
        "interrupted": "Interrupted",
        "undoing": "Undoing",
        "undone": "Undone",
    },
}


def preview_for(body: dict[str, Any]) -> dict[str, Any]:
    """A server-shaped preview: `resolved` reflects the request's explicit
    mapping, the way the backend reports what the request covers."""
    chosen = {m["dataset_class"]: m for m in body.get("mapping", [])}

    def resolved(name: str) -> dict[str, Any] | None:
        m = chosen.get(name)
        if not m:
            return None
        if m["action"] == "map":
            cls = next(c for c in CLASSES if c["class_id"] == m["class_id"])
            return {"dataset_class": name, "kind": "item", "class_id": cls["class_id"], "class_name": cls["class_name"]}
        if m["action"] == "create":
            return {"dataset_class": name, "kind": "item", "class_id": None, "class_name": m["new_class_name"]}
        return {"dataset_class": name, "kind": m["action"], "class_id": None, "class_name": None}

    return {
        "project": "default",
        "format": "yolo",
        "root": body["source"]["path"],
        "source_sha": "a" * 64,
        "import_key": IMPORT_KEY,
        "op_export": None,
        "splits": [
            {"split": "train", "images": 67, "labeled": 58, "negatives": 8, "unlabeled": 1, "boxes": 212},
            {"split": "val", "images": 14, "labeled": 12, "negatives": 2, "unlabeled": 0, "boxes": 41},
            {"split": "test", "images": 15, "labeled": 13, "negatives": 2, "unlabeled": 0, "boxes": 44},
        ],
        "totals": {"images": 96, "boxes": 297, "images_already_indexed": 0, "images_to_ingest": 96},
        "classes": [
            {
                "dataset_class": "Widget",
                "dataset_id": 0,
                "boxes": 250,
                "images": 70,
                "source_class_id": None,
                "index_would_have_mapped_to": {"class_id": 5, "class_name": "gadget"},
                "suggestion": {"action": "map", "class_id": 2, "class_name": "widget", "match": "case_insensitive"},
                "resolved": resolved("Widget"),
            },
            {
                "dataset_class": "sprocket",
                "dataset_id": 1,
                "boxes": 47,
                "images": 20,
                "source_class_id": None,
                "index_would_have_mapped_to": None,
                "suggestion": {"action": "create", "class_id": None, "class_name": "sprocket", "match": "none"},
                "resolved": resolved("sprocket"),
            },
        ],
        "region": None,
        "issues": [
            {
                "code": "label_file_missing",
                "id": "label_file_missing",
                "severity": "warning",
                "blocking": False,
                "bypassable": False,
                "message": "1 image has no label file.",
                "count": 1,
                "samples": [{"file": "images/train/w_0007.jpg", "line": None, "detail": {}}],
            },
            {
                "code": "class_index_name_mismatch",
                "id": "class_index_name_mismatch",
                "severity": "info",
                "blocking": False,
                "bypassable": False,
                "message": "By index, class 0 would have been imported as gadget.",
                "count": 1,
                "samples": [],
            },
        ],
        "blocking": False,
        "force_allowed": False,
        "estimate": {"detector_images": 0, "embeddings": 393},
    }


def job(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "project": "default",
        "import_id": IMPORT_ID,
        "import_key": IMPORT_KEY,
        "name": "widgets_v1",
        "status": "running",
        "reused": False,
        "progress": {
            "images_total": 96,
            "images_done": 64,
            "images_failed": 0,
            "chunks_total": 2,
            "chunks_done": 1,
            "images_per_s": 21.5,
            "eta_s": 2,
        },
        "waiting_for": None,
        "report": {
            "images_created": 64,
            "images_reused": 0,
            "images_failed": 0,
            "images_skipped": 0,
            "items_reconciled_removed": 0,
            "items_created": 190,
            "items_updated": 0,
            "items_noop": 0,
            "labels_written": 190,
            "boxes_written": 0,
            "standalone_regions": 0,
            "negatives": 8,
            "unlabeled": 1,
            "parents_detected": 0,
            "proposals_created": 0,
            "proposals_merged": 0,
            "holdout_frozen": 30,
            "label_conflicts_locked": 0,
            "disagreements": {"counts": {}, "samples": []},
        },
        "mapping": [
            {"dataset_class": "Widget", "kind": "item", "class_id": 2, "class_name": "widget"},
            {"dataset_class": "sprocket", "kind": "item", "class_id": 9, "class_name": "sprocket"},
        ],
        "options": {},
        "source": {"format": "yolo", "root": "/data/source/widgets/yolo", "source_sha": "a" * 64},
        "issues_summary": [],
        "undo": None,
        "next_steps": [],
        "started_at": "2026-09-27T12:00:00Z",
        "updated_at": "2026-09-27T12:00:03Z",
        "finished_at": None,
        "poll_after_s": 1,
        "labels": {"status": dict(FORMATS["status_labels"])},
        "error": None,
    }
    base.update(over)
    base.setdefault("actions", actions_for(base["status"]))
    return base


def actions_for(status: str) -> dict[str, Any]:
    """The served `actions` the stub backend computes for a status."""

    def act(allowed: bool, why: str) -> dict[str, Any]:
        return {"allowed": allowed, "reason": None if allowed else why}

    return {
        "can_cancel": act(
            status in {"queued", "running", "paused_backpressure"}, "The import is not running."
        ),
        "can_resume": act(
            status in {"interrupted", "failed", "cancelled"}, "Only a stopped import can resume."
        ),
        "can_undo": act(
            status
            in {"completed", "completed_with_errors", "failed", "cancelled", "interrupted"},
            "The import is still running.",
        ),
    }


def completed_job() -> dict[str, Any]:
    j = job(status="completed", poll_after_s=None, finished_at="2026-09-27T12:00:05Z")
    j["progress"] = {**j["progress"], "images_done": 96, "chunks_done": 2, "eta_s": 0}
    j["report"] = {**j["report"], "images_created": 96, "items_created": 285, "labels_written": 285}
    j["next_steps"] = [
        {
            "action": "Cluster the region boxes",
            "method": "POST",
            "path": "/regions/cluster",
            "reason": "Group the imported region boxes for review.",
        }
    ]
    return j


def serve_w10(stub: Any) -> None:
    stub.on("GET", r"/datasets/formats(\?|$)", copy.deepcopy(FORMATS))
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})


def serve_ingest_page(stub: Any) -> None:
    """/ingest's own mount reads (config, status, region drain, the
    clustering handoff), then this module's registry."""
    _base_ingest_stubs(stub)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})


def serve_job_sequence(stub: Any, sequence: list[dict[str, Any]]) -> list[int]:
    reads = [0]

    def handler(_request: Any, _match: Any):
        i = min(reads[0], len(sequence) - 1)
        reads[0] += 1
        return (200, sequence[i])

    stub.on("GET", rf"/datasets/imports/{IMPORT_ID}$", handler)
    return reads


def test_import_with_mapping_starts_and_follows_the_job(stub, page, app_url):
    serve_w10(stub)
    stub.on("GET", r"/datasets/imports(\?|$)", {"items": [], "total": 0, "page": 1, "page_size": 50})
    previews: list[dict[str, Any]] = []

    def preview_handler(request: Any, _match: Any):
        body = request.post_data_json
        previews.append(body)
        return (200, preview_for(body))

    stub.on("POST", r"/datasets/preview$", preview_handler)
    started: list[dict[str, Any]] = []

    def start_handler(request: Any, _match: Any):
        started.append(request.post_data_json)
        return (202, job())

    stub.on("POST", r"/datasets/imports$", start_handler)
    serve_job_sequence(stub, [job(), completed_job()])
    serve_ingest_page(stub)

    page.goto(f"{app_url}/p/default/ingest")
    link = page.get_by_test_id("ingest-dataset-import-link")
    link.wait_for(timeout=ACTION_TIMEOUT_MS)
    link.click()

    page.get_by_test_id("import-path").fill("/data/source/widgets/yolo")
    table = page.get_by_test_id("mapping-table")
    table.wait_for(timeout=ACTION_TIMEOUT_MS)
    widget = page.locator('[data-dataset-class="Widget"]')
    sprocket = page.locator('[data-dataset-class="sprocket"]')
    expect(widget).to_have_attribute("data-unmapped", "true")
    expect(widget.get_by_test_id("suggestion-chip")).to_have_text("Same name, different case")
    expect(widget.get_by_test_id("index-hint")).to_contain_text("gadget")
    expect(page.get_by_role("button", name="Start import")).to_be_disabled()

    widget.get_by_label("Action for Widget").select_option("map")
    widget.get_by_label("Class for Widget").select_option(label="widget")
    sprocket.get_by_role("button", name="Use suggestion").click()
    expect(widget.get_by_test_id("resolved")).to_have_text("widget", timeout=ACTION_TIMEOUT_MS)
    expect(sprocket.get_by_test_id("resolved")).to_have_text("sprocket", timeout=ACTION_TIMEOUT_MS)

    start = page.get_by_role("button", name="Start import")
    expect(start).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    start.click()
    dialog = page.get_by_role("dialog", name="Start the import")
    expect(dialog).to_contain_text("96 images and 297 boxes")
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/datasets/imports")):
        dialog.get_by_role("button", name="Start import").click()

    assert len(started) == 1
    body = started[0]
    assert body == {
        "source": {"path": "/data/source/widgets/yolo", "format": "auto"},
        "mapping": [
            {"dataset_class": "Widget", "action": "map", "class_id": 2},
            {"dataset_class": "sprocket", "action": "create", "new_class_name": "sprocket"},
        ],
        "accept_suggestions": False,
        "options": {},
        "expected_import_key": IMPORT_KEY,
    }, body
    assert all("project" not in p for p in previews)

    page.wait_for_url(f"**/p/default/datasets/imports/{IMPORT_ID}", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("job-status")).to_have_text("Done", timeout=ACTION_TIMEOUT_MS)

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_job_progresses_to_done(stub, page, app_url):
    serve_w10(stub)
    reads = serve_job_sequence(stub, [job(), job(), completed_job()])

    page.goto(f"{app_url}/p/default/datasets/imports/{IMPORT_ID}")
    status = page.get_by_test_id("job-status")
    expect(status).to_have_text("Importing", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("job-progress")).to_contain_text("64 / 96")
    expect(page.get_by_role("button", name="Cancel import")).to_be_visible()

    expect(status).to_have_text("Done", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("job-progress")).to_contain_text("96 / 96")
    expect(page.get_by_test_id("report-items_created")).to_have_text("285")
    expect(page.get_by_test_id("report-holdout_frozen")).to_have_text("30")
    expect(page.get_by_test_id("next-steps")).to_contain_text("Group the imported region boxes")
    expect(page.get_by_role("button", name="Cancel import")).to_have_count(0)
    expect(page.get_by_role("button", name="Undo import")).to_be_visible()
    assert reads[0] >= 3, reads


def test_failed_job_shows_its_served_error(stub, page, app_url):
    serve_w10(stub)
    failed = job(
        status="failed",
        poll_after_s=None,
        error="6 consecutive chunks failed: the items index is unavailable.",
    )
    serve_job_sequence(stub, [failed])

    page.goto(f"{app_url}/p/default/datasets/imports/{IMPORT_ID}")
    expect(page.get_by_test_id("job-status")).to_have_text("Failed", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("job-error")).to_have_text(
        "6 consecutive chunks failed: the items index is unavailable."
    )
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    expect(page.get_by_role("button", name="Cancel import")).to_have_count(0)
    expect(page.get_by_test_id("job-action-reasons")).to_contain_text(
        "Cancel import: The import is not running."
    )


def serve_cluster(stub: Any) -> None:
    stub.on(
        "GET",
        r"/clusters(\?|$)",
        {
            "items": [
                {
                    "cluster_id": 2,
                    "cluster_kind": "class",
                    "size": 3,
                    "validated_count": 0,
                    "dominant_class_id": 2,
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
        },
    )
    crops = [
        make_item(
            crop_id=f"crop-{i}",
            image_id=f"img-{i}",
            class_id=2,
            class_name="widget",
            cluster_id=2,
            label_validated=False,
            label_source="model_suggestion",
            thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        )
        for i in range(3)
    ]
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": crops})


def test_reprocess_confirm_on_cluster_selection(stub, page, app_url):
    serve_w10(stub)
    serve_cluster(stub)
    bodies: list[dict[str, Any]] = []

    def reprocess_handler(request: Any, _match: Any):
        body = request.post_data_json
        bodies.append(body)
        if body["dry_run"]:
            return (
                200,
                {
                    "dry_run": True,
                    "scopes": [{"scope": "region", "selected": 1, "locked_skipped": 0, "queued": 0, "breakdown": []}],
                    "job": None,
                    "items": [],
                },
            )
        return (
            200,
            {
                "dry_run": False,
                "scopes": [{"scope": "region", "selected": 1, "locked_skipped": 0, "queued": 1, "breakdown": []}],
                "job": None,
                "items": [],
            },
        )

    stub.on("POST", r"/reprocess$", reprocess_handler)

    page.goto(f"{app_url}/p/default/clusters/2")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.locator("img").nth(1).click()
    expect(page.get_by_text("1 selected").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    page.get_by_test_id("reprocess-open").click()
    dialog = page.get_by_role("dialog", name="Reprocess")
    # Scope labels are the served reprocess vocabulary's.
    expect(dialog.get_by_test_id("reprocess-lock-rule")).to_have_count(0)
    expect(dialog.locator("fieldset label")).to_have_text(
        ["Find objects", "Find by description", "Region stage", "Vision model", "Compute vectors"]
    )
    dialog.get_by_label("Region stage", exact=True).check()
    dialog.get_by_role("combobox").select_option("redetect")
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/reprocess")):
        dialog.get_by_role("button", name="Check what would run").click()
    # scope, selected, locked skipped, queued, failed, not found (omitted = em dash)
    expect(dialog.get_by_test_id("reprocess-dry-run").locator("tbody tr td")).to_have_text(
        ["Region stage", "1", "0", "0", "\u2014", "\u2014"]
    )
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/reprocess")):
        dialog.get_by_role("button", name="Reprocess").click()
    expect(dialog.get_by_test_id("reprocess-result").locator("tbody tr td")).to_have_text(
        ["Region stage", "1", "0", "1", "\u2014", "\u2014"]
    )

    selected = bodies[0]["targets"]["crop_ids"]
    assert len(selected) == 1 and selected[0].startswith("crop-"), bodies
    expected = {"targets": {"crop_ids": selected}, "scopes": ["region"], "region_mode": "redetect"}
    assert bodies == [{**expected, "dry_run": True}, {**expected, "dry_run": False}], bodies


def test_feature_absent_when_formats_404(stub, page, app_url):
    # conftest's default: GET {prefix}/datasets/formats -> 404.
    serve_ingest_page(stub)
    serve_cluster(stub)

    with page.expect_response(lambda r: r.url.endswith("/datasets/formats")):
        page.goto(f"{app_url}/p/default/ingest")
    page.get_by_role("heading", name="Ingest", exact=True).wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("ingest-dataset-import-link")).to_have_count(0)

    page.goto(f"{app_url}/p/default/datasets/import")
    expect(page.get_by_test_id("datasets-unavailable")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("import-path")).to_have_count(0)

    with page.expect_response(lambda r: r.url.endswith("/datasets/formats")):
        page.goto(f"{app_url}/p/default/clusters/2")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    page.locator("img").nth(1).click()
    expect(page.get_by_text("1 selected").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("reprocess-open")).to_have_count(0)

    seen = stub.handled + stub.unhandled
    w10 = [p for (_m, p) in seen if "/datasets/" in p or "/reprocess" in p]
    assert w10 and all(p.endswith("/datasets/formats") for p in w10), w10
