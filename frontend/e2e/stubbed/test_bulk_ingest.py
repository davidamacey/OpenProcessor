"""Stubbed e2e coverage for bulk ingest on `/ingest`: start a run on a folder
or manifest, follow served progress, cancel, resume, and read the failures.

The run lives in the backend; these tests serve a scripted sequence of runs and
check what the page sends and what it renders from what is served.
"""

from __future__ import annotations

import json

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_ingest import BULK_CONFIG, _base_ingest_stubs  # noqa: F401 - shared stub set

RUN_ID = "bulk_20261011T120000_1a2b3c4d"
NARROW = {"width": 430, "height": 900}


def _progress(**over):
    base = {
        "discovered": 0, "discovery_complete": False, "skipped": 0, "ingested": 0,
        "duplicates": 0, "failed": 0, "processed": 0, "crops_indexed": 0,
        "n_embedded": 0, "n_not_embedded": 0, "n_embed_failed": 0, "in_flight": 0,
        "images_per_s": None, "images_per_s_recent": None, "eta_s": None, "elapsed_s": 0.0,
    }
    return {**base, **over}


def _run(status="running", progress=None, **over):
    live = status in ("queued", "running")
    run = {
        "project": "default", "run_id": RUN_ID, "status": status, "resumed": False,
        "source": {"kind": "folder", "path": "/data/archive/coco", "extensions": [".jpg"]},
        "options": {"source_tag": "bulk_ingest", "batch_size": 16, "concurrency": 8},
        "progress": progress or _progress(),
        "stage_s": {}, "attempts": 1,
        "can_cancel": live, "can_resume": status in ("cancelled", "failed", "completed_with_errors"),
        "cancel_requested": False, "error": None,
        "started_at": "2026-10-11T12:00:00+00:00", "finished_at": None,
        "poll_after_s": 0.3 if live else None,
    }
    run.update(over)
    return run


def _base(stub) -> None:
    """The shared ingest stubs; call before a test registers its own (later wins)."""
    _base_ingest_stubs(stub, batch_source_roots=["/data/archive"])
    stub.on("POST", r"/ingest/path_lookup", {"known_paths": {}})


def _open(stub, page, app_url, *, base: bool = True) -> None:
    if base:
        _base(stub)
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector("text=Bulk ingest (folder or manifest)")
    page.get_by_test_id("bulk-path").wait_for(timeout=ACTION_TIMEOUT_MS)


def test_start_follow_progress_cancel_and_resume(stub, page, app_url):
    started = []
    state = {"polls": 0, "cancelled": False}

    def start(request, _m):
        started.append(json.loads(request.post_data))
        state["cancelled"] = False
        state["polls"] = 0
        return (202, _run("running", _progress(discovered=64, processed=16, ingested=16)))

    def get_run(_request, _m):
        if state["cancelled"]:
            return _run(
                "cancelled",
                _progress(discovered=400, processed=120, ingested=118, duplicates=2,
                          discovery_complete=True, images_per_s=20.1),
            )
        state["polls"] += 1
        n = min(state["polls"], 4)
        return _run(
            "running",
            _progress(
                discovered=200 + 50 * n, processed=40 * n, ingested=40 * n, in_flight=16,
                images_per_s=19.0 + n, images_per_s_recent=20.0 + n, eta_s=300 - 40 * n,
                elapsed_s=2.0 * n,
            ),
            stage_s={"embed": 12.5 * n, "batch_requests": 3.0 * n},
        )

    def cancel(_request, _m):
        state["cancelled"] = True
        return _run("running", _progress(discovered=300, processed=80), cancel_requested=True)

    stub.on("POST", r"/ingest/bulk$", start)
    stub.on("GET", rf"/ingest/bulk/{RUN_ID}$", get_run)
    stub.on("POST", rf"/ingest/bulk/{RUN_ID}/cancel$", cancel)
    _open(stub, page, app_url)

    page.get_by_test_id("bulk-path").fill("/data/archive/coco")
    page.get_by_test_id("bulk-source").fill("census")
    page.get_by_test_id("bulk-start").click()

    page.get_by_test_id("bulk-run").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert started == [{"folder": "/data/archive/coco", "source": "census"}]
    # Progress advances from what is served, with rate and time left.
    expect(page.get_by_test_id("bulk-ingested")).to_have_text("ingested 160", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("bulk-rate")).to_contain_text("img/s")
    expect(page.get_by_test_id("bulk-eta")).not_to_have_text("--")
    expect(page.get_by_test_id("bulk-start")).to_be_disabled()
    expect(page.get_by_test_id("bulk-path")).to_be_disabled()
    page.get_by_text("Where the time went").click()
    expect(page.get_by_test_id("bulk-stages")).to_contain_text("embed")

    page.get_by_test_id("bulk-cancel").click()
    expect(page.get_by_test_id("bulk-status")).to_have_text("cancelled", timeout=ACTION_TIMEOUT_MS)
    assert ("POST", f"/curation/projects/default/ingest/bulk/{RUN_ID}/cancel", None) in [
        (m, p, b) for m, p, b in stub.calls
    ]
    expect(page.get_by_test_id("bulk-cancel")).to_have_count(0)

    # Resume starts the same source again with the run's own options.
    page.get_by_test_id("bulk-resume").click()
    expect(page.get_by_test_id("bulk-status")).to_have_text("running", timeout=ACTION_TIMEOUT_MS)
    assert started[-1] == {
        "folder": "/data/archive/coco", "source": "bulk_ingest", "extensions": [".jpg"],
        "batch_size": 16, "concurrency": 8,
    }


def test_a_finished_run_lists_the_failed_images_with_the_reason(stub, page, app_url):
    run = _run(
        "completed_with_errors",
        _progress(discovered=5, discovery_complete=True, processed=5, ingested=3, failed=2,
                  images_per_s=18.0, elapsed_s=0.3),
    )
    stub.on("POST", r"/ingest/bulk$", (202, run))
    stub.on("GET", rf"/ingest/bulk/{RUN_ID}$", run)
    stub.on(
        "GET", rf"/ingest/bulk/{RUN_ID}/failures",
        {"items": [
            {"path": "/data/archive/coco/a.jpg", "error": "cannot read", "error_kind": "decode_failed"},
            {"path": "/outside/b.jpg", "error": "not under a source root", "error_kind": "unservable_path"},
        ], "total": 2, "offset": 0, "limit": 20},
    )
    _open(stub, page, app_url)
    page.get_by_test_id("bulk-path").fill("/data/archive/coco")
    page.get_by_test_id("bulk-start").click()

    failures = page.get_by_test_id("bulk-failures")
    failures.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(failures).to_contain_text("2 failed images")
    expect(failures).to_contain_text("a.jpg")
    expect(failures).to_contain_text("[unservable_path]")
    expect(failures).to_contain_text("not under a source root")
    expect(page.get_by_test_id("bulk-resume")).to_be_visible()


def test_a_refused_start_shows_the_served_reason(stub, page, app_url):
    stub.on(
        "POST", r"/ingest/bulk$",
        (409, {"detail": {"error": "bulk_ingest_busy", "message": "a bulk ingest is already running",
                          "run_id": RUN_ID}}),
    )
    _open(stub, page, app_url)
    page.get_by_test_id("bulk-path").fill("/data/archive/coco")
    page.get_by_test_id("bulk-start").click()
    expect(page.get_by_test_id("bulk-action-error")).to_contain_text(
        "a bulk ingest is already running", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("bulk-start")).to_be_enabled()


def test_a_run_started_elsewhere_is_picked_up_on_load(stub, page, app_url):
    live = _run("running", _progress(discovered=900, processed=300, ingested=300, images_per_s=21.0))
    _base(stub)
    stub.on("GET", r"/ingest/bulk(\?|$)", {"items": [live], "total": 1, "page": 1, "page_size": 5})
    stub.on("GET", rf"/ingest/bulk/{RUN_ID}$", live)
    _open(stub, page, app_url, base=False)
    expect(page.get_by_test_id("bulk-status")).to_have_text("running", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("bulk-start")).to_be_disabled()


def test_a_manifest_run_sends_the_manifest(stub, page, app_url):
    started = []

    def start(request, _m):
        started.append(json.loads(request.post_data))
        return (202, _run("queued"))

    stub.on("POST", r"/ingest/bulk$", start)
    stub.on("GET", rf"/ingest/bulk/{RUN_ID}$", _run("queued"))
    _open(stub, page, app_url)
    page.get_by_label("Manifest", exact=True).check()
    page.get_by_test_id("bulk-path").fill("/data/archive/list.txt")
    page.get_by_test_id("bulk-ingest").get_by_text("Advanced").click()
    page.get_by_label("Images per batch").fill("8")
    page.get_by_test_id("bulk-start").click()
    page.get_by_test_id("bulk-run").wait_for(timeout=ACTION_TIMEOUT_MS)
    assert started == [{"manifest": "/data/archive/list.txt", "source": "bulk_ingest", "batch_size": 8}]


def test_no_horizontal_overflow_at_430px_with_a_live_run(stub, page, app_url):
    live = _run(
        "running",
        _progress(discovered=1200, processed=400, ingested=390, duplicates=10, failed=3,
                  images_per_s=21.3, images_per_s_recent=22.0, eta_s=3600, elapsed_s=125.0),
        stage_s={"embed": 100.0, "batch_requests": 50.0},
        source={"kind": "folder", "extensions": [".jpg"],
                "path": "/data/archive/a/very/long/path/to/a/folder/with/many/nested/levels/of/images"},
    )
    _base(stub)
    stub.on("GET", r"/ingest/bulk(\?|$)", {"items": [live], "total": 1, "page": 1, "page_size": 5})
    stub.on("GET", rf"/ingest/bulk/{RUN_ID}$", live)
    page.set_viewport_size(NARROW)
    _open(stub, page, app_url, base=False)
    page.get_by_test_id("bulk-run").wait_for(timeout=ACTION_TIMEOUT_MS)
    width = page.evaluate(
        "() => ({sw: document.documentElement.scrollWidth, iw: window.innerWidth})"
    )
    assert width["sw"] <= width["iw"], width
