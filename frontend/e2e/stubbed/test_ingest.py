"""Stubbed e2e coverage for `/ingest`
(docs/design/ingest-ui-and-acceptance-plan-2026-09-24.md §B.4).

Fixtures live under `e2e/fixtures/ingest/batch1/` — `a.jpg` at the root,
`sub/a.jpg` (a duplicate basename in a nested folder, proving the
identifier is the full relative path, not just the filename), and
`b.jpg`.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import expect

from fixtures.multipart import parse_multipart

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingest" / "batch1"

IDLE_JOB = {
    "job_id": "job-0",
    "status": "idle",
    "stage": "",
    "processed": 0,
    "total": 0,
    "started_at": 0,
    "finished_at": 0,
    "error": None,
}


def _base_ingest_stubs(
    stub,
    *,
    status_total=0,
    drain_unfinished=0,
    drained=None,
    batch_source_roots=None,
    stall_reason=None,
    region_dependencies=None,
) -> None:
    """BA-1..BA-7 (OpenProcessor #36, c5c606f) baseline: `/ingest/config` is
    now real and always stubbed here (the page fetches it once
    `ingestAvailability` confirms the router is mounted), and the drain
    response always carries the BA-3 `drained` verdict.

    `drained` defaults to `drain_unfinished == 0` (the obvious case — a
    caller that wants to exercise "just reached zero, not yet stable"
    passes `drained=False` explicitly alongside `drain_unfinished=0`).
    """
    if drained is None:
        drained = drain_unfinished == 0
    # The root layout's classesStore.acquire() fires on every route.
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on(
        "GET",
        r"/ingest/status(\?|$)",
        {"total": status_total, "by_source": [], "by_day": []},
    )
    stub.on(
        "GET",
        r"/ingest/config(\?|$)",
        {
            "upload": {
                "enabled": True,
                "max_images_per_request": 128,
                "max_bytes_per_request": 268435456,
                "accepted_extensions": [".jpg", ".jpeg", ".png"],
                "persists_bytes": True,
            },
            "batch": {
                "enabled": True,
                "max_items": 256,
                "source_roots": batch_source_roots or [],
            },
            "region_drain": {"poll_interval_s": 10, "stable_polls": 3},
        },
    )
    stub.on(
        "GET",
        r"/ingest/region_drain(\?|$)",
        {
            "pending_detection": drain_unfinished,
            "pending_verification": 0,
            "total_unfinished": drain_unfinished,
            "drained": drained,
            "stable_for_s": 30 if drained else 0,
            "observed_at": "2026-09-25T00:00:00Z",
            # OpenProcessor d72cc63 (V-1): always served.
            "region_dependencies": region_dependencies or [],
            "stall_reason": stall_reason,
        },
    )
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)


def _select_folder(page, app_url) -> None:
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    folder_input = page.locator("input[type=file]").nth(1)
    folder_input.set_input_files(str(FIXTURES))


def test_upload_folder_happy_path(stub, page, app_url):
    _base_ingest_stubs(stub)

    lookup_calls = []

    def lookup_handler(request, _match):
        parsed = json.loads(request.post_data)
        lookup_calls.append(parsed["image_paths"])
        # Mark the "sub/a.jpg" identifier as already known.
        known = {
            p: "img-known" for p in parsed["image_paths"] if p.endswith("sub/a.jpg")
        }
        return {"known_paths": known}

    upload_calls = []

    def upload_handler(request, _match):
        parsed = parse_multipart(request.post_data)
        upload_calls.append(parsed)
        results = []
        for i, p in enumerate(parsed.image_paths):
            if p.endswith("b.jpg"):
                results.append(
                    {
                        "status": "failed",
                        "image_id": "",
                        # BA-1: even a failed item's image_path is whatever the
                        # client sent (never persisted) -- source_identifier
                        # still echoes the client identifier either way.
                        "image_path": p,
                        "imohash": "",
                        "n_crops": 0,
                        "n_regions": 0,
                        "error": "decode failed",
                        "error_kind": "decode_failed",
                        "source_identifier": p,
                    }
                )
            else:
                results.append(
                    {
                        "status": "success",
                        "image_id": f"img-{p}",
                        # BA-1: the server-persisted, content-addressed path --
                        # deliberately NOT the client identifier, to prove the
                        # controller maps back by source_identifier, not this.
                        "image_path": f"/data/uploads/ab/ab12{i}.jpg",
                        "imohash": f"ab12{i}",
                        "n_crops": 2,
                        "n_regions": 0,
                        "error": None,
                        "error_kind": None,
                        "source_identifier": p,
                    }
                )
        return {
            "status": "partial",
            "summary": {
                "successful": sum(1 for r in results if r["status"] == "success"),
                "duplicates": 0,
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "mismatches": 0,
                "missed_labels": 0,
                "unmatched_detections": 0,
                "labels_imported": 0,
                "crops_indexed": 2,
            },
            "results": results,
            "disagreements": [],
        }

    stub.on("POST", r"/ingest/path_lookup", lookup_handler)
    stub.on("POST", r"/ingest/upload", upload_handler)

    _select_folder(page, app_url)
    page.wait_for_selector("text=3 files selected")

    page.get_by_role("button", name="Start", exact=True).click()

    page.wait_for_selector("text=Failed (1)")

    # The known identifier (sub/a.jpg) was never sent to /ingest/upload.
    sent_paths = [p for call in upload_calls for p in call.image_paths]
    assert not any(p.endswith("sub/a.jpg") for p in sent_paths)
    assert any(p.endswith("a.jpg") and not p.endswith("sub/a.jpg") for p in sent_paths)
    assert any(p.endswith("b.jpg") for p in sent_paths)

    assert "decode failed" in page.locator("body").inner_text()
    assert "skipped 1" in page.locator("body").inner_text()

    # BA-7: the Failed tab groups/filters by the served error_kind.
    page.wait_for_selector("text=decode_failed (1)")
    page.get_by_role("button", name="decode_failed (1)", exact=True).click()
    assert "decode failed" in page.locator("body").inner_text()


def test_503_pauses_with_detail(stub, page, app_url):
    _base_ingest_stubs(stub)
    stub.on("POST", r"/ingest/path_lookup", {"known_paths": {}})
    stub.on(
        "POST",
        r"/ingest/upload",
        (503, {"detail": "detector not configured"}, "application/json"),
    )

    _select_folder(page, app_url)
    page.wait_for_selector("text=3 files selected")
    page.get_by_role("button", name="Start", exact=True).click()

    page.wait_for_selector("text=detector not configured")
    assert page.get_by_role("button", name="Resume", exact=True).is_visible()


def test_nginx_style_413_html_stops_run(stub, page, app_url):
    _base_ingest_stubs(stub)
    stub.on("POST", r"/ingest/path_lookup", {"known_paths": {}})
    stub.on(
        "POST",
        r"/ingest/upload",
        (413, "<html><body>413 Request Entity Too Large</body></html>", "text/html"),
    )

    _select_folder(page, app_url)
    page.wait_for_selector("text=3 files selected")
    page.get_by_role("button", name="Start", exact=True).click()

    page.wait_for_selector("text=CROPWRIGHT_INGEST_MAX_REQUEST_MB")


def test_drain_gate(stub, page, app_url):
    _base_ingest_stubs(stub, drain_unfinished=4)
    stub.on("POST", r"/ingest/path_lookup", {"known_paths": {}})

    start_calls = []
    stub.on("POST", r"/pipeline/auto_label/start", lambda *_: start_calls.append(1) or IDLE_JOB)

    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    page.wait_for_selector("text=4")

    recluster_btn = page.get_by_role("button", name="Recluster now", exact=True)
    recluster_btn.wait_for()
    assert recluster_btn.is_disabled()

    # BA-3: total_unfinished reaching 0 is not enough on its own — the
    # gate reads the server's own `drained` verdict. Flip the drain stub
    # to zero counts but drained=False (just reached zero, still inside
    # the stable_polls window) and reload; the gate must stay blocked.
    stub.on(
        "GET",
        r"/ingest/region_drain(\?|$)",
        {
            "pending_detection": 0,
            "pending_verification": 0,
            "total_unfinished": 0,
            "drained": False,
            "stable_for_s": 1,
            "observed_at": "2026-09-25T00:00:01Z",
        },
    )
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    recluster_btn = page.get_by_role("button", name="Recluster now", exact=True)
    recluster_btn.wait_for()
    assert recluster_btn.is_disabled()
    assert "waiting for the served stability verdict" in page.locator("body").inner_text()

    # Now the server actually confirms drained=True — the gate lifts.
    stub.on(
        "GET",
        r"/ingest/region_drain(\?|$)",
        {
            "pending_detection": 0,
            "pending_verification": 0,
            "total_unfinished": 0,
            "drained": True,
            "stable_for_s": 30,
            "observed_at": "2026-09-25T00:00:30Z",
        },
    )
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')

    recluster_btn = page.get_by_role("button", name="Recluster now", exact=True)
    expect(recluster_btn).to_be_enabled()
    recluster_btn.click()
    page.get_by_role("dialog").get_by_role("button", name="Start", exact=True).click()
    assert start_calls


def test_server_path_batch_panel(stub, page, app_url):
    """Piece 11: the server-path panel is absent without served
    `batch.source_roots`, and renders/submits against `POST
    /ingest/batch` when the backend advertises at least one root."""
    _base_ingest_stubs(stub, batch_source_roots=["/data/archive"])
    stub.on("POST", r"/ingest/path_lookup", {"known_paths": {}})

    batch_calls = []

    def batch_handler(request, _match):
        parsed = json.loads(request.post_data)
        batch_calls.append(parsed)
        return {
            "status": "success",
            "summary": {
                "successful": 1,
                "duplicates": 0,
                "failed": 0,
                "mismatches": 0,
                "missed_labels": 0,
                "unmatched_detections": 0,
                "labels_imported": 0,
                "crops_indexed": 3,
            },
            "results": [
                {
                    "status": "success",
                    "image_id": "img-1",
                    "image_path": parsed["items"][0]["path"],
                    "imohash": "h",
                    "n_crops": 3,
                    "n_regions": 0,
                    "error": None,
                    "error_kind": None,
                    "source_identifier": None,
                }
            ],
            "disagreements": [],
        }

    stub.on("POST", r"/ingest/batch", batch_handler)

    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    page.wait_for_selector("text=Server-path ingest")
    assert "/data/archive" in page.locator("body").inner_text()

    page.locator("textarea").first.fill("/data/archive/img001.jpg")
    page.get_by_role("button", name="Ingest 1 path", exact=True).click()

    page.wait_for_selector("text=successful 1")
    assert batch_calls
    assert batch_calls[0]["items"][0]["path"] == "/data/archive/img001.jpg"


def test_server_path_batch_panel_absent_without_source_roots(stub, page, app_url):
    _base_ingest_stubs(stub)  # batch_source_roots defaults to []
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    page.wait_for_selector("text=Upload")
    assert "Server-path ingest" not in page.locator("body").inner_text()


def test_ingest_absent(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/ingest/status(\?|$)", (404, {"detail": "not found"}, "application/json"))

    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector("text=This backend does not provide ingest.")
    expect(page.locator('nav[aria-label="Primary"] a[href="/ingest"]')).to_have_count(0)

    ingest_calls = [c for c in stub.calls if "/ingest/" in c[1]]
    assert ingest_calls == []
    assert not any(
        "/ingest/" in path for _, path in stub.handled if not path.endswith("/ingest/status")
    )


def test_region_drain_shows_served_stall_reason(stub, page, app_url):
    """V-1 (d72cc63): a stalled drain renders the served reason verbatim."""
    reason = "segmenter seg_b unavailable since 2026-09-25T09:57:00Z"
    _base_ingest_stubs(
        stub,
        drain_unfinished=12,
        stall_reason=reason,
        region_dependencies=[
            {
                "role": "segmenter",
                "model": "seg_b",
                "ready": False,
                "detail": "not loaded",
                "unavailable_since": "2026-09-25T09:57:00Z",
            }
        ],
    )
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    expect(page.get_by_test_id("region-drain-stall-reason")).to_contain_text(reason)
    expect(page.get_by_test_id("region-drain-dependencies")).to_contain_text("seg_b")


def test_duplicate_only_reupload_reports_already_indexed(stub, page, app_url):
    """F-59: re-selecting files the backend already has sends no upload and
    says so, instead of the result panel disappearing."""
    _base_ingest_stubs(stub)

    def lookup_handler(request, _match):
        parsed = json.loads(request.post_data)
        return {"known_paths": {p: "img-known" for p in parsed["image_paths"]}}

    stub.on("POST", r"/ingest/path_lookup", lookup_handler)
    _select_folder(page, app_url)
    page.get_by_role("button", name="Start", exact=True).click()
    expect(page.get_by_test_id("ingest-all-skipped")).to_contain_text("3 already indexed")
    expect(page.get_by_text("Skipped (3)")).to_be_visible()
