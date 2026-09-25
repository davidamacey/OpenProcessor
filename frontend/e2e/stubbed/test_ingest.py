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


def _base_ingest_stubs(stub, *, status_total=0, drain_unfinished=0) -> None:
    # The root layout's classesStore.acquire() fires on every route.
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on(
        "GET",
        r"/ingest/status(\?|$)",
        {"total": status_total, "by_source": [], "by_day": []},
    )
    stub.on(
        "GET",
        r"/ingest/region_drain(\?|$)",
        {
            "pending_detection": drain_unfinished,
            "pending_verification": 0,
            "total_unfinished": drain_unfinished,
        },
    )
    stub.on("GET", r"/pipeline/auto_label/status", IDLE_JOB)


def _select_folder(page, app_url) -> None:
    page.goto(f"{app_url}/ingest")
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
        for p in parsed.image_paths:
            if p.endswith("b.jpg"):
                results.append(
                    {
                        "status": "failed",
                        "image_id": "",
                        "image_path": p,
                        "imohash": "",
                        "n_crops": 0,
                        "n_regions": 0,
                        "error": "decode failed",
                    }
                )
            else:
                results.append(
                    {
                        "status": "success",
                        "image_id": f"img-{p}",
                        "image_path": p,
                        "imohash": "h",
                        "n_crops": 2,
                        "n_regions": 0,
                        "error": None,
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

    page.goto(f"{app_url}/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')
    page.wait_for_selector("text=4")

    recluster_btn = page.get_by_role("button", name="Recluster now", exact=True)
    recluster_btn.wait_for()
    assert recluster_btn.is_disabled()

    # Flip the drain stub to 0 and reload the panel's next poll by
    # re-navigating (avoids waiting out the real 10s poll interval).
    stub.on(
        "GET",
        r"/ingest/region_drain(\?|$)",
        {"pending_detection": 0, "pending_verification": 0, "total_unfinished": 0},
    )
    page.goto(f"{app_url}/ingest")
    page.wait_for_selector('h1:has-text("Ingest")')

    recluster_btn = page.get_by_role("button", name="Recluster now", exact=True)
    expect(recluster_btn).to_be_enabled()
    recluster_btn.click()
    page.get_by_role("dialog").get_by_role("button", name="Start", exact=True).click()
    assert start_calls


def test_ingest_absent(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/ingest/status(\?|$)", (404, {"detail": "not found"}, "application/json"))

    page.goto(f"{app_url}/ingest")
    page.wait_for_selector("text=This backend does not provide ingest.")
    expect(page.locator('nav[aria-label="Primary"] a[href="/ingest"]')).to_have_count(0)

    ingest_calls = [c for c in stub.calls if "/ingest/" in c[1]]
    assert ingest_calls == []
    assert not any("/ingest/" in path for _, path in stub.handled if path != "/curation/ingest/status")
