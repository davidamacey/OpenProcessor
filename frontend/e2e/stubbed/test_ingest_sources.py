"""Stubbed e2e for the Sources panel on `/ingest`: the list is rendered as
served, a source is created with its triggers, run now sends the right call,
a blocked source shows the served reason, and the page stays inside 430 px.
"""

from __future__ import annotations

import json

from conftest import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

from test_ingest import _base_ingest_stubs

SRC = "src_1a2b3c4d5e"
BASE = "/curation/projects/default/ingest/sources"


def _source(**over):
    row = {
        "project": "default", "source_id": SRC, "name": "Incoming photos", "backend": "s3",
        "location": "s3://photos/incoming/", "s3": {"bucket": "photos", "prefix": "incoming/"},
        "folder": None, "extensions": [".jpg"], "source": "source", "batch_size": 16,
        "concurrency": 16,
        "triggers": {"watch": {"enabled": True, "interval_s": 60},
                     "schedule": {"enabled": True, "cron": "0 2 * * *"}, "webhook_enabled": True},
        "created_at": "2026-10-12T10:00:00+00:00",
        "counts": {"pending": 3, "done": 120, "failed": 1, "dead": 2},
        "scan": {"active": False, "pages": 0, "listed": 0, "queued": 0, "last": None},
        "watch": {"last_poll_at": "2026-10-12T11:00:00+00:00",
                  "next_poll_at": "2026-10-12T11:01:00+00:00", "error": None, "backoff_until": None},
        "schedule": {"last_fire_at": None, "next_fire_at": "2026-10-13T02:00:00+00:00"},
        "webhook": {"path": f"{BASE}/{SRC}/events", "received": 0, "queued": 0,
                    "duplicates": 0, "ignored": 0, "last_event_at": None},
        "last_run": None, "errors": [], "can_run": True, "can_retry_dead": True,
        "blocked_reason": None, "blocked_message": None,
    }
    row.update(over)
    return row


def _list(items):
    return {"items": items, "s3_configured": True, "s3_allowed": ["photos/incoming/"],
            "default_batch_size": 16, "default_concurrency": 16,
            "default_watch_interval_s": 60, "min_watch_interval_s": 10, "poll_after_s": 5}


def _open(stub, page, app_url, items):
    _base_ingest_stubs(stub)
    stub.on("GET", r"/ingest/bulk(\?|$)", {"items": [], "total": 0, "page": 1, "page_size": 5})
    state = {"items": items}
    stub.on("GET", r"/ingest/sources(\?|$)", lambda _r, _m: _list(state["items"]))
    page.goto(f"{app_url}/p/default/ingest")
    page.wait_for_selector("text=Sources (S3 buckets and watched folders)")
    return state


def test_create_run_and_blocked_source_at_430px(stub, page, app_url):
    page.set_viewport_size({"width": 430, "height": 900})
    state = _open(stub, page, app_url, [])
    expect(page.get_by_test_id("sources-empty")).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    created = []

    def create(request, _m):
        created.append(json.loads(request.post_data))
        state["items"] = [_source()]
        return (201, _source())

    stub.on("POST", r"/ingest/sources$", create)
    page.get_by_test_id("source-add").click()
    page.get_by_test_id("source-name").fill("Incoming photos")
    page.get_by_test_id("source-bucket").fill("photos")
    page.get_by_test_id("source-prefix").fill("incoming/")
    page.get_by_test_id("trigger-watch").check()
    page.get_by_test_id("trigger-webhook").check()
    page.get_by_test_id("source-save").click()

    card = page.get_by_test_id("source-card")
    expect(card).to_contain_text("s3://photos/incoming/", timeout=ACTION_TIMEOUT_MS)
    body = created[0]
    assert body["s3"] == {"bucket": "photos", "prefix": "incoming/"}
    assert body["triggers"]["webhook_enabled"] is True
    assert body["triggers"]["watch"]["enabled"] is True
    assert "secret" not in json.dumps(body).lower()

    runs = []
    stub.on("POST", rf"{BASE}/{SRC}/run$", lambda _r, _m: runs.append(1) or (202, {"run_id": "bulk_1"}))
    page.get_by_test_id("source-run").click()
    expect(page.get_by_test_id("sources-notice")).to_contain_text("bulk_1", timeout=ACTION_TIMEOUT_MS)
    assert runs

    # The backend's verdict is shown verbatim and the button follows can_run.
    state["items"] = [_source(can_run=False, blocked_reason="project_run_live",
                              blocked_message="Another ingest run of this project is live")]
    expect(page.get_by_test_id("source-blocked")).to_have_text(
        "Another ingest run of this project is live", timeout=15000)
    expect(page.get_by_test_id("source-run")).to_be_disabled()

    width = page.evaluate(
        "() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})"
    )
    assert width["sw"] <= width["cw"], width
