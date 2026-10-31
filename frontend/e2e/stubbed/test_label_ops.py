"""Label operations (/label-ops): the served history list, the detail dialog,
and undo as dry run first, Confirm second. Nothing is written until Confirm.
Set LABEL_OPS_SHOTS=<dir> to also save full-page screenshots (1440 and 430 px).
"""

from __future__ import annotations

import json
import os
from urllib.parse import parse_qs, urlparse

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect

OPS = [
    {
        "op_id": "op-2",
        "kind": "label",
        "at": "2026-10-30T10:00:00+00:00",
        "finished_at": "2026-10-30T10:00:05+00:00",
        "class_name": "widget_a",
        "target_cluster_id": None,
        "target_kind": None,
        "label_source": "human",
        "target": "selection",
        "selected": 12,
        "count": 12,
        "conflicts": 0,
        "current": 12,
        "status": "written",
        "undone": False,
        "undoable": True,
        "last_undo": None,
    },
    {
        "op_id": "op-1",
        "kind": "move",
        "at": "2026-10-29T08:30:00+00:00",
        "finished_at": "2026-10-29T08:30:02+00:00",
        "class_name": "widget_b",
        "target_cluster_id": 7,
        "target_kind": "class",
        "label_source": "human",
        "target": "crop_ids",
        "selected": 3,
        "count": 3,
        "conflicts": 0,
        "current": 0,
        "status": "undone",
        "undone": True,
        "undoable": False,
        "last_undo": {
            "at": "2026-10-29T09:00:00+00:00",
            "restored": 3,
            "already_undone": 0,
            "kept_changed": 0,
            "kept_holdout": 0,
            "kept_stale_class": 0,
            "conflicts": 0,
        },
    },
]

DRY = {
    "op_id": "op-2",
    "dry_run": True,
    "status": "written",
    "members_known": True,
    "matched": 12,
    "restored": 9,
    "already_undone": 0,
    "kept_changed": 1,
    "kept_holdout": 1,
    "kept_stale_class": 0,
    "conflicts": 1,
    "conflict_ids": ["crop-conflict-1"],
}


def many_ops(n):
    """``n`` undoable operations, ids op-00 (oldest) upward, each with its own time and class."""
    return [
        {**OPS[0], "op_id": f"op-{i:02d}", "at": f"2026-10-{1 + i:02d}T10:00:00+00:00", "class_name": f"widget_{i:02d}"}
        for i in range(n)
    ]


def list_page(rows, request):
    """The served page: ``op_id`` descending, exclusive ``after``, ``next_after`` only when more remain."""
    q = parse_qs(urlparse(request.url).query)
    limit = int(q.get("limit", ["20"])[0])
    after = q.get("after", [None])[0]
    older = sorted((r for r in rows if after is None or r["op_id"] < after), key=lambda r: r["op_id"], reverse=True)
    page = older[:limit]
    more = len(older) > limit
    return {"items": page, "limit": limit, "retention": 200, "next_after": page[-1]["op_id"] if more else None}


def serve(stub, ops=None, queries=None):
    """Register the list and undo routes; return the recorded undo bodies.
    ``queries`` (a list) collects each list request's query string."""
    bodies: list[dict] = []
    rows = list(OPS if ops is None else ops)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})

    def listing(request, _m):
        if queries is not None:
            queries.append(urlparse(request.url).query)
        return list_page(rows, request)

    stub.on("GET", r"/label_ops(\?|$)", listing)

    def undo(request, _m):
        body = json.loads(request.post_data)
        bodies.append(body)
        if body["dry_run"]:
            return dict(DRY)
        rows[0] = {**rows[0], "status": "undone", "undone": True, "undoable": False, "current": 0}
        return {**DRY, "dry_run": False}

    stub.on("POST", r"/label_ops/op-2/undo$", undo)
    return bodies


def _shot(page, name):
    out = os.environ.get("LABEL_OPS_SHOTS")
    if out:
        os.makedirs(out, exist_ok=True)
        page.screenshot(path=f"{out}/{name}.png", full_page=True)


def _open_history(page, app_url):
    page.goto(f"{app_url}/p/default/label-ops")
    expect(page.get_by_test_id("label-op-row")).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("label-op-row").first).to_contain_text("UTC · by Human")
    expect(page.get_by_test_id("label-ops-end")).to_be_visible()
    if page.viewport_size["width"] >= 640:
        expect(page.get_by_test_id("breadcrumb")).to_contain_text("Label ops")


def test_history_list_dry_run_then_confirm(stub, page, app_url):
    bodies = serve(stub)
    for tag, size in (("1440", {"width": 1440, "height": 900}), ("430", {"width": 430, "height": 900})):
        page.set_viewport_size(size)
        _open_history(page, app_url)
        _shot(page, f"history-{tag}")

        page.get_by_role("button", name="Details for Label to widget_a at 2026-10-30 10:00:00 UTC").click()
        dialog = page.get_by_test_id("label-op-dialog")
        expect(dialog).to_be_visible(timeout=ACTION_TIMEOUT_MS)
        # Nothing is written, and no Confirm exists, before the dry run.
        expect(page.get_by_test_id("label-op-confirm")).to_have_count(0)
        page.get_by_test_id("label-op-undo").click()

        expect(page.get_by_test_id("label-op-result")).to_contain_text(
            "Dry run: nothing has been written", timeout=ACTION_TIMEOUT_MS
        )
        expect(page.get_by_test_id("label-op-restored")).to_contain_text("9")
        expect(page.get_by_test_id("label-op-kept_holdout")).to_contain_text("1")
        expect(page.get_by_test_id("label-op-conflict-ids")).to_contain_text("crop-conflict-1")
        assert bodies and all(b["dry_run"] is True for b in bodies)
        _shot(page, f"dry-run-{tag}")

        if tag == "430":
            break
        page.keyboard.press("Escape")
        expect(dialog).to_have_count(0)

    # At 430 px the dialog stays inside the viewport.
    box = page.get_by_test_id("label-op-dialog").locator("> div").nth(1).bounding_box()
    assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= 430

    page.get_by_test_id("label-op-confirm").click()
    expect(page.get_by_test_id("label-op-result")).to_contain_text(
        "Undo finished", timeout=ACTION_TIMEOUT_MS
    )
    assert bodies[-1] == {"dry_run": False}
    assert [b["dry_run"] for b in bodies].count(False) == 1
    page.get_by_role("button", name="Close").click()
    expect(page.get_by_test_id("label-op-status").first).to_have_text("Undone", timeout=ACTION_TIMEOUT_MS)


def test_back_from_dry_run_writes_nothing(stub, page, app_url):
    bodies = serve(stub)
    _open_history(page, app_url)
    page.get_by_role("button", name="Details for Label to widget_a at 2026-10-30 10:00:00 UTC").click()
    page.get_by_test_id("label-op-undo").click()
    expect(page.get_by_test_id("label-op-confirm")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_role("button", name="Back").click()
    expect(page.get_by_test_id("label-op-confirm")).to_have_count(0)
    assert [b["dry_run"] for b in bodies] == [True]


def test_already_undone_operation_offers_no_undo(stub, page, app_url):
    serve(stub)
    _open_history(page, app_url)
    page.get_by_role("button", name="Details for Move to widget_b at 2026-10-29 08:30:00 UTC").click()
    expect(page.get_by_test_id("label-op-last-undo")).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("label-op-undo")).to_have_count(0)


def test_served_refusal_is_shown_verbatim(stub, page, app_url):
    serve(stub)
    stub.on(
        "POST",
        r"/label_ops/op-2/undo$",
        (409, {"detail": {"error": "label_op_busy", "message": "another undo is running"}}),
    )
    _open_history(page, app_url)
    page.get_by_role("button", name="Details for Label to widget_a at 2026-10-30 10:00:00 UTC").click()
    page.get_by_test_id("label-op-undo").click()
    expect(page.get_by_test_id("label-op-undo-error")).to_have_text(
        "another undo is running", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("label-op-confirm")).to_have_count(0)


def test_empty_and_error_states(stub, page, app_url):
    serve(stub, ops=[])
    page.goto(f"{app_url}/p/default/label-ops")
    expect(page.get_by_test_id("label-ops-empty")).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    stub.on("GET", r"/label_ops(\?|$)", (500, {"detail": "index unavailable"}))
    page.get_by_role("button", name="Refresh").click()
    expect(page.get_by_test_id("label-ops-error")).to_contain_text(
        "index unavailable", timeout=ACTION_TIMEOUT_MS
    )


def test_load_older_pages_to_the_end(stub, page, app_url):
    queries: list[str] = []
    serve(stub, ops=many_ops(25), queries=queries)
    for tag, size in (("1440", {"width": 1440, "height": 900}), ("430", {"width": 430, "height": 900})):
        queries.clear()
        page.set_viewport_size(size)
        page.goto(f"{app_url}/p/default/label-ops")
        rows = page.get_by_test_id("label-op-row")
        expect(rows).to_have_count(20, timeout=ACTION_TIMEOUT_MS)
        expect(page.get_by_test_id("label-ops-end")).to_have_count(0)
        page.get_by_test_id("label-ops-older").scroll_into_view_if_needed()
        _shot(page, f"older-before-{tag}")

        page.get_by_test_id("label-ops-older").click()
        expect(rows).to_have_count(25, timeout=ACTION_TIMEOUT_MS)
        assert "after=op-05" in queries[-1]
        expect(page.get_by_test_id("label-ops-older")).to_have_count(0)
        end = page.get_by_test_id("label-ops-end")
        expect(end).to_be_visible()
        end.scroll_into_view_if_needed()
        box = end.bounding_box()
        assert box is not None and box["x"] >= 0 and box["x"] + box["width"] <= size["width"]
        _shot(page, f"older-{tag}")


def test_load_older_failure_is_shown_and_retries(stub, page, app_url):
    rows = many_ops(25)
    serve(stub, ops=rows)
    page.goto(f"{app_url}/p/default/label-ops")
    expect(page.get_by_test_id("label-op-row")).to_have_count(20, timeout=ACTION_TIMEOUT_MS)

    stub.on("GET", r"/label_ops(\?|$)", (500, {"detail": "index unavailable"}))
    page.get_by_test_id("label-ops-older").click()
    expect(page.get_by_test_id("label-ops-older-error")).to_have_text(
        "index unavailable", timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id("label-op-row")).to_have_count(20)

    stub.on("GET", r"/label_ops(\?|$)", lambda request, _m: list_page(rows, request))
    page.get_by_test_id("label-ops-older").click()
    expect(page.get_by_test_id("label-op-row")).to_have_count(25, timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("label-ops-older-error")).to_have_count(0)
