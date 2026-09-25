"""Ported from scripts/playwright_curation_settings.py (deleted; see
docs/design/test-audit-2026-09-24.md recommendation 5 / P1-1).

Four passes against `/settings`:

  Pass 1 — the settings endpoint 404s (backend predates the feature). No
           controls at all.
  Pass 2 — today's real /methods shape (no assist axes). The two real
           controls render, and a save round-trip is observed verbatim.
  Pass 3 — /methods advertises the two advisory axes too. They render
           read-only; the control count does not change.
  Pass 4 — a PUT 422s. The server's own words render on the page, and the
           rejected value does not linger in the control.
"""

from __future__ import annotations

import re

METHODS_TODAY = {
    "strategies": [
        {"id": "ivf", "axis": "cluster", "settable": True, "label": "FAISS IVF-512 (production)", "status": "stable", "default": True},
        {"id": "recent", "axis": "sort", "settable": True, "label": "Recent first", "status": "stable", "default": True},
        {"id": "uncertainty_entropy", "axis": "sort", "settable": True, "label": "Uncertainty margin", "status": "experimental"},
        {"id": "hdbscan_probe", "axis": "sort", "settable": True, "label": "HDBSCAN probe sort", "status": "shadow"},
    ],
    "flags": {},
}

METHODS_WITH_ASSIST_AXES = {
    "strategies": [
        *METHODS_TODAY["strategies"],
        {"id": "grounding_v2", "axis": "detection_profile", "settable": False, "label": "grounding_v2", "status": "stable", "default": True},
        {"id": "warehouse_v1", "axis": "prompt_pack", "settable": True, "label": "warehouse_v1", "status": "stable", "default": True},
    ],
    "flags": {},
}

SETTINGS_EMPTY = {"defaults": {}, "updated_at": None, "updated_by": None}

# Deliberately contains 'cluster', which pass 2's PUT never sends, so
# adopting-the-response (not an optimistic guess) is observable in the DOM.
SETTINGS_MERGED = {
    "defaults": {"cluster": "ivf", "sort": "uncertainty_entropy"},
    "updated_at": "2026-09-21T00:00:00+00:00",
    "updated_by": None,
}

SETTINGS_422 = {
    "detail": "'uncertainty_entropy' is not a currently-advertised id for axis 'sort'; "
    "valid ids: ['recent']"
}


def register(stub, methods_body, settings_get_status, settings_get_body, put_response=None, put_status=200):
    put_calls: list[tuple[str, str]] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/methods(\?|$)", methods_body)
    # G10 "Curation scores" card: this suite doesn't exercise it (see
    # test_scores_card.py), so a plain 404 keeps it absent and out of the
    # way of every assertion here — same "backend predates the feature"
    # path the card itself degrades to.
    stub.on("GET", r"/scores/coverage(\?|$)", (404, {"detail": "not found"}))

    def settings_get(_request, _match):
        if settings_get_status != 200:
            body = {"detail": "not found"} if settings_get_status == 404 else {}
            return (settings_get_status, body)
        return (200, settings_get_body)

    def settings_put(request, _match):
        put_calls.append((request.url, request.post_data or ""))
        return (put_status, put_response if put_response is not None else SETTINGS_MERGED)

    stub.on("GET", r"/settings(\?|$)", settings_get)
    stub.on("PUT", r"/settings(\?|$)", settings_put)
    return put_calls


def test_curation_settings(stub, page, app_url):
    # ================================================================
    # Pass 1 — feature absent (404).
    # ================================================================
    register(stub, METHODS_TODAY, 404, None)

    page.goto(f"{app_url}/settings")
    page.get_by_text(re.compile(r"does not support shared curation defaults")).first.wait_for(timeout=15000)
    page.wait_for_timeout(200)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")], "page should render with no pageerror"
    assert page.locator("select").count() == 0, "zero <select> elements expected"
    assert page.get_by_role("button", name="Save").count() == 0, "zero Save buttons expected"

    # ================================================================
    # Pass 2 — the two real controls, and a save round-trip.
    # ================================================================
    stub.console_errors.clear()
    put_calls = register(stub, METHODS_TODAY, 200, SETTINGS_EMPTY, put_response=SETTINGS_MERGED)

    page.goto(f"{app_url}/settings")
    page.get_by_text("Clustering method").first.wait_for(timeout=15000)
    page.wait_for_timeout(200)

    selects = page.locator("select")
    assert selects.count() == 2, f"exactly 2 <select> elements expected, got {selects.count()}"
    assert page.get_by_text("Clustering method").count() > 0 and page.get_by_text("Review queue sort").count() > 0

    options_text = page.locator("select option").all_inner_texts()
    assert "HDBSCAN probe sort" not in " ".join(options_text), "the shadow sort must not be among the options"
    assert page.get_by_text("Set by the backend's startup config").count() == 0

    sort_select = selects.nth(1)
    sort_select.select_option("uncertainty_entropy")
    page.wait_for_timeout(150)
    save_buttons = page.get_by_role("button", name="Save")
    assert save_buttons.count() > 0, "a Save button should be enabled after changing the sort"
    save_buttons.nth(1).click()
    page.wait_for_timeout(200)

    dialog = page.get_by_role("dialog")
    assert dialog.count() > 0, "confirm dialog should appear"
    assert dialog.get_by_text("Review queue sort").count() > 0, "dialog should name the sort axis"

    put_calls.clear()
    dialog.get_by_role("button", name="Confirm").click()
    page.wait_for_timeout(300)

    assert len(put_calls) == 1, f"exactly one PUT expected: {put_calls}"
    _url, body = put_calls[0]
    # JS's JSON.stringify (what the client actually sends) has no separator
    # whitespace, unlike Python's json.dumps default.
    assert body == '{"defaults":{"sort":"uncertainty_entropy"}}', body

    assert page.get_by_text("pinned").count() >= 1, (
        "the cluster axis should now show pinned, though the client never sent it "
        "(adopted the response, not a guess)"
    )

    # ================================================================
    # Pass 3 — non-settable axis visible but inert.
    # ================================================================
    stub.console_errors.clear()
    register(stub, METHODS_WITH_ASSIST_AXES, 200, SETTINGS_EMPTY)

    page.goto(f"{app_url}/settings")
    page.get_by_text("Set by the backend's startup config").first.wait_for(timeout=15000)
    page.wait_for_timeout(200)

    assert page.get_by_text("grounding_v2").count() > 0
    assert page.get_by_text("warehouse_v1").count() > 0
    assert page.locator("select").count() == 3, (
        "exactly 3 <select>s expected (cluster, sort, settable prompt_pack) — the "
        f"non-settable detection_profile should add no control, got {page.locator('select').count()}"
    )
    advisory_heading = page.get_by_text("Set by the backend's startup config")
    advisory_section = advisory_heading.locator("xpath=ancestor::section[1]")
    assert advisory_section.get_by_role("button", name="Save").count() == 0

    # ================================================================
    # Pass 4 — 422 surfaces the server's own words.
    # ================================================================
    stub.console_errors.clear()
    register(stub, METHODS_TODAY, 200, SETTINGS_EMPTY, put_response=SETTINGS_422, put_status=422)

    page.goto(f"{app_url}/settings")
    page.get_by_text("Clustering method").first.wait_for(timeout=15000)
    page.wait_for_timeout(200)

    sort_select4 = page.locator("select").nth(1)
    sort_select4.select_option("uncertainty_entropy")
    page.wait_for_timeout(150)
    page.get_by_role("button", name="Save").nth(1).click()
    page.wait_for_timeout(200)
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()
    page.wait_for_timeout(400)

    assert page.get_by_text(re.compile(r"is not a currently-advertised id for axis 'sort'")).count() > 0
    assert page.get_by_text(re.compile(r"valid ids:\s*\['recent'\]")).count() > 0
    assert page.locator("select").nth(1).input_value() != "uncertainty_entropy", (
        "the control's value must not be left showing the rejected selection as though saved"
    )
