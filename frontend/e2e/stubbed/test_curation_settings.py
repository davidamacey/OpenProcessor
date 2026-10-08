"""Ported from scripts/playwright_curation_settings.py (deleted; see
docs/design/test-audit-2026-09-24.md recommendation 5 / P1-1).

Four passes against `/settings`:

  Pass 1 — the settings endpoint fails. The error and a Retry button
           render; no controls at all.
  Pass 2 — today's real /methods shape (no assist axes). The two real
           controls render, and a save round-trip is observed verbatim.
  Pass 3 — /methods advertises the two advisory axes too. They render
           read-only; the control count does not change.
  Pass 4 — a PUT 422s. The server's own words render on the page, and the
           rejected value does not linger in the control.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

import re

from playwright.sync_api import expect

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
    # test_scores_card.py); an empty coverage map and an idle job keep it
    # out of the way of every assertion here.
    stub.on("GET", r"/scores/coverage(\?|$)", {"coverage": {}})
    stub.on("GET", r"/scores/status(\?|$)", {"status": "idle"})

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
    # Pass 1 — the settings read fails.
    # ================================================================
    register(stub, METHODS_TODAY, 500, None)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_role("button", name="Retry").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)

    assert not [c for c in stub.console_errors if c.startswith("pageerror")], "page should render with no pageerror"
    assert page.locator("select").count() == 0, "zero <select> elements expected"
    assert page.get_by_role("button", name="Save").count() == 0, "zero Save buttons expected"

    # ================================================================
    # Pass 2 — the two real controls, and a save round-trip.
    # ================================================================
    stub.console_errors.clear()
    put_calls = register(stub, METHODS_TODAY, 200, SETTINGS_EMPTY, put_response=SETTINGS_MERGED)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Clustering method").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    selects = page.locator("select")
    expect(selects).to_have_count(2, timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("Clustering method").count() > 0 and page.get_by_text("Review queue sort").count() > 0

    options_text = page.locator("select option").all_inner_texts()
    assert "HDBSCAN probe sort" not in " ".join(options_text), "the shadow sort must not be among the options"
    assert page.get_by_text("Not settable on this backend").count() == 0

    sort_select = selects.nth(1)
    sort_select.select_option("uncertainty_entropy")
    save_buttons = page.get_by_role("button", name="Save")
    expect(save_buttons.nth(1)).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    save_buttons.nth(1).click()

    dialog = page.get_by_role("dialog")
    expect(dialog).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert dialog.get_by_text("Review queue sort").count() > 0, "dialog should name the sort axis"

    put_calls.clear()
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith("/settings"), timeout=ACTION_TIMEOUT_MS
    ):
        dialog.get_by_role("button", name="Confirm").click()

    assert len(put_calls) == 1, f"exactly one PUT expected: {put_calls}"
    _url, body = put_calls[0]
    # JS's JSON.stringify (what the client actually sends) has no separator
    # whitespace, unlike Python's json.dumps default.
    assert body == '{"defaults":{"sort":"uncertainty_entropy"}}', body

    expect(page.get_by_text("pinned").first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # ================================================================
    # Pass 3 — non-settable axis visible but inert.
    # ================================================================
    stub.console_errors.clear()
    register(stub, METHODS_WITH_ASSIST_AXES, 200, SETTINGS_EMPTY)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Not settable on this backend").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    assert page.get_by_text("grounding_v2").count() > 0
    assert page.get_by_text("warehouse_v1").count() > 0
    expect(page.locator("select")).to_have_count(3, timeout=ACTION_TIMEOUT_MS)
    advisory_heading = page.get_by_text("Not settable on this backend")
    advisory_section = advisory_heading.locator("xpath=ancestor::section[1]")
    assert advisory_section.get_by_role("button", name="Save").count() == 0

    # ================================================================
    # Pass 3b — detection_profile is activation-backed and settable; a
    # served `off` (explicit deactivation) is shown, not swallowed.
    # ================================================================
    stub.console_errors.clear()
    methods_settable = {
        "strategies": [
            *METHODS_TODAY["strategies"],
            {"id": "grounding_v2", "axis": "detection_profile", "settable": True, "label": "grounding_v2", "status": "stable"},
        ],
        "flags": {},
    }
    register(
        stub,
        methods_settable,
        200,
        {"defaults": {"detection_profile": "off"}, "updated_at": None, "updated_by": None},
    )
    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Detection profile").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text("Not settable on this backend").count() == 0
    expect(page.locator("select option", has_text="off (served")).to_have_count(1, timeout=ACTION_TIMEOUT_MS)

    # ================================================================
    # Pass 4 — 422 surfaces the server's own words.
    # ================================================================
    stub.console_errors.clear()
    register(stub, METHODS_TODAY, 200, SETTINGS_EMPTY, put_response=SETTINGS_422, put_status=422)

    page.goto(f"{app_url}/p/default/settings")
    page.get_by_text("Clustering method").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    sort_select4 = page.locator("select").nth(1)
    sort_select4.select_option("uncertainty_entropy")
    save_btn4 = page.get_by_role("button", name="Save").nth(1)
    expect(save_btn4).to_be_enabled(timeout=ACTION_TIMEOUT_MS)
    save_btn4.click()
    confirm_dialog4 = page.get_by_role("dialog")
    expect(confirm_dialog4).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == "PUT" and r.url.endswith("/settings"), timeout=ACTION_TIMEOUT_MS
    ):
        confirm_dialog4.get_by_role("button", name="Confirm").click()

    error_text = page.get_by_text(re.compile(r"is not a currently-advertised id for axis 'sort'"))
    expect(error_text.first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert page.get_by_text(re.compile(r"valid ids:\s*\['recent'\]")).count() > 0
    assert page.locator("select").nth(1).input_value() != "uncertainty_entropy", (
        "the control's value must not be left showing the rejected selection as though saved"
    )
