#!/usr/bin/env python3
"""Mock-backed browser check for the /settings deployment-defaults page.

No live OpenProcessor backend exists that serves `GET,PUT {API_PREFIX}/
settings` yet (docs/design/curation-settings-ui-plan-2026-09-21.md §6.1) —
this drives the real /settings UI against a stubbed `/curation/*` backend
(Playwright route interception, same structure as
scripts/playwright_assist_scope.py) so the rendering + wire-composition
logic can be verified without one. Nothing here is claimed to have been
exercised against a running server; see the plan's §6.1/§8.3/§8.4.

Four passes:

  Pass 1 — the settings endpoint 404s (backend predates the feature). No
           controls at all.
  Pass 2 — today's real /methods shape (no assist axes). The two real
           controls render, and a save round-trip is observed verbatim.
  Pass 3 — /methods advertises the two advisory axes too. They render
           read-only; the control count does not change.
  Pass 4 — a PUT 422s. The server's own words render on the page, not a
           generic "failed" message, and the rejected value does not
           linger in the control.

Usage:  python3 scripts/playwright_curation_settings.py [base_url]
        DISPLAY=:11 python3 scripts/playwright_curation_settings.py --headed [base_url]
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from playwright.sync_api import sync_playwright

args = sys.argv[1:]
HEADED = "--headed" in args
args = [a for a in args if a != "--headed"]
BASE = args[0] if args else "http://localhost:5173"

# Today's real /curation/methods shape (with the server's per-entry
# `settable` flag) — no assist axes, one shadow sort to
# prove the status filter, no synthetic 'default' sentinel.
METHODS_TODAY = {
    "strategies": [
        {
            "id": "ivf",
            "axis": "cluster",
            "settable": True,
            "label": "FAISS IVF-512 (production)",
            "status": "stable",
            "default": True,
        },
        {
            "id": "recent",
            "axis": "sort",
            "settable": True,
            "label": "Recent first",
            "status": "stable",
            "default": True,
        },
        {
            "id": "uncertainty_entropy",
            "axis": "sort",
            "settable": True,
            "label": "Uncertainty margin",
            "status": "experimental",
        },
        {
            "id": "hdbscan_probe",
            "axis": "sort",
            "settable": True,
            "label": "HDBSCAN probe sort",
            "status": "shadow",
        },
    ],
    "flags": {},
}

# Same payload plus both assist axes, matching what wt-oss-hardening
# actually emits (plan §1.3.2).
METHODS_WITH_ASSIST_AXES = {
    "strategies": [
        *METHODS_TODAY["strategies"],
        {
            "id": "grounding_v2",
            "axis": "detection_profile",
            "settable": False,
            "label": "grounding_v2",
            "status": "stable",
            "default": True,
        },
        {
            "id": "warehouse_v1",
            "axis": "prompt_pack",
            "settable": True,
            "label": "warehouse_v1",
            "status": "stable",
            "default": True,
        },
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

CLASSES: list[dict[str, Any]] = []


class Stub:
    """Canned /curation/* responses + a record of every PUT /settings call."""

    def __init__(
        self,
        methods_body: dict[str, Any],
        settings_get_body: dict[str, Any] | None,
        settings_get_status: int = 200,
        put_response: dict[str, Any] | None = None,
        put_status: int = 200,
    ) -> None:
        self.methods_body = methods_body
        self.settings_get_body = settings_get_body
        self.settings_get_status = settings_get_status
        self.put_response = put_response if put_response is not None else SETTINGS_MERGED
        self.put_status = put_status
        self.put_calls: list[tuple[str, str]] = []  # (url, body)

    def handle(self, route, request) -> None:
        url = request.url
        path = re.sub(r"^https?://[^/]+", "", url)
        method = request.method

        def ok(payload: Any, status: int = 200) -> None:
            route.fulfill(
                status=status,
                content_type="application/json",
                body=json.dumps(payload),
            )

        if "/curation/health" in path:
            return ok({"status": "ok"})
        if path.startswith("/curation/classes"):
            return ok(CLASSES)
        if path.startswith("/curation/methods"):
            return ok(self.methods_body)
        if path.startswith("/curation/settings"):
            if method == "PUT":
                self.put_calls.append((url, request.post_data or ""))
                return ok(self.put_response, self.put_status)
            # GET
            if self.settings_get_status != 200:
                return ok(
                    {"detail": "not found"} if self.settings_get_status == 404 else {},
                    self.settings_get_status,
                )
            return ok(self.settings_get_body)
        return ok({})


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{'' if cond else ' — ' + detail}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not HEADED)
        page = browser.new_page()
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        # ================================================================
        # Pass 1 — feature absent (404).
        # ================================================================
        print("\nPass 1 — feature absent (GET /settings 404s)")
        stub1 = Stub(METHODS_TODAY, None, settings_get_status=404)
        page.route("**/curation/**", stub1.handle)

        page.goto(f"{BASE}/settings", wait_until="networkidle")
        page.wait_for_timeout(500)

        errors = [c for c in console if c.startswith("pageerror")]
        check("page renders with no pageerror", not errors, str(errors[:3]))
        check(
            "the 'does not support shared curation defaults' note is present",
            page.get_by_text(re.compile(r"does not support shared curation defaults")).count()
            > 0,
        )
        check("zero <select> elements on the page", page.locator("select").count() == 0)
        check(
            "zero Save buttons",
            page.get_by_role("button", name="Save").count() == 0,
        )

        page.unroute("**/curation/**")

        # ================================================================
        # Pass 2 — the two real controls, and a save round-trip.
        # ================================================================
        print("\nPass 2 — the two real controls, and a save round-trip")
        stub2 = Stub(METHODS_TODAY, SETTINGS_EMPTY, put_response=SETTINGS_MERGED)
        page.route("**/curation/**", stub2.handle)

        console.clear()
        page.goto(f"{BASE}/settings", wait_until="networkidle")
        page.wait_for_timeout(500)

        selects = page.locator("select")
        check("exactly 2 <select> elements", selects.count() == 2, str(selects.count()))
        check(
            "labels are 'Clustering method' and 'Review queue sort'",
            page.get_by_text("Clustering method").count() > 0
            and page.get_by_text("Review queue sort").count() > 0,
        )
        options_text = page.locator("select option").all_inner_texts()
        check(
            "the shadow sort is not among the options",
            "HDBSCAN probe sort" not in " ".join(options_text),
            str(options_text),
        )
        check(
            "the read-only startup-config heading is absent",
            page.get_by_text("Set by the backend's startup config").count() == 0,
        )

        # Select the sort axis's non-default option, then Save.
        sort_select = selects.nth(1)
        sort_select.select_option("uncertainty_entropy")
        page.wait_for_timeout(150)
        save_buttons = page.get_by_role("button", name="Save")
        check("a Save button is enabled after changing the sort", save_buttons.count() > 0)
        save_buttons.nth(1).click()
        page.wait_for_timeout(200)

        dialog = page.get_by_role("dialog")
        check("confirm dialog appears", dialog.count() > 0)
        check(
            "dialog names the sort axis",
            dialog.get_by_text("Review queue sort").count() > 0,
        )
        check(
            "dialog carries no irreversibility warning (the sort pin is clearable now)",
            dialog.get_by_text(re.compile(r"no way to clear a shared sort default")).count()
            == 0,
        )

        stub2.put_calls.clear()
        dialog.get_by_role("button", name="Confirm").click()
        page.wait_for_timeout(300)

        check("exactly one PUT recorded", len(stub2.put_calls) == 1, str(stub2.put_calls))
        if stub2.put_calls:
            _url, body = stub2.put_calls[0]
            print(f"    recorded PUT body: {body!r}")
            check(
                "PUT body is {'defaults': {'sort': 'uncertainty_entropy'}} verbatim"
                " (single axis, no cluster key)",
                # JS's JSON.stringify (what the client actually sends) has no
                # separator whitespace, unlike Python's json.dumps default.
                body == '{"defaults":{"sort":"uncertainty_entropy"}}',
                body,
            )

        check(
            "the cluster axis now shows pinned, though the client never sent it"
            " (adopted the response, not a guess)",
            page.get_by_text("pinned").count() >= 1,
        )

        page.unroute("**/curation/**")

        # ================================================================
        # Pass 3 — non-settable axis visible but inert.
        # ================================================================
        print("\nPass 3 — non-settable axis visible but inert")
        stub3 = Stub(METHODS_WITH_ASSIST_AXES, SETTINGS_EMPTY)
        page.route("**/curation/**", stub3.handle)

        console.clear()
        page.goto(f"{BASE}/settings", wait_until="networkidle")
        page.wait_for_timeout(500)

        check(
            "the read-only startup-config section is present",
            page.get_by_text("Set by the backend's startup config").count() > 0,
        )
        check("names grounding_v2", page.get_by_text("grounding_v2").count() > 0)
        check("names warehouse_v1", page.get_by_text("warehouse_v1").count() > 0)
        check(
            "exactly 3 <select>s (cluster, sort, settable prompt_pack) — the"
            " non-settable detection_profile adds no control",
            page.locator("select").count() == 3,
            str(page.locator("select").count()),
        )
        check(
            "startup-config wording is on screen",
            page.get_by_text(re.compile(r"chosen by the backend's startup config")).count() > 0,
        )
        advisory_heading = page.get_by_text("Set by the backend's startup config")
        advisory_section = advisory_heading.locator("xpath=ancestor::section[1]")
        check(
            "no Save button exists inside the advisory section",
            advisory_section.get_by_role("button", name="Save").count() == 0,
        )

        page.unroute("**/curation/**")

        # ================================================================
        # Pass 4 — 422 surfaces the server's own words.
        # ================================================================
        print("\nPass 4 — 422 surfaces the server's own words")
        stub4 = Stub(
            METHODS_TODAY,
            SETTINGS_EMPTY,
            put_response=SETTINGS_422,
            put_status=422,
        )
        page.route("**/curation/**", stub4.handle)

        console.clear()
        page.goto(f"{BASE}/settings", wait_until="networkidle")
        page.wait_for_timeout(500)

        sort_select4 = page.locator("select").nth(1)
        sort_select4.select_option("uncertainty_entropy")
        page.wait_for_timeout(150)
        page.get_by_role("button", name="Save").nth(1).click()
        page.wait_for_timeout(200)
        page.get_by_role("dialog").get_by_role("button", name="Confirm").click()
        page.wait_for_timeout(400)

        check(
            "the literal substring \"is not a currently-advertised id for axis 'sort'\" is displayed",
            page.get_by_text(
                re.compile(r"is not a currently-advertised id for axis 'sort'")
            ).count()
            > 0,
        )
        check(
            "'valid ids: ['recent']' is visible",
            page.get_by_text(re.compile(r"valid ids:\s*\['recent'\]")).count() > 0,
        )
        check(
            "the control's value is not left showing the rejected selection as though saved",
            page.locator("select").nth(1).input_value() != "uncertainty_entropy",
            page.locator("select").nth(1).input_value(),
        )

        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
