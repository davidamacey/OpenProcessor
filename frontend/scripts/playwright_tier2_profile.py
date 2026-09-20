#!/usr/bin/env python3
"""Mock-backed browser check for a JSON-configured tier-2 annotation slot.

No live deployment ships `static/annotation-profiles.json` — the whole
point of tier 2 (docs/design/tier2-annotation-profile-config-plan-2026-
09-20.md) is that ABSENT is the normal case and must be silent. This
drives the real `/review` UI against a stubbed `/curation/*` backend AND a
stubbed `/annotation-profiles.json` route (Playwright route interception,
same structure as scripts/playwright_assist_scope.py) across three
passes:

  Pass 1 — no deployment profile (route answers 200/text-html, exactly
           nginx's `try_files` SPA fallback for a file that doesn't
           exist). The tab strip must show only the 5 core tabs plus
           Plates — no "Pallet labels" tab, no console error mentioning
           annotation-profiles.

  Pass 2 — the shipped `static/annotation-profiles.example.json` served
           with `content-type: application/json`. A "Pallet labels" tab
           appears, is clickable, drives a real `GET /curation/review/
           pallet_labels` request, and its text-filter input's label
           reads "SSCC".

  Pass 3 — a malformed profile (`{"version": 1, "slots": [{"key": "bad"}]}`).
           The page still renders, the Plates tab still works, and a
           toast mentioning the deployment annotation profile appears —
           proving "malformed must not crash the app" in a real browser,
           not just in vitest.

Usage:  python3 scripts/playwright_tier2_profile.py [base_url]
        DISPLAY=:11 python3 scripts/playwright_tier2_profile.py --headed [base_url] [--out DIR]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

args = sys.argv[1:]
HEADED = "--headed" in args
args = [a for a in args if a != "--headed"]
OUT_DIR: Path | None = None
if "--out" in args:
    i = args.index("--out")
    OUT_DIR = Path(args[i + 1])
    args = args[:i] + args[i + 2 :]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE = args[0] if args else "http://localhost:5173"

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PROFILE = json.loads(
    (REPO_ROOT / "static" / "annotation-profiles.example.json").read_text()
)
MALFORMED_PROFILE = {"version": 1, "slots": [{"key": "bad"}]}

CLASSES = [
    {
        "id": 1,
        "name": "license_plate",
        "group": "vehicle",
        "hotkey_letter": "l",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "wooden_pallet",
        "group": "warehouse",
        "hotkey_letter": "w",
        "count": 20,
        "validated_count": 5,
        "cluster_size": 22,
        "deprecated": False,
    },
]

METHODS = {"strategies": [], "flags": {}}

EMPTY_QUEUE = {"total": 0, "page": 1, "page_size": 30, "items": [], "sort_fallback_reason": None}

GIF_1PX = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)


class Stub:
    """Canned /curation/* + /annotation-profiles.json responses, and a record of
    every /curation/review/{id} request the page actually made."""

    def __init__(self, profile_status: int, profile_body: str | None, profile_type: str) -> None:
        self.profile_status = profile_status
        self.profile_body = profile_body
        self.profile_type = profile_type
        self.review_calls: list[str] = []

    def handle_profile(self, route, request) -> None:
        if self.profile_body is None:
            # Simulate nginx's/Vite's SPA fallback for a genuinely absent
            # file: HTTP 200, text/html, some index-ish body.
            route.fulfill(
                status=200,
                content_type="text/html",
                body="<!doctype html><html><body>spa fallback</body></html>",
            )
            return
        route.fulfill(
            status=self.profile_status,
            content_type=self.profile_type,
            body=self.profile_body,
        )

    def handle_kb(self, route, request) -> None:
        url = request.url
        path = re.sub(r"^https?://[^/]+", "", url)

        def ok(payload: Any, status: int = 200) -> None:
            route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))

        if "/curation/health" in path:
            return ok({"status": "ok"})
        if path.startswith("/curation/classes"):
            return ok(CLASSES)
        if path.startswith("/curation/methods"):
            return ok(METHODS)
        if path.startswith("/curation/review/"):
            self.review_calls.append(path)
            return ok(EMPTY_QUEUE)
        if "/thumbnail" in path or "/source" in path or "/region_thumbnail" in path:
            return route.fulfill(status=200, content_type="image/gif", body=GIF_1PX)
        return ok({})


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{'' if cond else ' — ' + detail}")
    if not cond:
        FAILURES.append(name)


def tab_labels(page) -> list[str]:
    # Wait for the tab strip to actually render (client-side SvelteKit
    # hydration + the layout's `load()` awaiting `loadDeploymentProfiles()`
    # both happen after `domcontentloaded`) before reading its contents.
    page.get_by_role("button", name="All", exact=True).wait_for(timeout=15000)
    return page.locator("div.border-b.border-zinc-800 button").all_inner_texts()


def safe_screenshot(page, path: Path) -> None:
    try:
        page.screenshot(path=str(path))
    except Exception as e:  # noqa: BLE001 — best-effort artifact, never fatal
        print(f"    (screenshot skipped: {e})")


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not HEADED)
        page = browser.new_page()
        page.set_default_timeout(30000)
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        # ================================================================
        # Pass 1 — no deployment profile (SPA-fallback text/html, HTTP 200).
        # ================================================================
        print("\nPass 1 — no annotation-profiles.json (SPA fallback)")
        stub1 = Stub(profile_status=200, profile_body=None, profile_type="text/html")
        page.route("**/annotation-profiles.json", stub1.handle_profile)
        page.route("**/curation/**", stub1.handle_kb)

        page.goto(f"{BASE}/review", wait_until="domcontentloaded")
        page.wait_for_timeout(1000)

        labels = tab_labels(page)
        check("tab strip has no 'Pallet labels' tab", "Pallet labels" not in labels, str(labels))
        check("tab strip DOES have 'Plates'", "Plates" in labels, str(labels))
        errors = [c for c in console if c.startswith("pageerror")]
        check("no pageerror on pass 1", not errors, str(errors[:3]))
        annotation_msgs = [c for c in console if "annotation-profiles" in c.lower()]
        check(
            "no console message mentions annotation-profiles (silent absent path)",
            not annotation_msgs,
            str(annotation_msgs),
        )
        if OUT_DIR:
            safe_screenshot(page, OUT_DIR / "pass1_no_profile.png")

        page.unroute("**/annotation-profiles.json")
        page.unroute("**/curation/**")

        # ================================================================
        # Pass 2 — the shipped example profile, served as real JSON.
        # ================================================================
        print("\nPass 2 — annotation-profiles.example.json served as application/json")
        stub2 = Stub(
            profile_status=200,
            profile_body=json.dumps(EXAMPLE_PROFILE),
            profile_type="application/json",
        )
        page.route("**/annotation-profiles.json", stub2.handle_profile)
        page.route("**/curation/**", stub2.handle_kb)

        page.goto(f"{BASE}/review", wait_until="domcontentloaded")
        page.wait_for_timeout(1000)

        labels = tab_labels(page)
        check("'Pallet labels' tab is present", "Pallet labels" in labels, str(labels))

        pallet_tab = page.get_by_role("button", name="Pallet labels")
        check("Pallet labels tab is clickable", pallet_tab.count() > 0)
        stub2.review_calls.clear()
        pallet_tab.first.click()
        page.wait_for_timeout(1000)
        check(
            "clicking it drives GET /curation/review/pallet_labels",
            any("/curation/review/pallet_labels" in c for c in stub2.review_calls),
            str(stub2.review_calls),
        )

        text_filter_label = page.get_by_text("SSCC", exact=True)
        check("the text-filter input's label reads 'SSCC'", text_filter_label.count() > 0)

        hint_strip_text = page.locator("span.hidden.text-\\[11px\\].text-zinc-500.md\\:inline")
        hint_text = hint_strip_text.first.inner_text() if hint_strip_text.count() > 0 else ""
        print(f"    hint strip text for the pallet_label tab: {hint_text!r}")
        check(
            "the hint strip shows an F (false-pos) and E (edit) binding",
            "F" in hint_text and "E" in hint_text,
            hint_text,
        )
        # KNOWN, PRE-EXISTING gap (not introduced by tier 2): the reject-key
        # hint in review/+page.svelte is hardcoded to the literal glyph "D"
        # regardless of what a slot's own keymap actually binds `reject` to
        # (buildSlotKeymap/the ACTUAL dispatcher IS generic — only this
        # display string isn't). pallet_label binds reject to "R", so the
        # hint strip is misleading for a second slot with a different
        # letter. Documented, not silently "fixed" here — out of tier 2's
        # scope (a hint-string display bug, not a config-loading one).
        print(
            "    NOTE: hint strip shows 'D reject' even though pallet_label's own "
            "keymap binds reject to 'R' — pre-existing, non-generic hint string "
            "in review/+page.svelte, out of tier-2 scope. Actual key dispatch is "
            "correct (buildSlotKeymap reads the real keymap); only this display "
            "hint is stale."
        )

        errors = [c for c in console if c.startswith("pageerror")]
        check("no pageerror on pass 2", not errors, str(errors[:3]))
        if OUT_DIR:
            safe_screenshot(page, OUT_DIR / "pass2_profile_served.png")

        page.unroute("**/annotation-profiles.json")
        page.unroute("**/curation/**")

        # ================================================================
        # Pass 3 — a malformed profile: one bad slot, no valid ones.
        # ================================================================
        print("\nPass 3 — malformed annotation-profiles.json")
        stub3 = Stub(
            profile_status=200,
            profile_body=json.dumps(MALFORMED_PROFILE),
            profile_type="application/json",
        )
        page.route("**/annotation-profiles.json", stub3.handle_profile)
        page.route("**/curation/**", stub3.handle_kb)

        console.clear()
        page.goto(f"{BASE}/review", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)

        labels = tab_labels(page)
        check("the page still renders with the 5 core tabs + Plates", "Plates" in labels, str(labels))
        check(
            "no 'Pallet labels' tab from a malformed document",
            "Pallet labels" not in labels,
            str(labels),
        )

        plates_tab = page.get_by_role("button", name="Plates")
        check("Plates tab is still clickable", plates_tab.count() > 0)
        plates_tab.first.click()
        page.wait_for_timeout(300)

        toast_text = page.locator("text=/annotation profile/i")
        check(
            "a toast mentions the deployment annotation profile",
            toast_text.count() > 0,
            "no toast matched /annotation profile/i",
        )

        warn_msgs = [c for c in console if "annotation-profiles" in c.lower()]
        print(f"    console warning(s) for the malformed doc: {warn_msgs}")
        check(
            'the console names the specific reason ("bind must be an object")',
            any("bind must be an object" in m for m in warn_msgs),
            str(warn_msgs),
        )

        errors = [c for c in console if c.startswith("pageerror")]
        check("no pageerror on pass 3 (malformed doc must not crash the app)", not errors, str(errors[:3]))
        if OUT_DIR:
            safe_screenshot(page, OUT_DIR / "pass3_malformed_profile.png")

        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
