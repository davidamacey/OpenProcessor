#!/usr/bin/env python3
"""Mock-backed browser check for the scoped-VLM-labeling-assist bar.

No live OpenProcessor backend exists that advertises the `detection_profile`
/ `prompt_pack` `/methods` axes or accepts `class_id` on
`POST /curation/pipeline/auto_label/start` (docs/design/
vlm-scoped-labeling-assist-plan-2026-09-20.md). This drives the real
dashboard UI against a stubbed `/curation/*` backend (Playwright route
interception, same structure as scripts/playwright_labeling_flow.py) so the
rendering + wire-composition logic can be verified without one.

Runs the dashboard twice:

  Pass 1 — /curation/methods answers with today's real shape (no assist axes).
           The assist bar must be entirely absent, and the unscoped start
           request must carry none of class_id/detection_profile/prompt_pack.

  Pass 2 — /curation/methods advertises both assist axes (plus one shadow profile
           and one disabled pack, to prove the status filter). The bar
           appears, the class picker narrows to "pallets", a non-default
           detector/prompt pack can be picked, and the recorded start
           request carries every scoped param plus all the pre-existing
           ones.

Usage:  python3 scripts/playwright_assist_scope.py [base_url]
        DISPLAY=:11 python3 scripts/playwright_assist_scope.py --headed [base_url]
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

args = sys.argv[1:]
HEADED = "--headed" in args
args = [a for a in args if a != "--headed"]
BASE = args[0] if args else "http://localhost:5173"

CLASSES = [
    {
        "id": 1,
        "name": "pallets",
        "group": "warehouse",
        "hotkey_letter": "p",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "forklift",
        "group": "warehouse",
        "hotkey_letter": "f",
        "count": 20,
        "validated_count": 5,
        "cluster_size": 22,
        "deprecated": False,
    },
]

# Today's real /curation/methods shape — no assist axes at all.
METHODS_TODAY = {
    "strategies": [
        {
            "id": "ivf",
            "axis": "cluster",
            "label": "FAISS IVF-512 (production)",
            "status": "stable",
            "default": True,
        },
        {
            "id": "default",
            "axis": "sort",
            "label": "Recent first",
            "status": "stable",
            "default": True,
        },
    ],
    "flags": {},
}

# Same payload plus both assist axes, plus one deliberately-shadow profile
# and one deliberately-disabled pack so the status filter is exercised, not
# just the happy path.
METHODS_WITH_ASSIST_AXES = {
    "strategies": [
        *METHODS_TODAY["strategies"],
        {
            "id": "grounding_v2",
            "axis": "detection_profile",
            "label": "Grounding detector v2",
            "status": "stable",
            "default": True,
        },
        {
            "id": "legacy_profile",
            "axis": "detection_profile",
            "label": "Legacy profile (mid-validation)",
            "status": "shadow",
        },
        {
            "id": "warehouse_v1",
            "axis": "prompt_pack",
            "label": "Warehouse vocabulary",
            "status": "stable",
            "default": True,
        },
        {
            "id": "retired_pack",
            "axis": "prompt_pack",
            "label": "Retired prompt pack",
            "status": "disabled",
        },
    ],
    "flags": {},
}

IDLE_JOB = {
    "job_id": "job-0",
    "status": "idle",
    "stage": "",
    "processed": 0,
    "total": 0,
    "started_at": 0,
    "finished_at": 0,
    "error": None,
    "result": {},
    "args": {},
    "eta_seconds": None,
    "elapsed_seconds": 0,
}


class Stub:
    """Canned /curation/* responses + a record of every auto_label/start call."""

    def __init__(self, methods_body: dict[str, Any]) -> None:
        self.methods_body = methods_body
        self.calls: list[str] = []

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
        if path.startswith("/curation/pipeline/auto_label/status"):
            return ok(IDLE_JOB)
        if path.startswith("/curation/pipeline/auto_label/start") and method == "POST":
            # This recording is the point of the whole script — the only
            # place in the plan where the composed wire request is observed
            # as the browser actually sends it.
            self.calls.append(url)
            return ok(IDLE_JOB)
        if "/thumbnail" in path or "/source" in path:
            return route.fulfill(
                status=200,
                content_type="image/gif",
                body=bytes.fromhex(
                    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
                ),
            )
        return ok({})


FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{'' if cond else ' — ' + detail}")
    if not cond:
        FAILURES.append(name)


def query_of(url: str) -> str:
    return urlsplit(url).query


def main() -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not HEADED)
        page = browser.new_page()
        console: list[str] = []
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))

        # ================================================================
        # Pass 1 — today's real backend: no assist axes advertised at all.
        # ================================================================
        print("\nPass 1 — METHODS_TODAY (no assist axes)")
        stub1 = Stub(METHODS_TODAY)
        page.route("**/curation/**", stub1.handle)

        page.goto(f"{BASE}/dashboard", wait_until="networkidle")
        page.wait_for_timeout(500)

        errors = [c for c in console if c.startswith("pageerror")]
        check("dashboard renders with no pageerror", not errors, str(errors[:3]))

        assist_chip = page.get_by_text(re.compile(r"^assist:"))
        check("no 'assist:' chip renders — absent, not disabled", assist_chip.count() == 0)

        start_btn = page.get_by_role("button", name="Recluster now")
        check("primary button reads exactly 'Recluster now'", start_btn.count() > 0)

        stub1.calls.clear()
        start_btn.first.click()
        page.wait_for_timeout(500)
        check("exactly one POST to auto_label/start", len(stub1.calls) == 1, str(stub1.calls))
        if stub1.calls:
            qs = query_of(stub1.calls[0])
            print(f"    recorded unscoped query string: {qs!r}")
            check("unscoped request carries no class_id", "class_id" not in qs, qs)
            check(
                "unscoped request carries no detection_profile", "detection_profile" not in qs, qs
            )
            check("unscoped request carries no prompt_pack", "prompt_pack" not in qs, qs)

        page.unroute("**/curation/**")

        # ================================================================
        # Pass 2 — backend advertises both assist axes.
        # ================================================================
        print("\nPass 2 — METHODS_WITH_ASSIST_AXES")
        stub2 = Stub(METHODS_WITH_ASSIST_AXES)
        page.route("**/curation/**", stub2.handle)

        console.clear()
        page.goto(f"{BASE}/dashboard", wait_until="networkidle")
        page.wait_for_timeout(500)

        chip = page.get_by_text(re.compile(r"^assist:\s*whole dataset"))
        check("'assist: whole dataset' chip is present", chip.count() > 0)

        chip.first.click()
        page.wait_for_timeout(200)

        class_search = page.locator("input[type=search]")
        check("class search box appears", class_search.count() > 0)
        detector_select = page.locator("select").filter(has_text="Server default").nth(0)
        check("detector <select> appears", page.get_by_text("Server default").count() >= 1)

        options_text = page.locator("select option").all_inner_texts()
        check(
            "shadow profile / disabled pack are not offered as options",
            "Legacy profile (mid-validation)" not in options_text
            and "Retired prompt pack" not in options_text,
            str(options_text),
        )
        check(
            "usable profile/pack ARE offered",
            any("Grounding detector v2" in t for t in options_text)
            and any("Warehouse vocabulary" in t for t in options_text),
            str(options_text),
        )

        class_search.first.fill("pall")
        page.wait_for_timeout(200)
        pallets_row = page.get_by_text("pallets", exact=False)
        check("typing 'pall' narrows the list to pallets", pallets_row.count() > 0)
        pallets_row.first.click()
        page.wait_for_timeout(150)

        selects = page.locator("select")
        if selects.count() >= 2:
            selects.nth(0).select_option("grounding_v2")
            selects.nth(1).select_option("warehouse_v1")
        page.wait_for_timeout(150)

        collapse_btn = page.get_by_role("button", name="×")
        if collapse_btn.count() > 0:
            collapse_btn.first.click()
        page.wait_for_timeout(200)

        scoped_chip = page.get_by_text(re.compile(r"^assist:\s*pallets"))
        check("chip now reads 'assist: pallets'", scoped_chip.count() > 0)
        scoped_btn = page.get_by_role("button", name=re.compile(r"^Recluster · pallets$"))
        check("button now reads 'Recluster · pallets'", scoped_btn.count() > 0)

        stub2.calls.clear()
        scoped_btn.first.click()
        page.wait_for_timeout(500)
        check("exactly one scoped POST to auto_label/start", len(stub2.calls) == 1, str(stub2.calls))
        if stub2.calls:
            qs = query_of(stub2.calls[0])
            print(f"    recorded scoped query string: {qs!r}")
            check("scoped request carries class_id=1 (pallets)", "class_id=1" in qs, qs)
            check(
                "scoped request carries detection_profile=grounding_v2",
                "detection_profile=grounding_v2" in qs,
                qs,
            )
            check(
                "scoped request carries prompt_pack=warehouse_v1",
                "prompt_pack=warehouse_v1" in qs,
                qs,
            )
            check(
                "scoped request still carries the seven pre-existing params",
                all(
                    p in qs
                    for p in [
                        "train_clusters",
                        "gemma_concurrency",
                        "max_gemma_crops",
                        "recluster_unvalidated",
                    ]
                ),
                qs,
            )

        # Re-expand, reset — chip and button return to the unscoped defaults.
        page.goto(f"{BASE}/dashboard", wait_until="networkidle")
        page.wait_for_timeout(500)
        chip2 = page.get_by_text(re.compile(r"^assist:"))
        chip2.first.click()
        page.wait_for_timeout(200)
        reset_btn = page.get_by_role("button", name="reset")
        if reset_btn.count() > 0:
            reset_btn.first.click()
        page.wait_for_timeout(200)
        # The "assist: <summary>" text only renders in the collapsed state
        # (the expanded state shows the class search + selects instead), so
        # collapse back down before checking the chip's summary text.
        collapse_btn2 = page.get_by_role("button", name="×")
        if collapse_btn2.count() > 0:
            collapse_btn2.first.click()
        page.wait_for_timeout(200)
        check(
            "reset returns the chip to 'assist: whole dataset'",
            page.get_by_text(re.compile(r"^assist:\s*whole dataset")).count() > 0,
        )

        errors = [c for c in console if c.startswith("pageerror")]
        check("no pageerror across pass 2", not errors, str(errors[:3]))

        browser.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
