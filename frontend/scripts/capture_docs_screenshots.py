#!/usr/bin/env python3
"""Docs-site screenshot capture — against a REAL Cropwright instance.

Captures full-page screenshots for the `<Screenshot>` slots in docs-site/docs/cropwright/**, at 1600px and 800px
wide, from a live Cropwright instance pointed at a real OpenProcessor
backend.

**Public sample data only.** This script does not stub anything — it
drives whatever backend the target Cropwright instance is actually
talking to. That backend MUST hold only public sample data (COCO
val2017 via `make sample-coco`, Open Images plates via `make
sample-plates`, both in the OpenProcessor checkout). NEVER point this at
a real deployment: its imagery, class names and counts are not public,
and a screenshot captured from one must never be committed. See
docs-site/docs/cropwright/developer-guide/screenshots.md.

This replaces the old stub-backed version (which rendered synthetic SVG
tiles against a fake `/curation/*` backend) — the docs site now wants
screenshots that show the actual app against actual (public) data, not a
fixture-driven mock.

Usage:
    npm run test:e2e   # once, to provision e2e/.venv with Playwright
    e2e/.venv/bin/python scripts/capture_docs_screenshots.py \\
        --base-url http://localhost:<port-of-a-public-data-instance>
    # or: OP_CROPWRIGHT_URL=... e2e/.venv/bin/python scripts/capture_docs_screenshots.py
    # or, from the repo root: make docs-screenshots OP_CROPWRIGHT_URL=...

The capture is read-only: every non-GET/HEAD request is aborted, so a
screenshot pass can never write to the backend it points at.

Besides the routes it captures a set of STATE slots (a dialog opened, a crop
selected, a pack test run) with scripted navigation — see STATES in
docs_screenshots/registry.py. A
state only ever opens and looks: nothing is saved, submitted or confirmed.

Read-only guard: GET/HEAD pass; every other request is aborted EXCEPT the
small explicit allow-list in READ_ONLY_CALLS (docs_screenshots/guard.py; calls the UI makes only to
render a report/dialog, each of which writes nothing server-side). Keep
docs-site/docs/cropwright/developer-guide/screenshots.md's list in step with it.

The route list is NOT hardcoded here: it is read from
frontend/scripts/screenshot_routes.json. Add a route there, or a state to
STATES in docs_screenshots/registry.py, when a doc page gains a new <Screenshot> slot. The full guide
(data rule, fixtures, `make docs-screenshots`) is docs-site/docs/developer-guide/screenshots.mdx.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from docs_screenshots.ctx import Ctx, _settle
from docs_screenshots.guard import _read_only
from docs_screenshots.registry import STATE_HEIGHT, STATES
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTES_FILE = REPO_ROOT / "scripts" / "screenshot_routes.json"
OUT_DIR = REPO_ROOT.parent / "docs-site" / "static" / "img" / "screenshots" / "cropwright"

WIDTHS = (1600, 800)
VIEWPORT_HEIGHT = 1000


def load_routes() -> list[dict]:
    if not ROUTES_FILE.exists():
        sys.exit(f"missing route list: {ROUTES_FILE}")
    return json.loads(ROUTES_FILE.read_text())


def scoped_route(route: str, project: str | None) -> str:
    """Bare per-project paths get the `/p/<project>` prefix; `/projects*` is global."""
    if project is None or route.startswith("/projects"):
        return route
    return f"/p/{project}{route}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OP_CROPWRIGHT_URL"),
        required="OP_CROPWRIGHT_URL" not in os.environ,
        help="Base URL of a Cropwright instance pointed at a PUBLIC-sample-data "
        "OpenProcessor backend only. Never a real deployment.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("OP_CROPWRIGHT_PROJECT"),
        help="Project slug holding the public sample data. Without it bare routes "
        "redirect to the default project.",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Capture only these route/state `name`s, for a quick re-run.",
    )
    parser.add_argument(
        "--widths",
        nargs="*",
        type=int,
        default=list(WIDTHS),
        help="Viewport widths for the route captures (states are always 1600).",
    )
    parser.add_argument(
        "--skip-routes", action="store_true", help="Capture only the scripted states."
    )
    args = parser.parse_args()
    args.out = args.out.resolve()

    routes = [] if args.skip_routes else load_routes()
    states = dict(STATES)
    if args.only:
        routes = [r for r in routes if r["name"] in args.only]
        states = {k: v for k, v in states.items() if k in args.only}
        if not routes and not states:
            sys.exit(f"no matching routes or states for --only {args.only}")
    elif args.project is None:
        states = {}  # states need a project

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Capturing against {args.base_url} — confirm this is a PUBLIC-sample-data instance.")

    def finish(page, name: str, width: int, clip=None) -> None:
        out_path = args.out / f"{name}-{width}.png"
        if clip is not None:  # a state may return a locator to crop to
            clip.screenshot(path=str(out_path))
        else:
            page.screenshot(path=str(out_path), full_page=True)
        print(f"  wrote {out_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-gpu"])
        try:
            for route in routes:
                for width in args.widths:
                    page = browser.new_page(viewport={"width": width, "height": VIEWPORT_HEIGHT})
                    page.route("**/*", _read_only)
                    url = args.base_url.rstrip("/") + scoped_route(route["route"], args.project)
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    wait_for = route.get("wait_for")
                    if wait_for:
                        try:
                            page.wait_for_selector(wait_for, timeout=10_000)
                        except Exception:
                            print(f"  warning: selector {wait_for!r} not found for {route['route']} @ {width}px")
                    _settle(page)
                    finish(page, route["name"], width)
                    page.close()
            ctx = Ctx(args.base_url, args.project or "")
            for name, fn in states.items():
                page = browser.new_page(
                    viewport={"width": 1600, "height": STATE_HEIGHT.get(name, VIEWPORT_HEIGHT)}
                )
                page.route("**/*", _read_only)
                try:
                    finish(page, name, 1600, fn(page, ctx))
                except Exception as e:  # keep going; a failed state is reported, not faked
                    print(f"  FAILED state {name}: {type(e).__name__}: {str(e)[:300]}")
                page.close()
        finally:
            browser.close()

    print(
        "\nDone. Review every PNG for leaked hostnames/paths before committing, "
        "then commit under docs-site/static/img/screenshots/cropwright/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
