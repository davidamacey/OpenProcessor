#!/usr/bin/env python3
"""Docs-site screenshot capture — against a REAL Cropwright instance.

Captures full-page screenshots for docs-site/src/data/screenshots.json
and the `<Screenshot>` slots in docs-site/docs/**, at 1600px and 800px
wide, from a live Cropwright instance pointed at a real OpenProcessor
backend.

**Public sample data only.** This script does not stub anything — it
drives whatever backend the target Cropwright instance is actually
talking to. That backend MUST hold only public sample data (COCO
val2017 via `make sample-coco`, Open Images plates via `make
sample-plates`, both in the OpenProcessor checkout). NEVER point this at
a real deployment: its imagery, class names and counts are not public,
and a screenshot captured from one must never be committed. See
docs-site/docs/developer-guide/screenshots.md.

This replaces the old stub-backed version (which rendered synthetic SVG
tiles against a fake `/curation/*` backend) — the docs site now wants
screenshots that show the actual app against actual (public) data, not a
fixture-driven mock.

Usage:
    npm run test:e2e   # once, to provision e2e/.venv with Playwright
    e2e/.venv/bin/python scripts/capture_docs_screenshots.py \\
        --base-url http://localhost:5184
    # or: CROPWRIGHT_URL=http://localhost:5184 e2e/.venv/bin/python scripts/capture_docs_screenshots.py

The route list is NOT hardcoded here — it's read from
docs-site/src/data/screenshot_routes.json, the same directory the
landing page's ScreenshotShowcase component and doc pages' <Screenshot>
slots pull filenames from. Add a route there, not in this script, when a
doc page gains a new screenshot slot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTES_FILE = REPO_ROOT / "docs-site" / "src" / "data" / "screenshot_routes.json"
OUT_DIR = REPO_ROOT / "docs-site" / "static" / "img" / "screenshots"

WIDTHS = (1600, 800)
VIEWPORT_HEIGHT = 1000


def load_routes() -> list[dict]:
    if not ROUTES_FILE.exists():
        sys.exit(f"missing route list: {ROUTES_FILE}")
    return json.loads(ROUTES_FILE.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CROPWRIGHT_URL", "http://localhost:5184"),
        help="Base URL of a Cropwright instance pointed at a PUBLIC-sample-data "
        "OpenProcessor backend only. Never a real deployment.",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Capture only these route `name`s (from screenshot_routes.json), for a quick re-run.",
    )
    args = parser.parse_args()

    routes = load_routes()
    if args.only:
        routes = [r for r in routes if r["name"] in args.only]
        if not routes:
            sys.exit(f"no matching routes for --only {args.only}")

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Capturing against {args.base_url} — confirm this is a PUBLIC-sample-data instance.")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--disable-gpu"])
        try:
            for route in routes:
                for width in WIDTHS:
                    page = browser.new_page(viewport={"width": width, "height": VIEWPORT_HEIGHT})
                    url = args.base_url.rstrip("/") + route["route"]
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    wait_for = route.get("wait_for")
                    if wait_for:
                        try:
                            page.wait_for_selector(wait_for, timeout=10_000)
                        except Exception:
                            print(f"  warning: selector {wait_for!r} not found for {route['route']} @ {width}px")
                    page.wait_for_timeout(500)  # let in-flight images/animations settle
                    out_path = args.out / f"{route['name']}-{width}.png"
                    page.screenshot(path=str(out_path), full_page=True)
                    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")
                    page.close()
        finally:
            browser.close()

    print(
        "\nDone. Review every PNG for leaked hostnames/paths before committing, "
        "then commit under docs-site/static/img/screenshots/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
