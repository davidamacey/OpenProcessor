#!/usr/bin/env python3
"""Headed Playwright smoke test for Cropwright.

Run via the openprocessor .venv (it ships with playwright):

    DISPLAY=:11 source /data/repos/openprocessor/.venv/bin/activate
    python scripts/playwright_smoke.py [--url URL] [--out DIR]

Captures network requests, console messages, and image render status for
each page in PAGES, then writes screenshots + a JSON summary to OUT.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, Page, ConsoleMessage, Response

PAGES = [
    ("home", "/"),
    ("clusters", "/clusters"),
    ("classes", "/classes"),
    ("review", "/review"),
    ("export", "/export"),
    ("train", "/train"),
    ("models", "/models"),
]


def visit(page: Page, name: str, base_url: str, path: str, out_dir: Path) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    consoles: list[dict[str, Any]] = []

    def on_response(resp: Response) -> None:
        try:
            ct = resp.headers.get("content-type", "")
            cl = resp.headers.get("content-length", "")
            requests.append({
                "url": resp.url,
                "status": resp.status,
                "content_type": ct,
                "content_length": cl,
            })
        except Exception as e:  # noqa: BLE001
            requests.append({"url": resp.url, "error": str(e)})

    def on_console(msg: ConsoleMessage) -> None:
        consoles.append({"type": msg.type, "text": msg.text})

    page.on("response", on_response)
    page.on("console", on_console)

    url = f"{base_url}{path}"
    nav_err: str | None = None
    t0 = time.time()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass  # not all pages reach idle (SSE keeps connection open)
        # Explicitly wait for in-flight images to finish.
        try:
            page.wait_for_function(
                "() => Array.from(document.images).every(i => i.complete)",
                timeout=15000,
            )
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        nav_err = str(e)
    elapsed = time.time() - t0

    # Inspect <img> elements
    imgs = page.evaluate(
        """() => Array.from(document.querySelectorAll('img')).map(i => ({
          src: i.currentSrc || i.src,
          naturalWidth: i.naturalWidth,
          naturalHeight: i.naturalHeight,
          complete: i.complete,
        }))"""
    )

    shot = out_dir / f"{name}.png"
    try:
        page.screenshot(path=str(shot), full_page=True)
    except Exception as e:  # noqa: BLE001
        shot = None  # type: ignore[assignment]
        print(f"[{name}] screenshot failed: {e}", file=sys.stderr)

    page.remove_listener("response", on_response)
    page.remove_listener("console", on_console)

    img_loaded = sum(1 for i in imgs if i["naturalWidth"] > 0)
    img_total = len(imgs)
    failed_reqs = [r for r in requests if r.get("status", 0) >= 400]

    summary = {
        "name": name,
        "url": url,
        "elapsed_s": round(elapsed, 2),
        "nav_error": nav_err,
        "img_total": img_total,
        "img_loaded": img_loaded,
        "img_broken_examples": [i for i in imgs if i["naturalWidth"] == 0][:5],
        "console_errors": [c for c in consoles if c["type"] in ("error", "warning")][:20],
        "failed_requests": failed_reqs[:20],
        "screenshot": str(shot) if shot else None,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.0.2.10:5184")
    p.add_argument("--out", default="/tmp/labeler_smoke")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()

        results = []
        for name, path in PAGES:
            print(f"[smoke] visiting {name} {path}", flush=True)
            results.append(visit(page, name, args.url, path, out_dir))

        # Try a cluster detail page: pull a cluster_id from the clusters list
        try:
            import urllib.request
            with urllib.request.urlopen(f"{args.url}/curation/clusters?per_cluster=1&max_clusters=5", timeout=10) as r:
                stats = json.load(r)
            ids = [c["cluster_id"] for c in stats.get("items", [])]
            if ids:
                results.append(visit(page, "cluster_detail", args.url, f"/clusters/{ids[0]}", out_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[smoke] cluster_detail skip: {e}", file=sys.stderr)

        browser.close()

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[smoke] summary -> {summary_path}")
    for r in results:
        status = "OK" if r["img_total"] == 0 or r["img_loaded"] > 0 else "FAIL"
        if r["nav_error"]:
            status = "NAV-FAIL"
        print(f"  [{status}] {r['name']:18s} imgs={r['img_loaded']}/{r['img_total']} "
              f"console_warn_err={len(r['console_errors'])} failed_reqs={len(r['failed_requests'])} "
              f"elapsed={r['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
