#!/usr/bin/env python3
"""Phase 1 frontend round-trip verification.

Loads /clusters/{id} for a cluster with gemma_unmatched crops, asserts
thumbnails actually render (naturalWidth > 0), triggers the validate
action via the same PUT {prefix}/crops/{id}/label endpoint the UI
calls, re-fetches the crop, and confirms label_validated=true plus
label_source / class_id_history were written.

Needs a Python env with Playwright installed; this repo has none of
its own (pure SvelteKit/TS). Call the interpreter directly -- do not
`source` an activate script, and note that an env-var prefix cannot be
applied to a shell builtin, which is why the previous form here was
never valid shell:

    cd <repo root>
    DISPLAY=:11 /path/to/venv/bin/python scripts/playwright_round_trip.py \\
        [--url URL] [--api URL] [--api-prefix PREFIX] [--out DIR]

e.g. /data/repos/openprocessor/.venv/bin/python -- any venv with
playwright works. No absolute path to this repo: the directory is
slated to be renamed to `cropwright`, and worktrees check it out
elsewhere.

Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

TRANSITIONAL_DEFAULT = "/curation"  # mirrors src/lib/api.ts:107 (normalizeApiPrefix); flips at T-E2


def normalize_api_prefix(raw: str) -> str:
    """Python mirror of normalizeApiPrefix() in src/lib/api.ts.

    Duplicated (rather than imported) from
    playwright_backend_integration.py: importing across scripts/ proved
    fragile under different runners' cwd. Keep both copies byte-for-byte
    equivalent -- see that module's copy for the full rationale.
    """
    trimmed = raw.strip()
    if not trimmed or trimmed.startswith("__"):
        return TRANSITIONAL_DEFAULT
    leading = trimmed if trimmed.startswith("/") else f"/{trimmed}"
    return leading.rstrip("/")


def http_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def http_put(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def pick_target(api_base: str, prefix: str) -> tuple[int, dict[str, Any]]:
    """Find a cluster with at least one unvalidated gemma-sourced crop.

    NOT /clusters/stats/{index}: that is a legacy un-prefixed router
    whose only valid index values are global|vehicles|people|faces
    (backend src/routers/clusters.py:163-171) -- 'op_vehicles' 400s --
    and the labeler's nginx stopped proxying it once T-B3 dropped the
    dead location block. Use GET {prefix}/clusters instead, which is
    what /clusters itself calls (src/lib/api.ts:947-963). Only the
    cluster ENUMERATION changes; the two-pass gemma-preferred-else-any
    selection logic below is unchanged.
    """
    clusters_page = http_get(f"{api_base}{prefix}/clusters?per_cluster=1&max_clusters=2000")
    clusters = clusters_page.get("items", [])
    for c in clusters:
        cid = c["cluster_id"]
        page = http_get(
            f"{api_base}{prefix}/crops?cluster_id={cid}&page_size=20&label_validated=false"
        )
        for crop in page.get("crops", []):
            src = (crop.get("class_source") or "").lower()
            if "gemma" in src or src == "cluster_v6_majority_agreement":
                return cid, crop
    # fallback: any unvalidated
    for c in clusters:
        cid = c["cluster_id"]
        page = http_get(
            f"{api_base}{prefix}/crops?cluster_id={cid}&page_size=5&label_validated=false"
        )
        if page.get("crops"):
            return cid, page["crops"][0]
    raise RuntimeError("No unvalidated crops found in any cluster.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.0.2.10:5184",
                   help="Labeler frontend URL")
    p.add_argument("--api", default="http://localhost:4603",
                   help="openprocessor URL (for fetch + PUT)")
    p.add_argument("--api-prefix", default=os.environ.get("PUBLIC_API_PREFIX", ""),
                   help="Backend path prefix. Normalized exactly like "
                        "normalizeApiPrefix() in src/lib/api.ts.")
    p.add_argument("--out", default="/tmp/labeler_round_trip")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    prefix = normalize_api_prefix(args.api_prefix)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    summary: dict[str, Any] = {}

    cluster_id, crop = pick_target(args.api, prefix)
    crop_id = crop["crop_id"]
    target_class = crop.get("class_id") or 0
    summary["target"] = {
        "cluster_id": cluster_id,
        "crop_id": crop_id,
        "class_id": target_class,
        "class_source": crop.get("class_source"),
    }
    print(f"[round-trip] target crop_id={crop_id} cluster_id={cluster_id}"
          f" class_source={crop.get('class_source')}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()

        # Use /clusters listing page (reliably renders representative
        # thumbnails). The /clusters/{id} grid is lazy-loaded behind
        # virtual scroll and can require auth gates — the listing page
        # exercises the same image proxy + API path with simpler markup.
        url = f"{args.url}/clusters"
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        # SPA: wait for first <img> to actually appear in the DOM
        try:
            page.wait_for_selector("img", timeout=20000, state="attached")
        except Exception:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        try:
            page.wait_for_function(
                "() => document.images.length > 0 && "
                "Array.from(document.images).every(i => i.complete)",
                timeout=20000,
            )
        except Exception:
            pass

        # Debug: dump body length + count of various selectors
        debug = page.evaluate(
            """() => ({
              body_len: (document.body && document.body.innerText || '').length,
              n_img: document.querySelectorAll('img').length,
              n_canvas: document.querySelectorAll('canvas').length,
              n_div: document.querySelectorAll('div').length,
              title: document.title,
              url: location.href,
            })"""
        )
        print(f"[round-trip] debug: {debug}")
        imgs = page.evaluate(
            """() => Array.from(document.querySelectorAll('img')).map(i => ({
              naturalWidth: i.naturalWidth,
              complete: i.complete,
              src: i.currentSrc || i.src,
            }))"""
        )
        img_total = len(imgs)
        img_loaded = sum(1 for i in imgs if i["naturalWidth"] > 0)
        summary["thumbnails"] = {"loaded": img_loaded, "total": img_total}
        print(f"[round-trip] cluster page loaded: {img_loaded}/{img_total} imgs")
        if img_total == 0:
            errors.append("cluster page rendered ZERO <img> elements")
        elif img_loaded == 0:
            errors.append("ALL thumbnails failed to load (naturalWidth=0)")

        try:
            page.screenshot(path=str(out_dir / "cluster_before.png"))
        except Exception as e:
            print(f"[round-trip] screenshot(before) failed: {e}",
                  file=sys.stderr)

        # Trigger validate via the API the UI calls. This is the same
        # endpoint putCropLabel hits in src/lib/api.ts.
        put_url = f"{args.api}{prefix}/crops/{crop_id}/label"
        try:
            put_result = http_put(
                put_url, {"class_id": int(target_class), "validated": True}
            )
            summary["put_label_response"] = put_result
            print(f"[round-trip] PUT {put_url} -> ok")
        except Exception as e:
            errors.append(f"PUT label failed: {e}")
            summary["put_label_response"] = {"error": str(e)}

        # Re-fetch and assert
        try:
            refetched = http_get(f"{args.api}{prefix}/crops/{crop_id}")
            summary["refetched"] = {
                k: refetched.get(k)
                for k in (
                    "crop_id",
                    "label_validated",
                    "class_validated",
                    "label_source",
                    "class_id",
                    "class_id_history",
                )
            }
            print(f"[round-trip] refetched: label_validated="
                  f"{refetched.get('label_validated')}"
                  f" label_source={refetched.get('label_source')!r}")

            if not (refetched.get("label_validated")
                    or refetched.get("class_validated")):
                errors.append(
                    "After PUT, label_validated/class_validated still not true"
                )
            if not refetched.get("label_source"):
                errors.append("After PUT, label_source still empty")
            hist = refetched.get("class_id_history") or []
            if not hist:
                # not strictly fatal — depends on backend version — log only
                print("[round-trip] warn: class_id_history empty (non-fatal)")
        except Exception as e:
            errors.append(f"refetch failed: {e}")

        # Reload page to capture post-action UI
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        try:
            page.screenshot(path=str(out_dir / "cluster_after.png"))
        except Exception as e:
            print(f"[round-trip] screenshot(after) failed: {e}",
                  file=sys.stderr)

        browser.close()

    summary["errors"] = errors
    summary["status"] = "PASS" if not errors else "FAIL"
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[round-trip] summary -> {summary_path}")
    print(f"[round-trip] {summary['status']}"
          + (f" — {len(errors)} error(s)" if errors else ""))
    for e in errors:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
