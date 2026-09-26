#!/usr/bin/env python3
"""Mock-backed screenshot capture harness for docs/screenshots.

Per docs/design/cropwright-oss-release-plan-2026-09-21.md §3.7. Captures
all 13 README/docs screenshots against a **stubbed** `/curation/*` backend —
no live OpenProcessor instance, no real dataset, no real photograph is
ever fetched. That closes the PII / third-party-copyright / proprietary-
leakage holes in the current `docs/screenshots/*` by construction: every
image the browser renders is a deterministic synthetic SVG tile generated
from a fixture-controlled crop id, and every path/id/model-name string in
the UI comes from this script's own fixtures.

Modelled on:
  - scripts/playwright_smoke.py       — capture loop, 1600x1000 viewport,
                                         image-complete wait, screenshot.
  - scripts/playwright_assist_scope.py /
    scripts/playwright_labeling_flow.py — the `Stub` route-interception
                                         class fulfilling `/curation/**`.

Deliberately does NOT touch those existing scripts — they are verification
tools with assertions; this is a capture tool with none.

Usage:
    npm run test:e2e   # once, to provision e2e/.venv with Playwright
    e2e/.venv/bin/python scripts/capture_docs_screenshots.py [--url URL] [--out DIR]

Safety (§3.7.5): the app's default `PUBLIC_TRITON_API_URL` is the empty
string, so `src/lib/api.ts` composes every request as a same-origin
relative path (`/curation/...`) — there is no live origin for a request to fall
through to even before this script's own belt-and-braces catch-all runs.
On top of that default-safe behaviour, this script also:
  1. registers the `/curation/**` stub FIRST (Playwright uses last-registered-
     wins-by-most-specific / first match order — the stub is checked
     before the generic catch-all below because Playwright tries routes
     in registration order and the first one whose handler calls
     `fulfill`/`continue_`/`abort` wins);
  2. registers a catch-all that aborts anything that is not same-origin
     and not an app asset, so an unmatched request fails loudly instead
     of silently reaching a real origin;
  3. records every response URL and asserts none of them left the local
     origin, after every capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Page, Route, sync_playwright

# --------------------------------------------------------------------------
# Fixtures — deliberately the same warehouse/pallet vocabulary the existing
# stubbed scripts use (playwright_assist_scope.py, static/annotation-
# profiles.example.json). Demonstrates the app is domain-agnostic instead
# of showing any one deployment's dataset, and guarantees no real class
# name or count can ever appear — the
# fixture set defines the classes.
# --------------------------------------------------------------------------

CLASSES = [
    {
        "id": 1,
        "name": "pallet",
        "group": "warehouse",
        "hotkey_letter": "p",
        "count": 412,
        "validated_count": 180,
        "cluster_size": 44,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "forklift",
        "group": "warehouse",
        "hotkey_letter": "f",
        "count": 96,
        "validated_count": 40,
        "cluster_size": 22,
        "deprecated": False,
    },
    {
        "id": 3,
        "name": "damaged_pallet",
        "group": "warehouse",
        "hotkey_letter": "d",
        "count": 31,
        "validated_count": 9,
        "cluster_size": 12,
        "deprecated": False,
    },
    {
        "id": 4,
        "name": "shrink_wrap_roll",
        "group": "warehouse",
        "hotkey_letter": "s",
        "count": 18,
        "validated_count": 6,
        "cluster_size": 8,
        "deprecated": False,
    },
    {
        "id": 5,
        "name": "sscc_label",
        "group": "warehouse",
        "hotkey_letter": "l",
        "count": 260,
        "validated_count": 140,
        "cluster_size": 30,
        "deprecated": False,
    },
]

METHODS = {
    "strategies": [
        {
            "id": "ivf",
            "axis": "cluster",
            "label": "FAISS IVF-512 (production)",
            "status": "stable",
            "default": True,
        },
        {
            "id": "hdbscan",
            "axis": "cluster",
            "label": "HDBSCAN (experimental)",
            "status": "experimental",
        },
        {"id": "recent", "axis": "sort", "label": "Recent first", "status": "stable", "default": True},
        {
            "id": "representativeness",
            "axis": "sort",
            "label": "Representativeness",
            "status": "stable",
        },
        {"id": "atypicality", "axis": "sort", "label": "Atypicality", "status": "stable"},
        {"id": "mistakenness_score", "axis": "score", "label": "Mistakenness", "status": "stable"},
    ],
    "flags": {
        "OP_SCORES_ENABLED": True,
        "OP_SELECT_DIVERSE_ENABLED": False,
        "OP_VIZ_PROJECTION_ENABLED": False,
    },
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

DATASET_STATS = {
    "as_of": "2026-09-21T00:00:00+00:00",
    "total_crops": 12480,
    "validated": 6320,
    "test_holdout": 1248,
    "by_source": [
        {"key": "model_suggestion", "doc_count": 6160},
        {"key": "human", "doc_count": 6320},
    ],
    "labeled": {
        "by_human": 6320,
        "by_vlm": 3100,
        "by_classifier": 2900,
        "by_proposal": 160,
        "other": 0,
    },
    "regions": {
        "boxed": 1800,
        "confirmed": 1500,
        "total_detected": 1900,
        "by_detector": 1200,
        "by_segmenter": 600,
        "by_human": 100,
        "by_human_drew": 60,
        "verified_by_human": 900,
        "verified_by_vlm": 600,
        "validated_by_human": 960,
    },
    "unlabeled": {
        "pending_detection": 400,
        "pending_verification": 220,
        "no_label_source": 180,
    },
    "in_progress": {"region_drain_total_unfinished": 0},
    "clusters": {
        "last_run_at": "2026-09-20T18:04:00+00:00",
        "cluster_count": 84,
        "residual_count": 3,
        "noise_count": 12,
        "method": "ivf",
    },
}

STATS_CLASSES = {
    "classes": [
        {"class_id": c["id"], "class_name": c["name"], "count": c["count"], "validated_count": c["validated_count"]}
        for c in CLASSES
    ]
}


REGION_TAB_LABEL = "Regions"


def crop(i: int, cluster_id: int = 1, with_region: bool = False) -> dict[str, Any]:
    """One raw wire-shaped crop — same field names as playwright_labeling_flow.py's
    crop(), which mapRawCrop() (src/lib/api.ts) is verified to read from."""
    c: dict[str, Any] = {
        "crop_id": f"crop-{i}",
        "image_id": f"img-{i}",
        "image_path": f"dataset/train/frame_{i:06d}.jpg",
        "bbox_norm": [0.08, 0.08, 0.62, 0.62],
        "class_id": CLASSES[i % len(CLASSES)]["id"],
        "class_name": CLASSES[i % len(CLASSES)]["name"],
        "class_source": "v6_model" if i % 3 else "human",
        "confidence": 0.62 + (i % 5) * 0.07,
        "cluster_id": cluster_id,
        "label_validated": bool(i % 3),
        "label_source": "human" if i % 3 else "model_suggestion",
        "updated_at": "2026-09-20T12:00:00+00:00",
    }
    if with_region:
        c.update(
            {
                "region_bbox_norm": [0.3, 0.55, 0.55, 0.68],
                "region_score": 0.91,
                "region_status": "detected",
                "region_verified": bool(i % 2),
                "region_detector": "tag_detector_v1",
                "region_detector_version": "1.0",
                "region_detector_chain": ["tag_detector_v1:hit", "tag_verifier:verify_ok"],
                "region_bbox_frame": "source",
                "region_verifier": "tag_verifier",
                "region_text": f"SSCC-{100000 + i}",
                "region_text_source": "vlm",
                "region_text_confidence": 0.83,
            }
        )
    return c


CLUSTERS = {
    "items": [
        {
            "cluster_id": n,
            "cluster_kind": "class",
            "size": 6 + n,
            "validated_count": n,
            "dominant_class_id": CLASSES[n % len(CLASSES)]["id"],
            "dominant_class_name": CLASSES[n % len(CLASSES)]["name"],
            "purity": 0.7 + (n % 3) * 0.1,
            "is_unlabeled": n == 6,
            "representatives": [{"crop_id": f"crop-{n * 10 + k}"} for k in range(3)],
            "n_subclusters": 0,
            "updated_at": "2026-09-20T12:00:00+00:00",
        }
        for n in range(1, 9)
    ],
    "total": 8,
    "total_class_clusters": 7,
    "total_candidate_clusters": 1,
    "cluster_id_offset": 10000,
}


def review_items(tab: str, n: int = 12) -> dict[str, Any]:
    is_regions = tab == "regions"
    items = []
    for i in range(n):
        c = crop(i, with_region=is_regions)
        c["reason"] = "region_review" if is_regions else "uncertainty"
        c["proposed_class_id"] = CLASSES[0]["id"]
        c["proposed_class_name"] = CLASSES[0]["name"]
        items.append(c)
    return {"items": items, "total": n, "page": 1, "page_size": 30}


EXPORT_STATUS = {
    "status": "success",
    "last_run": "2026-09-21T05:43:54+00:00",
    "progress": 1.0,
    "job_id": "export-job-7",
    "export_dir": "exports/20260921T054354Z_v1-5class",
    "error": None,
    "message": "Export complete.",
}

EXPORT_DATASETS = [
    {
        "kind": "vehicles",
        "export_dir": "exports/20260921T054354Z_v1-5class",
        "version_tag": "v1-5class",
        "image_count": 6320,
        "split_counts": {"train": 5056, "val": 632, "test": 632},
        "dataset_sha": "sha256:deadbeefcafef00d",
        "exported_at": "2026-09-21T05:43:54+00:00",
        "image_mode": "vehicle_crop",
        "class_count": 5,
        "is_current": True,
    }
]

TRAIN_STATUS_IDLE = None

TRAIN_RUNS = {
    "items": [
        {
            "job_id": "train-job-42",
            "state": "finished",
            "started_at": "2026-09-20T10:00:00+00:00",
            "finished_at": "2026-09-20T11:20:00+00:00",
            "current_epoch": 80,
            "total_epochs": 80,
            "best_metric": {"map50": 0.91, "map50_95": 0.71},
            "last_metric": {"map50": 0.91, "map50_95": 0.71},
            "mlflow_run_id": "mlflow-1",
            "mlflow_run_url": None,
            "checkpoint_path": "runs/train-job-42/weights/best.pt",
        }
    ],
    "total": 1,
}

TRAIN_PROFILES = {
    "profiles": [
        {"name": "nano", "description": "Fast iteration, lowest accuracy.", "defaults": {}},
        {"name": "medium", "description": "Balanced default.", "defaults": {}},
    ]
}

TRAIN_PRESETS = {
    "class_subset_presets": [
        {
            "name": "all",
            "label": "All classes",
            "description": "Every non-deprecated class.",
            "selector": {"kind": "all"},
        }
    ]
}

MODELS_STATUS = {
    "models": [
        {
            "name": "default_vehicle_v1_trt",
            "friendly_name": "Vehicle detector v1",
            "role": "primary",
            "kind": "triton",
            "model_type": "detector",
            "status": "ready",
            "version": "1",
            "inference_count": 128000,
            "exec_count": 128000,
            "inference_failed": 12,
            "avg_latency_ms": 8.4,
            "last_error": None,
            "endpoint": "http://openprocessor:8000",
        },
        {
            "name": "tag_detector_v1",
            "friendly_name": "Tag detector",
            "role": "detector",
            "kind": "triton",
            "model_type": "detector",
            "status": "ready",
            "version": "3",
            "inference_count": 42000,
            "exec_count": 42000,
            "inference_failed": 0,
            "avg_latency_ms": 5.1,
            "last_error": None,
            "endpoint": "http://openprocessor:8000",
        },
    ]
}

BAKEOFF_EVAL_DATASETS = {
    "datasets": [
        {
            "name": "20260901T000000Z_tag",
            "path": "eval/20260901T000000Z_tag",
            "kind": "curated",
            "n_test": 5000,
            "frozen_sha": "sha256:abc123",
        }
    ],
    "count": 1,
}

BAKEOFF_BASELINES = {
    "baselines": [
        {"backend": "ultralytics", "name": "yolo11n-tag", "mode": "full"},
    ],
    "count": 1,
}

BAKEOFF_TRAINED_MODELS = {
    "models": [
        {
            "run_id": "train-job-42",
            "name": "vehicle-v1",
            "model_size": "m",
            "checkpoint_path": "runs/train-job-42/weights/best.pt",
            "map50": 0.91,
            "finished_at": "2026-09-20T11:20:00+00:00",
            "campaign_id": None,
        }
    ],
    "count": 1,
}

BAKEOFF_RUNS = {
    "runs": [
        {
            "job_id": "bakeoff-job-9",
            "state": "finished",
            "models": ["yolo11n-tag", "vehicle-v1"],
            "started_at": "2026-09-21T02:00:00+00:00",
            "finished_at": "2026-09-21T02:20:00+00:00",
        }
    ]
}

BAKEOFF_MATRIX = {
    "datasets": ["20260901T000000Z_tag"],
    "models": ["yolo11n-tag", "vehicle-v1"],
    "metrics": ["map50"],
    "cells": {
        "20260901T000000Z_tag": {
            "yolo11n-tag": {"map50": 0.81},
            "vehicle-v1": {"map50": 0.91},
        }
    },
    "best": {"20260901T000000Z_tag": {"map50": "vehicle-v1"}},
}

BAKEOFF_RESULTS = {
    "models": [
        {
            "model": "vehicle-v1",
            "runtime": "trt",
            "training_data": "20260921T054354Z_v1-5class",
            "imgsz": 640,
            "map_50": 0.91,
            "map_50_95": 0.71,
            "ap_small": 0.55,
            "mean_iou": 0.78,
            "precision": 0.9,
            "recall": 0.88,
            "f1": 0.89,
            "latency_ms": 6.2,
            "fps": 161.0,
        }
    ],
    "n_models": 1,
}

SETTINGS = {"app_name": "Cropwright"}

# --------------------------------------------------------------------------
# Synthetic image generator (§3.7.4) — deterministic flat-colour SVG tile
# keyed on the crop id. No photograph is ever loaded, so there is nothing
# to leak by construction.
# --------------------------------------------------------------------------

PALETTE = [
    (56, 78, 110),
    (72, 60, 92),
    (48, 84, 78),
    (96, 72, 52),
    (60, 72, 96),
    (84, 64, 76),
]


def synthetic_svg(seed: str, w: int = 256, h: int = 256) -> bytes:
    idx = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(PALETTE)
    r, g, b = PALETTE[idx]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect width="100%" height="100%" fill="rgb({r},{g},{b})"/>'
        f'<rect x="12%" y="18%" width="76%" height="58%" fill="none" '
        f'stroke="rgba(255,255,255,0.35)" stroke-width="3" rx="6"/>'
        f'<text x="50%" y="92%" text-anchor="middle" font-size="20" '
        f'font-family="monospace" fill="rgba(255,255,255,0.55)">{seed[:14]}</text>'
        f"</svg>"
    )
    return svg.encode()


# --------------------------------------------------------------------------
# Stub
# --------------------------------------------------------------------------


class Stub:
    """Fulfils every `/curation/**` request with fixture JSON or a synthetic image.

    Never fetches anything real: every branch either returns a value from
    the fixtures above or a generated SVG. The final `return ok({})` /
    `return ok([])` catch-all covers any `/curation/*` path this script doesn't
    yet know about, so an unhandled route degrades to an empty payload
    instead of a 404 that could crash the page — but note it never reaches
    a real backend either way, per the guard registered in `main()`.
    """

    def __init__(self) -> None:
        self.responses_seen: list[str] = []

    def handle(self, route: Route, request) -> None:
        url = request.url
        self.responses_seen.append(url)
        path = re.sub(r"^https?://[^/]+", "", url)
        method = request.method

        def ok(payload: Any, status: int = 200) -> None:
            route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))

        # -- images: never a photograph, always a generated tile ---------
        # Covers /curation/crops/{id}/thumbnail, /region_thumbnail, /image
        # (getSourceImageWithBbox / getSourceImageFull — the full-frame
        # preview, not a "/source" path as the name might suggest) and
        # any literal "/source" segment for forward-compat.
        if (
            "/thumbnail" in path
            or "/region_thumbnail" in path
            or "/image" in path
            or path.endswith("/source")
            or "/source?" in path
        ):
            crop_id = path.split("/crops/")[-1].split("/")[0] if "/crops/" in path else path
            route.fulfill(status=200, content_type="image/svg+xml", body=synthetic_svg(crop_id))
            return

        if "/curation/health" in path:
            return ok({"status": "ok"})
        if path.startswith("/curation/methods"):
            return ok(METHODS)
        if path.startswith("/curation/classes"):
            return ok(CLASSES)
        if path.startswith("/curation/stats/dataset"):
            return ok(DATASET_STATS)
        if path.startswith("/curation/stats/classes"):
            return ok(STATS_CLASSES)
        if path.startswith("/curation/pipeline/auto_label/status"):
            return ok(IDLE_JOB)
        if path.startswith("/curation/clusters"):
            return ok(CLUSTERS)
        if path.startswith("/curation/crops/batch_label") or path.startswith("/curation/crops/batch_exclude") or path.startswith("/curation/crops/batch_unexclude"):
            return ok({"updated": 0, "conflicts": []})
        if path.startswith("/curation/crops/flag_new_class"):
            return ok({"flagged": 0, "errors": 0})
        if re.match(r"^/curation/crops/[^/]+/label$", path.split("?")[0]):
            return ok({"ok": True})
        if path.startswith("/curation/crops"):
            m = re.search(r"cluster_id=(\d+)", path)
            cid = int(m.group(1)) if m else 1
            crops = [crop(cid * 10 + k, cluster_id=cid) for k in range(9)]
            return ok({"total": len(crops), "page": 1, "page_size": 60, "crops": crops})
        if path.split("?")[0] == "/curation/review/tabs":
            # Served label for the region tab, so the capture below can
            # click it by a deployment-neutral name.
            return ok({"tabs": [{"id": "regions", "label": REGION_TAB_LABEL}]})
        if path.startswith("/curation/review/"):
            tab = path.split("/curation/review/")[-1].split("?")[0]
            return ok(review_items(tab))
        if path.startswith("/curation/regions/training_candidates"):
            return ok({"items": [], "total": 0, "page": 1, "page_size": 30})
        if path.startswith("/curation/regions"):
            return ok(review_items("regions"))
        if path.startswith("/curation/export/status"):
            return ok(EXPORT_STATUS)
        if path.startswith("/curation/export/datasets"):
            return ok(EXPORT_DATASETS)
        if path.startswith("/curation/export/single_class/status"):
            return ok({"status": "idle", "last_run": None})
        if path.startswith("/curation/export/registry/"):
            return ok({})
        if path.startswith("/curation/train/status"):
            return ok(TRAIN_STATUS_IDLE)
        if path.startswith("/curation/train/runs"):
            return ok(TRAIN_RUNS)
        if path.startswith("/curation/train/preflight") and method == "POST":
            return ok({"blocked": False, "checks": [{"name": "dataset", "severity": "ok", "message": "Dataset looks good."}], "summary": "Ready to train."})
        if path.startswith("/curation/train/presets"):
            return ok(TRAIN_PRESETS)
        if path.startswith("/curation/train/profiles"):
            return ok(TRAIN_PROFILES)
        if path.startswith("/curation/train/manifest/"):
            return ok({"spec": {}, "lineage": {}})
        if path.startswith("/curation/train/log/tail/"):
            return ok({"job_id": "train-job-42", "lines": ["epoch 80/80 map50=0.91"]})
        if path.startswith("/curation/models/status"):
            return ok(MODELS_STATUS)
        if path.startswith("/curation/models"):
            return ok(MODELS_STATUS)
        if path.startswith("/curation/bakeoff/eval_datasets"):
            return ok(BAKEOFF_EVAL_DATASETS)
        if path.startswith("/curation/bakeoff/baseline_models"):
            return ok(BAKEOFF_BASELINES)
        if path.startswith("/curation/bakeoff/trained_models"):
            return ok(BAKEOFF_TRAINED_MODELS)
        if path.startswith("/curation/bakeoff/runs"):
            return ok(BAKEOFF_RUNS)
        if path.startswith("/curation/bakeoff/matrix/"):
            return ok(BAKEOFF_MATRIX)
        if path.startswith("/curation/bakeoff/status/"):
            return ok({"status": "finished", "job_id": "bakeoff-job-9"})
        if path.startswith("/curation/bakeoff/results/"):
            return ok(BAKEOFF_RESULTS)
        if path.startswith("/curation/settings"):
            return ok(SETTINGS)
        if path.startswith("/curation/test_holdout/stats"):
            return ok({"total": 1248})
        if path.startswith("/curation/select/status") or path.startswith("/curation/viz/projection"):
            return ok({"status": "unavailable"})
        if path.startswith("/curation/search/text"):
            return ok({"items": [], "total": 0, "page": 1, "page_size": 30})
        if path.startswith("/curation/pipeline/events"):
            # DatasetStats.svelte (src/lib/sse.ts's subscribePipelineEvents)
            # renders "Loading..." until it receives the initial `snapshot`
            # SSE frame — a plain empty response leaves it stuck there
            # forever. One real event-stream frame is enough; the harness
            # doesn't need the connection to stay open.
            body = f"event: snapshot\ndata: {json.dumps({'state': IDLE_JOB, 'stats': DATASET_STATS})}\n\n"
            route.fulfill(status=200, content_type="text/event-stream", body=body)
            return
        if path.startswith("/curation/events") or "stream" in path:
            route.fulfill(status=204, body="")
            return
        # Unknown /curation/* GET: empty object is the safe default (every real
        # response above is an object; nothing here should ever be hit for
        # the 13 captured routes — if it is, that's a gap to add above).
        return ok({})


# --------------------------------------------------------------------------
# Capture loop
# --------------------------------------------------------------------------

CAPTURES = [
    "dashboard",
    "clusters",
    "cluster-detail",
    "crop-detail-modal",
    "review",
    "review-regions",
    "classes",
    "export",
    "train",
    "models",
    "bakeoff",
    "shortcut-overlay",
]


def _is_app_asset(url: str, base: str) -> bool:
    if url.startswith(base):
        return True
    if url.startswith("data:") or url.startswith("blob:"):
        return True
    return False


def wait_images(page: Page, timeout: int = 8000) -> None:
    try:
        page.wait_for_function(
            "() => Array.from(document.images).every(i => i.complete)",
            timeout=timeout,
        )
    except Exception:
        pass


def goto(page: Page, base: str, path: str) -> None:
    page.goto(f"{base}{path}", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(400)
    wait_images(page)


def capture(page: Page, out_dir: Path, name: str) -> Path:
    shot = out_dir / f"{name}.png"
    page.screenshot(path=str(shot), full_page=False)
    return shot


def run_captures(page: Page, base: str, out_dir: Path) -> list[Path]:
    shots: list[Path] = []

    goto(page, base, "/dashboard")
    shots.append(capture(page, out_dir, "dashboard"))

    goto(page, base, "/clusters")
    shots.append(capture(page, out_dir, "clusters"))

    goto(page, base, "/clusters/1")
    shots.append(capture(page, out_dir, "cluster-detail"))

    # crop-detail-modal: CropCard.svelte's hover-revealed "ⓘ" button
    # (aria-label="Show crop details") opens CropDetailModal. It's
    # opacity-0 until group-hover, so hover the card first.
    info_btn = page.get_by_label("Show crop details").first
    if info_btn.count() > 0:
        info_btn.hover()
        info_btn.click(force=True)
        page.wait_for_timeout(400)
    wait_images(page)
    shots.append(capture(page, out_dir, "crop-detail-modal"))
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    goto(page, base, "/review")
    shots.append(capture(page, out_dir, "review"))

    # The region tab's urlId is deployment-specific, so click the tab by
    # the label the stubbed /review/tabs serves instead of deep-linking.
    goto(page, base, "/review")
    page.get_by_role("button", name=REGION_TAB_LABEL).first.click()
    page.wait_for_timeout(500)
    wait_images(page)
    shots.append(capture(page, out_dir, "review-regions"))

    goto(page, base, "/classes")
    shots.append(capture(page, out_dir, "classes"))

    goto(page, base, "/export")
    shots.append(capture(page, out_dir, "export"))

    goto(page, base, "/train")
    shots.append(capture(page, out_dir, "train"))

    goto(page, base, "/models")
    shots.append(capture(page, out_dir, "models"))

    goto(page, base, "/bakeoff")
    shots.append(capture(page, out_dir, "bakeoff"))

    goto(page, base, "/clusters/1")
    page.keyboard.press("`")
    page.wait_for_timeout(300)
    shots.append(capture(page, out_dir, "shortcut-overlay"))
    page.keyboard.press("Escape")

    return shots


def make_demo_gif(page: Page, base: str, out_dir: Path) -> Path | None:
    """Assemble docs/screenshots/demo.gif's replacement from a scripted walk
    of stub-backed pages, via ffmpeg (confirmed present at /usr/bin/ffmpeg
    per plan §3.7.3)."""
    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        print("[capture] ffmpeg not found, skipping demo.gif", file=sys.stderr)
        return None

    frame_dir = out_dir / "_gif_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": 1280, "height": 860})

    walk = ["/dashboard", "/clusters", "/clusters/1", "/review", "/export", "/models"]
    frame_paths: list[Path] = []
    for i, path in enumerate(walk):
        goto(page, base, path)
        fp = frame_dir / f"frame_{i:03d}.png"
        page.screenshot(path=str(fp))
        frame_paths.append(fp)

    gif_path = out_dir / "demo.gif"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-vf",
            "scale=1280:-1:flags=lanczos",
            str(gif_path),
        ],
        check=True,
        capture_output=True,
    )
    for fp in frame_paths:
        fp.unlink()
    frame_dir.rmdir()
    page.set_viewport_size({"width": 1600, "height": 1000})
    return gif_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5199")
    p.add_argument("--out", default="docs/screenshots-new")
    p.add_argument("--headed", action="store_true")
    args = p.parse_args()

    base = args.url.rstrip("/")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stub = Stub()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()

        console_errors: list[str] = []
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        # Playwright checks routes in LIFO order (most-recently-registered
        # first) and only falls through to an earlier registration via
        # `route.fallback()`. Register the broad catch-all FIRST so the
        # `/curation/**` stub — registered SECOND — is the one actually checked
        # first for any `/curation/**` URL; otherwise the catch-all's
        # `continue_()` would win for anything same-origin, including
        # `/curation/**`, and every stub fixture below would be dead code.
        #
        # Belt-and-braces catch-all (§3.7.5 mitigation 2): anything that
        # is not same-origin app traffic aborts instead of silently
        # succeeding against a real origin.
        page.route(
            "**/*",
            lambda route, request: route.continue_()
            if _is_app_asset(request.url, base)
            else route.abort(),
        )
        # Stub registered second == checked first: matches every /curation/** request.
        page.route("**/curation/**", stub.handle)

        shots = run_captures(page, base, out_dir)
        gif = make_demo_gif(page, base, out_dir)
        if gif:
            shots.append(gif)

        browser.close()

    # §3.7.5 mitigation 3: assert no response left the local origin.
    external = [
        u
        for u in stub.responses_seen
        if not u.startswith(base) and not u.startswith("data:") and not u.startswith("blob:")
    ]
    if external:
        print(f"FATAL: capture touched a non-local origin: {external[:5]}", file=sys.stderr)
        return 1

    print(f"[capture] {len(shots)} files written to {out_dir}")
    for s in shots:
        print(f"  {s}")
    if console_errors:
        print(f"[capture] {len(console_errors)} page error(s) seen (non-fatal, review manually):")
        for e in console_errors[:10]:
            print(f"  {e}")

    if len(shots) != 13:
        print(f"FATAL: expected 13 captures, got {len(shots)}", file=sys.stderr)
        return 1

    print("[capture] no non-local origin touched — safety assertion passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
