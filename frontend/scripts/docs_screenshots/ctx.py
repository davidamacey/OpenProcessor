"""What a capture state needs: the target, the project it reads from, the
fixed public-sample projects, and a settle wait."""

from __future__ import annotations

import json
import os
import urllib.request


def api_get(base: str, path: str) -> dict:
    """A plain GET against the API (used only to look up ids for the states)."""
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


class Ctx:
    """What a state function needs: the target and the project to read from."""

    def __init__(self, base: str, project: str) -> None:
        self.base = base.rstrip("/")
        self.project = project

    def url(self, path: str) -> str:
        return f"{self.base}/p/{self.project}{path}"

    def api(self, path: str) -> dict:
        return api_get(self.base, f"/curation/projects/{self.project}{path}")


def _settle(page, ms: int = 1500) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


# Fixed public-sample projects the newer states read from (override via env).
VEHICLES_PROJECT = os.environ.get("VEHICLES_PROJECT", "sample-coco-vehicles")
IMPORT_PROJECT = os.environ.get("IMPORT_PROJECT", "sample-coco-import")

SHARE_OWNER = os.environ.get("SHARE_OWNER_PROJECT", "model-demo-owner")
SHARE_CONSUMER = os.environ.get("SHARE_CONSUMER_PROJECT", "model-demo-consumer")
SHARE_MODEL = os.environ.get("SHARE_MODEL", "model-demo-owner__model_demo_car")


# MAIN_PROJECT holds the full public COCO sample (VLM labels, embeddings); LIFE_PROJECT is
# a throwaway `cwlife-<epoch>-x` project (see states_region_embedding.py).
MAIN_PROJECT = os.environ.get("MAIN_PROJECT", "sample-coco-2k-v3")
LIFE_PROJECT = os.environ.get("LIFE_PROJECT", "")


def _life() -> str:
    if not LIFE_PROJECT.startswith("cwlife-"):
        raise RuntimeError("set LIFE_PROJECT to the throwaway cwlife-<epoch>-x project")
    return LIFE_PROJECT


# Throwaway cwlife-* projects docs_screenshot_fixtures.py creates (see states_ingest.py).
BULK_RUN_PROJECT = os.environ.get("BULK_RUN_PROJECT", "")
BULK_FAIL_PROJECT = os.environ.get("BULK_FAIL_PROJECT", "")
VLM_OFF_PROJECT = os.environ.get("VLM_OFF_PROJECT", "cwlife-vlm-off")


def _cwlife(slug: str) -> str:
    if not slug.startswith("cwlife-"):
        raise RuntimeError(f"{slug!r} is not a throwaway cwlife-* project (set it from the fixtures output)")
    return slug
