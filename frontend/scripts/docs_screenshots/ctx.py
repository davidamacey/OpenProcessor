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


# The throwaway public projects docs_screenshots/seed.py creates on an isolated stack.
MAIN_PROJECT = "coco-sample"  # the 200-image COCO sample, class-labelled
VEHICLES_PROJECT = "coco-vehicles"  # COCO cars with a wheel region profile and worker-made boxes
IMPORT_PROJECT = "coco-import"  # the labeled-dataset import fixture, imported
EDITORS_PROJECT = "coco-editors"  # holds the editable prompt pack and region profile
VLM_OFF_PROJECT = "coco-vlm-off"  # VLM explicitly off, no run yet
SEEDED_PROJECTS = (MAIN_PROJECT, VEHICLES_PROJECT, IMPORT_PROJECT, EDITORS_PROJECT, VLM_OFF_PROJECT)
PACK_NAME = "docs_pack"
PROFILE_NAME = "docs_wheels"
VEHICLES_PROFILE = "coco_vehicle_wheels"
# Server path of the dataset the import wizard previews (under the stack's source root).
IMPORT_PREVIEW_PATH = "/data/source/import_fixture/yolo"

SHARE_OWNER = os.environ.get("SHARE_OWNER_PROJECT", "model-demo-owner")
SHARE_CONSUMER = os.environ.get("SHARE_CONSUMER_PROJECT", "model-demo-consumer")
SHARE_MODEL = os.environ.get("SHARE_MODEL", "model-demo-owner__model_demo_car")


# LIFE_PROJECT is a throwaway `cwlife-<epoch>-x` project (see states_region_embedding.py).
LIFE_PROJECT = os.environ.get("LIFE_PROJECT", "")


def _life() -> str:
    if not LIFE_PROJECT.startswith("cwlife-"):
        raise RuntimeError("set LIFE_PROJECT to the throwaway cwlife-<epoch>-x project")
    return LIFE_PROJECT


# Throwaway cwlife-* projects docs_screenshot_fixtures.py creates (see states_ingest.py).
BULK_RUN_PROJECT = os.environ.get("BULK_RUN_PROJECT", "")
BULK_FAIL_PROJECT = os.environ.get("BULK_FAIL_PROJECT", "")


def _cwlife(slug: str) -> str:
    if not slug.startswith("cwlife-"):
        raise RuntimeError(f"{slug!r} is not a throwaway cwlife-* project (set it from the fixtures output)")
    return slug
