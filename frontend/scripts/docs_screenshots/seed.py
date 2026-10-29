"""Seed an isolated, empty stack with the public throw-away projects the seeded
docs screenshots read.

Everything goes through the curation API, as a user would do it: projects,
classes, dataset imports of public COCO images, a region profile, the real
detection worker (against the harness's stand-in segmenter and VLM), a
training run (against the harness's stand-in trainer) and the VLM switches. The
only direct writes are the files the stack mounts (the public images) and the
illustrative results of one finished run (``seed_docs_run_results.py``).

Safety: the stack must hold nothing but the ``default`` project (or already
the five seeded ones, in which case seeding is skipped). A stack with any other
project is refused, so this can never write beside someone's data.

Needs the stack from ``docker/test/compose.yml`` + ``compose.docs-shots.yml``
(no GPU, no Triton); see docs-site/docs/developer-guide/screenshots.mdx.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from docs_screenshots.ctx import (
    EDITORS_PROJECT,
    MAIN_PROJECT,
    PACK_NAME,
    PROFILE_NAME,
    SEEDED_PROJECTS,
    VEHICLES_PROFILE,
    VEHICLES_PROJECT,
    VLM_OFF_PROJECT,
    IMPORT_PROJECT,
)


API = "/curation"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "samples"
DEFAULT_SOURCE = REPO_ROOT / "docker" / "test" / "verify-data" / "source"
WHEEL_EXAMPLE = REPO_ROOT / "examples" / "region_profiles" / "vehicle_wheel.json"

# Public datasets, as `make sample-coco-import` and `make sample-coco-readme` fetch them,
# and where each is staged under the stack's /data/source.
STAGED = {"import_fixture": "import_fixture", "coco_va_readme": "coco_sample"}
VEHICLE_CLASSES = ("car", "truck", "bus")
# yolo variant of the import fixture: names that differ from the registry on purpose.
YOLO_TO_CLASS = {"Car": "car", "automobile": "car", "truck": "truck", "bus": "bus"}
TRAIN_CLASSES = ("car", "truck", "bus")

Call = Callable[[str, str, "dict[str, Any] | None"], "tuple[int, Any]"]


class SeedError(RuntimeError):
    """A seeding step the stack refused or could not finish."""


# ---------------------------------------------------------------- pure helpers


def stack_state(listing: dict[str, Any]) -> str:
    """``empty`` (only ``default``), ``seeded`` (all five seeded projects and nothing
    else besides ``default``); anything else is refused."""
    slugs = {p["slug"] for p in listing.get("projects", [])} - {"default"}
    if not slugs:
        return "empty"
    if slugs == set(SEEDED_PROJECTS):
        return "seeded"
    raise SeedError(
        f"refusing to seed: the stack holds projects {sorted(slugs - set(SEEDED_PROJECTS))} "
        "beside the seeded ones; seed only a stack started from empty"
    )


def import_mapping(preview: dict[str, Any], class_ids: dict[str, int], rename: dict[str, str]) -> list[dict[str, Any]]:
    """One ``map`` entry per dataset class of a preview: to the class ``rename`` names
    (default: the dataset class itself), by the id the project registry holds."""
    out = []
    for row in preview["classes"]:
        name = rename.get(row["dataset_class"], row["dataset_class"])
        out.append({"dataset_class": row["dataset_class"], "action": "map", "class_id": class_ids[name]})
    return out


def wheel_profile_body(example: dict[str, Any]) -> dict[str, Any]:
    """The example region profile as a create body: the document without its comment and name."""
    return {k: v for k, v in example.items() if k not in ("_comment", "name")}


def worker_processed(log: str) -> int:
    """``processed=N`` of the detection worker's final log line (0 when it printed none)."""
    found = re.findall(r"region_worker_done\s.*?processed=(\d+)", log)
    return int(found[-1]) if found else 0


def missing_data(samples: Path) -> list[str]:
    return [name for name in STAGED if not (samples / name).is_dir()]


# ------------------------------------------------------------------- HTTP I/O


def make_call(base: str) -> Call:
    def call(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            base.rstrip("/") + API + path,
            data=data,
            method=method,
            headers={"content-type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # nosec B310  the isolated stack's own URL
                raw = r.read()
                return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw.decode(errors="replace")[:300]}

    return call


def _ok(call: Call, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    status, out = call(method, path, body)
    if status >= 300:
        raise SeedError(f"{method} {path} -> {status}: {json.dumps(out)[:400]}")
    return out


def _wait(what: str, check: Callable[[], Any], seconds: int = 180) -> Any:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(1)
    raise SeedError(f"timed out waiting for {what}")


# ---------------------------------------------------------------------- steps


def stage_public_data(samples: Path, source: Path) -> None:
    gone = missing_data(samples)
    if gone:
        raise SeedError(
            f"public sample data missing under {samples}: {gone}. Fetch it first: "
            "`make sample-coco-import` and `make sample-coco-readme`"
        )
    source.mkdir(parents=True, exist_ok=True)
    for src, dest in STAGED.items():
        if not (source / dest).is_dir():
            shutil.copytree(samples / src, source / dest)


def create_project(call: Call, slug: str, display_name: str) -> None:
    _ok(call, "POST", "/projects", {"slug": slug, "display_name": display_name, "description": "Public COCO sample"})
    def active() -> bool:
        status, out = call("GET", f"/projects/{slug}")
        return status == 200 and out.get("project", out).get("status") in (None, "active")

    _wait(f"project {slug}", active)
    time.sleep(3)  # its indexes and directories finish a moment after it reads active


def create_classes(call: Call, slug: str, names: list[str], group: str) -> dict[str, int]:
    for name in names:
        _ok(call, "POST", f"/projects/{slug}/classes", {"name": name, "group": group})
    listed = _ok(call, "GET", f"/projects/{slug}/classes")["classes"]
    return {c["class_name"]: c["class_id"] for c in listed}


def run_import(
    call: Call, slug: str, source: dict[str, Any], mapping: list[dict[str, Any]], name: str,
    label_trust: str = "validated",
) -> None:
    started = _ok(
        call, "POST", f"/projects/{slug}/datasets/imports",
        {"source": source, "mapping": mapping, "options": {"name": name, "label_trust": label_trust}},
    )
    iid = started["import_id"]

    def done() -> dict[str, Any] | None:
        job = _ok(call, "GET", f"/projects/{slug}/datasets/imports/{iid}")
        return job if job["status"] not in ("queued", "running") else None

    job = _wait(f"import {name}", done, 300)
    if job["status"] != "completed":
        raise SeedError(f"import {name} ended {job['status']}: {job.get('error')}")


def activate_vlm(call: Call, slug: str) -> None:
    _ok(call, "POST", f"/projects/{slug}/vlm/endpoints/env/activate", {"expected_active": None})


def create_wheel_profile(call: Call, slug: str, name: str, activate: bool) -> None:
    body = wheel_profile_body(json.loads(WHEEL_EXAMPLE.read_text()))
    _ok(call, "POST", f"/projects/{slug}/region_profiles", {"name": name, "body": body})
    if activate:
        _ok(call, "POST", f"/projects/{slug}/region_profiles/{name}/activate", {"expected_active": None})


def run_worker(container: str, slug: str) -> None:
    """The real detection worker, in the API container, until a pass finds nothing queued."""
    for _ in range(10):
        out = subprocess.run(  # noqa: S603  # argv list, container name from the operator
            ["docker", "exec", container, "python", "-m", "scripts.curation.worker", "--project", slug,
             "--max-iterations", "200", "--poll-interval", "1"],
            capture_output=True, text=True, check=True, timeout=900,
        )
        if worker_processed(out.stdout + out.stderr) == 0:
            return
    raise SeedError(f"the detection worker still had work after 10 passes for {slug}")


def embed_stand_in(container: str, slug: str) -> None:
    """Stand-in vectors for the imported items (the encoder needs Triton); see
    ``scripts/curation/seed_docs_embeddings.py``. They only order the cluster grid."""
    subprocess.run(  # noqa: S603  # argv list, container name from the operator
        ["docker", "exec", container, "python", "scripts/curation/seed_docs_embeddings.py", "--project", slug],
        check=True, capture_output=True, text=True, timeout=300,
    )


def seed_vehicles(call: Call, container: str, source: str) -> None:
    slug = VEHICLES_PROJECT
    create_project(call, slug, "COCO vehicles")
    ids = create_classes(call, slug, list(VEHICLE_CLASSES), "vehicle")
    create_wheel_profile(call, slug, VEHICLES_PROFILE, activate=True)
    activate_vlm(call, slug)
    preview = _ok(call, "POST", f"/projects/{slug}/datasets/preview", {"source": {"path": source}})
    run_import(call, slug, {"path": source}, import_mapping(preview, ids, YOLO_TO_CLASS), "coco_vehicles")
    _ok(call, "POST", f"/projects/{slug}/reprocess",
        {"targets": {"filter": {"missing_status": True}}, "scopes": ["region"], "dry_run": False})
    run_worker(container, slug)


def seed_import(call: Call, container: str, source: str, jobs_root: Path) -> None:
    slug = IMPORT_PROJECT
    create_project(call, slug, "COCO import")
    ids = create_classes(call, slug, list(VEHICLE_CLASSES), "vehicle")
    preview = _ok(call, "POST", f"/projects/{slug}/datasets/preview", {"source": {"path": source}})
    run_import(call, slug, {"path": source}, import_mapping(preview, ids, YOLO_TO_CLASS), "coco_import_88")
    embed_stand_in(container, slug)
    seed_training_run(call, container, slug, jobs_root)


def seed_editors(call: Call) -> None:
    slug = EDITORS_PROJECT
    create_project(call, slug, "COCO editors")
    _ok(call, "POST", f"/projects/{slug}/prompt_packs/generic_item_v1/clone", {"new_name": PACK_NAME, "source": "builtin"})
    create_wheel_profile(call, slug, PROFILE_NAME, activate=False)


def seed_main(call: Call, container: str, coco_source: dict[str, Any]) -> None:
    slug = MAIN_PROJECT
    create_project(call, slug, "COCO sample")
    preview = _ok(call, "POST", f"/projects/{slug}/datasets/preview", {"source": coco_source})
    names = sorted(row["dataset_class"] for row in preview["classes"])
    ids = create_classes(call, slug, names, "coco")
    run_import(call, slug, coco_source, import_mapping(preview, ids, {}), "coco_sample_200", label_trust="suggestion")
    embed_stand_in(container, slug)
    activate_vlm(call, slug)


def seed_training_run(call: Call, container: str, slug: str, jobs_root: Path) -> None:
    """Export, start a run on the stand-in trainer, wait for it to finish, then fill in its
    illustrative results (see ``scripts/curation/seed_docs_run_results.py``)."""
    export = _ok(call, "POST", f"/projects/{slug}/export/yolo", {"version_tag": "docs-v1", "seed": 42})
    started = _ok(
        call, "POST", f"/projects/{slug}/train/start?force=true",
        {"dataset_export_dir": export["export_dir"], "model_size": "n", "profile": "probe", "cuda_visible_devices": "0"},
    )
    job_id = started["job_id"]
    _wait(
        "the stand-in trainer",
        lambda: _ok(call, "GET", f"/projects/{slug}/train/status/{job_id}")["state"] == "finished",
        120,
    )
    listed = _ok(call, "GET", f"/projects/{slug}/classes")["classes"]
    support = {c["class_name"]: c["sample_count"] for c in listed if c["kind"] == "item"}
    args = [f"--class={name}={count}" for name, count in support.items() if count]
    subprocess.run(  # noqa: S603  # argv list, container name from the operator
        ["docker", "exec", container, "python", "scripts/curation/seed_docs_run_results.py",
         "--project", slug, "--jobs-dir", str(jobs_root / "projects" / slug), "--job-id", job_id, *args],
        check=True, capture_output=True, text=True, timeout=120,
    )


def seed_vlm_off(call: Call) -> None:
    create_project(call, VLM_OFF_PROJECT, "COCO VLM off")
    _ok(call, "POST", f"/projects/{VLM_OFF_PROJECT}/vlm/endpoints/deactivate", {})


def seed(base: str, container: str, samples: Path = DEFAULT_SAMPLES, source: Path = DEFAULT_SOURCE,
         jobs_root: Path = Path("/jobs")) -> bool:
    """Seed the stack at ``base`` (the Cropwright or API origin). Returns False when it was
    already seeded. ``jobs_root`` is the trainer's jobs directory inside the API container."""
    call = make_call(base)
    state = stack_state(_ok(call, "GET", "/projects?include_archived=true"))
    if state == "seeded":
        print("  stack already seeded; skipping")
        return False
    stage_public_data(samples, source)
    coco = {
        "path": "/data/source/coco_sample",
        "format": "coco",
        "coco_annotations": [
            {"path": "/data/source/coco_sample/coco_gt.json", "images_dir": "/data/source/coco_sample/images"}
        ],
    }
    yolo = "/data/source/import_fixture/yolo"
    for step, run in (
        ("coco-vlm-off", lambda: seed_vlm_off(call)),
        ("coco-editors", lambda: seed_editors(call)),
        ("coco-import", lambda: seed_import(call, container, yolo, jobs_root)),
        ("coco-vehicles", lambda: seed_vehicles(call, container, yolo)),
        ("coco-sample", lambda: seed_main(call, container, coco)),
    ):
        print(f"  seeding {step}")
        run()
    return True
