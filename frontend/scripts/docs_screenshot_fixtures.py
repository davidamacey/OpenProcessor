#!/usr/bin/env python3
"""Throwaway fixtures for the docs screenshots that need a write.

The capture script (`capture_docs_screenshots.py`) is read-only by
construction. A few slots show a state that only exists after a write
(a combine job, a paused project, an editable prompt pack / region
profile). Those writes happen here, and ONLY against projects whose slug
starts with `cwlife-` (every write asserts the prefix first; a combine
may only read `default` or `cwlife-` sources and writes a `cwlife-`
target). `teardown` deletes every `cwlife-` project and waits for the
asynchronous delete to finish.

    python3 scripts/docs_screenshot_fixtures.py setup    --base-url http://localhost:5184
    python3 scripts/docs_screenshot_fixtures.py teardown --base-url http://localhost:5184

`bulk` and `vlm-off` set up the slots of the large-folder ingest guide and
the VLM-off state. They need a folder of PUBLIC images that the API container
can read (`--folder`, a container path) and the name of the API container
(`--container`, to write a manifest under the throwaway project's upload root):

    python3 scripts/docs_screenshot_fixtures.py bulk    --base-url http://localhost:5184 \\
        --folder /data/source/valid/images --container <api-container>
    python3 scripts/docs_screenshot_fixtures.py vlm-off --base-url http://localhost:5184
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PREFIX = "cwlife-"
API = "/curation"

# slug -> why it exists
SRC = "cwlife-src"  # empty combine source
PAUSED = "cwlife-paused"  # shows the paused chip on /projects
EDITORS = "cwlife-editors"  # holds the editable prompt pack + region profile
MERGED = "cwlife-merged"  # combine target
VLM_OFF = "cwlife-vlm-off"  # a project with its VLM explicitly deactivated
PACK_NAME = "cwlife_pack"
PROFILE_NAME = "cwlife_profile"


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + API + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw.decode(errors="replace")[:300]}


def assert_cwlife(slug: str) -> None:
    if not slug.startswith(PREFIX):
        sys.exit(f"refusing to write to non-{PREFIX} project {slug!r}")


def write(base: str, method: str, slug: str, path: str, body: dict | None = None) -> dict:
    """A write scoped to one project; asserts the throwaway prefix first."""
    assert_cwlife(slug)
    status, out = call(base, method, f"/projects/{slug}{path}", body)
    print(f"  {method} /projects/{slug}{path} -> {status}")
    if status >= 300:
        sys.exit(f"failed: {out}")
    return out


def create_project(base: str, slug: str, name: str) -> None:
    assert_cwlife(slug)
    status, out = call(
        base, "POST", "/projects", {"slug": slug, "display_name": name, "description": "Docs screenshot fixture"}
    )
    print(f"  POST /projects {slug} -> {status}")
    if status >= 300:
        sys.exit(f"failed: {out}")


def wait_active(base: str, slug: str) -> None:
    for _ in range(60):
        status, out = call(base, "GET", f"/projects/{slug}")
        if status == 200 and out.get("project", out).get("status") in (None, "active", "paused"):
            return
        time.sleep(1)


def setup(base: str) -> None:
    for slug, name in ((SRC, "cwlife-src"), (PAUSED, "cwlife-paused"), (EDITORS, "cwlife-editors")):
        create_project(base, slug, name)
        wait_active(base, slug)
    write(base, "POST", PAUSED, "/pause", {})
    write(base, "POST", EDITORS, f"/prompt_packs/generic_item_v1/clone", {"new_name": PACK_NAME, "source": "builtin"})
    write(
        base,
        "POST",
        EDITORS,
        "/region_profiles/license_plate/clone",
        {"new_name": PROFILE_NAME, "source": "registered"},
    )
    # Combine only `default` / `cwlife-` sources into a `cwlife-` target.
    sources = [{"project": "default"}, {"project": SRC}]
    assert all(s["project"] == "default" or s["project"].startswith(PREFIX) for s in sources)
    assert_cwlife(MERGED)
    body = {"sources": sources, "target": {"slug": MERGED, "display_name": "cwlife-merged"}}
    status, prev = call(base, "POST", "/projects/combine/preview", body)  # read-only
    if status >= 300 or not prev.get("ok"):
        sys.exit(f"combine preview refused: {json.dumps(prev)[:300]}")
    status, out = call(
        base, "POST", "/projects/combine", {**body, "expected_preview_sha": prev["preview_sha"]}
    )
    print(f"  POST /projects/combine -> {status} {json.dumps(out)[:200]}")
    if status >= 300:
        sys.exit("combine failed")


def _ensure(base: str, slug: str) -> None:
    assert_cwlife(slug)
    status, _ = call(base, "GET", f"/projects/{slug}")
    if status == 404:
        create_project(base, slug, slug)
    wait_active(base, slug)


def bulk(base: str, folder: str, container: str) -> None:
    """Start a folder run (left running for the screenshot) and a manifest run
    with two missing paths (so the failures list has rows). Both go to fresh
    projects (a deleted slug is retired for good, and re-ingesting stored images
    is all duplicates and ends in a second). Run the capture right after: a
    folder of a few hundred images takes about half a minute."""
    stamp = int(time.time())
    BULK_RUN = f"cwlife-bulk-run-{stamp}"
    BULK_FAIL = f"cwlife-bulk-fail-{stamp}"
    _ensure(base, BULK_RUN)
    write(base, "POST", BULK_RUN, "/ingest/bulk", {"folder": folder, "batch_size": 8, "concurrency": 2})  # slow enough to catch mid-run
    _ensure(base, BULK_FAIL)
    root = f"/var/lib/openprocessor/projects/{BULK_FAIL}/uploads"
    listing = subprocess.run(
        ["docker", "exec", container, "sh", "-c", f"ls {folder} | head -n 10"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    lines = [f"{folder}/{n}" for n in listing] + [f"{folder}/missing_1.jpg", f"{folder}/missing_2.jpg"]
    subprocess.run(
        ["docker", "exec", "-i", container, "sh", "-c", f"mkdir -p {root} && cat > {root}/manifest.txt"],
        check=True, input="\n".join(lines) + "\n", text=True,
    )
    write(base, "POST", BULK_FAIL, "/ingest/bulk", {"manifest": f"{root}/manifest.txt"})
    print(f"export BULK_RUN_PROJECT={BULK_RUN} BULK_FAIL_PROJECT={BULK_FAIL}")


def vlm_off(base: str) -> None:
    _ensure(base, VLM_OFF)
    write(base, "POST", VLM_OFF, "/vlm/endpoints/deactivate", {})


def teardown(base: str) -> None:
    status, out = call(base, "GET", "/projects?include_archived=true")
    slugs = [p["slug"] for p in out.get("projects", []) if p["slug"].startswith(PREFIX)]
    for slug in slugs:
        assert_cwlife(slug)
        # A paused project may need resuming? Deleting is guarded by a typed
        # confirmation equal to the slug.
        status, out = call(base, "DELETE", f"/projects/{slug}?confirm={slug}")
        print(f"  DELETE /projects/{slug} -> {status} {'' if status < 300 else json.dumps(out)[:300]}")
    for _ in range(120):
        status, out = call(base, "GET", "/projects?include_archived=true")
        left = [p["slug"] for p in out.get("projects", []) if p["slug"].startswith(PREFIX)]
        if not left:
            print("no cwlife-* project remains")
            return
        time.sleep(2)
    sys.exit(f"still present after waiting: {left}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["setup", "teardown", "bulk", "vlm-off"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--folder", help="bulk: a container path holding public images")
    ap.add_argument("--container", help="bulk: the API container name")
    a = ap.parse_args()
    if a.command == "bulk":
        if not (a.folder and a.container):
            sys.exit("bulk needs --folder and --container")
        bulk(a.base_url, a.folder, a.container)
    elif a.command == "vlm-off":
        vlm_off(a.base_url)
    else:
        (setup if a.command == "setup" else teardown)(a.base_url)


if __name__ == "__main__":
    main()
