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
        --base-url http://localhost:<port-of-a-public-data-instance>
    # or: CROPWRIGHT_URL=... e2e/.venv/bin/python scripts/capture_docs_screenshots.py

The capture is read-only: every non-GET/HEAD request is aborted, so a
screenshot pass can never write to the backend it points at.

Besides the routes it captures a set of STATE slots (a dialog opened, a crop
selected, a pack test run) with scripted navigation — see STATES below. A
state only ever opens and looks: nothing is saved, submitted or confirmed.

Read-only guard: GET/HEAD pass; every other request is aborted EXCEPT the
small explicit allow-list in READ_ONLY_CALLS (calls the UI makes only to
render a report/dialog, each of which writes nothing server-side). Keep
docs-site/docs/developer-guide/screenshots.md's list in step with it.

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
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTES_FILE = REPO_ROOT / "docs-site" / "src" / "data" / "screenshot_routes.json"
OUT_DIR = REPO_ROOT / "docs-site" / "static" / "img" / "screenshots"

WIDTHS = (1600, 800)
VIEWPORT_HEIGHT = 1000
# States whose page scrolls inside the app shell need a taller viewport to show it all.
STATE_HEIGHT = {"region-profile-test": 1600, "import-wizard": 2250, "import-job": 1500, "prompt-pack-editor": 1500, "prompt-pack-test": 1500, "region-profile-editor": 2300}


def load_routes() -> list[dict]:
    if not ROUTES_FILE.exists():
        sys.exit(f"missing route list: {ROUTES_FILE}")
    return json.loads(ROUTES_FILE.read_text())


# The ONLY non-GET/HEAD calls the capture lets through. Each one is
# side-effect-free (a dry run, a validation, a preview or a test that writes
# nothing); anything else is aborted. (method, path regex, why, optional
# predicate over (query string, request body text)).
READ_ONLY_CALLS: list[tuple[str, re.Pattern[str], str, object]] = [
    ("POST", re.compile(r"/train/preflight$"), "preflight report", None),
    (
        "DELETE",
        re.compile(r"/projects/[^/]+$"),
        "project delete DRY RUN (dry_run=true, no confirm) shown in the delete dialog",
        lambda q, body: "dry_run=true" in q and "confirm=" not in q,
    ),
    ("POST", re.compile(r"/projects/combine/preview$"), "combine preview report", None),
    (
        "POST",
        re.compile(r"/projects/[^/]+/reprocess$"),
        "multi-item Reprocess 'Check what would run' (body dry_run: true)",
        lambda q, body: '"dry_run":true' in body.replace(" ", ""),
    ),
    ("POST", re.compile(r"/prompt_packs/validate$"), "prompt-pack validation report", None),
    (
        "POST",
        re.compile(r"/prompt_packs/test$"),
        "prompt-pack 'Test on a crop' (runs the VLM on stored crops, 'Nothing is written')",
        None,
    ),
    ("POST", re.compile(r"/region_profiles/validate$"), "region-profile validation report", None),
    (
        "POST",
        re.compile(r"/region_profiles/test$"),
        "region-profile 'Test on a crop' (runs the detector/segmenter on stored crops, writes nothing)",
        None,
    ),
    (
        "POST",
        re.compile(r"/datasets/preview$"),
        "import wizard preview (dry run: 'writes nothing', dataset problems come back as issues)",
        None,
    ),
    ("POST", re.compile(r"/keymap/validate$"), "keymap validation report", None),
    ("POST", re.compile(r"/vlm/endpoints/validate$"), "VLM endpoint validation report", None),
]


def _allowed(method: str, url: str, body: str) -> bool:
    if method in ("GET", "HEAD"):
        return True
    parts = urlsplit(url)
    for m, pattern, _why, pred in READ_ONLY_CALLS:
        if method == m and pattern.search(parts.path) and (pred is None or pred(parts.query, body)):
            return True
    return False


def _read_only(route) -> None:
    req = route.request
    if _allowed(req.method, req.url, req.post_data or ""):
        route.continue_()
    else:
        print(f"  blocked {req.method} {req.url}")
        route.abort()


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


def _open_projects(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/projects", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("table", timeout=10_000)
    _settle(page)


def state_projects_delete_dry_run(page, ctx: Ctx) -> None:
    """The delete dialog's dry-run report WITH a blocking reason.

    A real blocking reason (`project_busy`) only exists while a job runs, so
    this starts a ~3 s auto-label on the throwaway `cwlife-src` project (via
    the fixtures module, which asserts the `cwlife-` prefix; the browser
    itself stays guarded) and clicks Delete inside that window. Without the
    fixtures it falls back to the plain report on the sample project.
    """
    _open_projects(page, ctx)
    slug = ctx.project
    busy = False
    if page.locator("tr", has_text="cwlife-src").count() > 0:
        import docs_screenshot_fixtures as fx  # scripts/ is on sys.path

        slug = fx.SRC
        fx.write(ctx.base, "POST", slug, "/pipeline/auto_label/start", {})
        busy = True
        for _ in range(40):  # the job reads as busy a moment after it is accepted
            _, rep = fx.call(ctx.base, "DELETE", f"/projects/{slug}?dry_run=true")
            if "project_busy" in rep.get("blocking", []):
                break
            time.sleep(0.1)
    row = page.locator("tr", has_text=slug).first
    row.get_by_role("button", name="Delete").click()
    page.wait_for_selector("[data-testid='delete-project-report']", timeout=15_000)
    if busy:
        page.wait_for_selector("[data-testid='delete-project-blocking']", timeout=5_000)
    _settle(page, 600)


def state_projects_copy_settings(page, ctx: Ctx) -> None:
    _open_projects(page, ctx)
    row = page.locator("tr", has_text=ctx.project).first
    row.get_by_role("button", name="Copy settings").click()
    page.wait_for_selector("[data-testid='clone-settings-dialog']", timeout=10_000)
    _settle(page, 800)


def state_combine_wizard(page, ctx: Ctx) -> None:
    # Two real sources and a target slug that does not exist: the preview is a
    # read-only report (allow-listed); "Start combine" is never clicked.
    page.goto(f"{ctx.base}/projects/combine", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("text=Add a source project", timeout=10_000)
    picker = page.locator("xpath=//span[normalize-space()='Add a source project']/following::select[1]")
    for slug in (ctx.project, "default"):
        picker.select_option(value=slug)
        page.wait_for_timeout(600)
    page.get_by_label("Target slug").fill("sample-coco-combined")
    page.get_by_label("Display name").fill("Sample COCO combined")
    page.wait_for_selector("[data-testid='combine-preview']", timeout=20_000)
    _settle(page, 1500)


def state_combine_job(page, ctx: Ctx) -> None:
    job = os.environ.get("COMBINE_JOB_ID")
    if not job:
        raise RuntimeError("set COMBINE_JOB_ID to the cwlife- combine job (docs_screenshot_fixtures.py setup)")
    page.goto(f"{ctx.base}/projects/combine/{job}", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("main", timeout=10_000)
    _settle(page)


def state_reprocess_dialog(page, ctx: Ctx) -> None:
    cl = ctx.api("/clusters?page_size=40")
    cid = next(
        c["cluster_id"]
        for c in cl.get("clusters", cl.get("items", []))
        if c.get("cluster_id", -1) >= 0 and 12 <= c.get("size", 0) <= 20  # >20 items: the server demands one scope at a time
    )
    page.goto(ctx.url(f"/clusters/{cid}"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='cluster-header-counts']", timeout=15_000)
    _settle(page)
    page.keyboard.press("a")  # select all on the page
    page.wait_for_timeout(500)
    page.locator("[data-testid='reprocess-open']").first.click()
    page.wait_for_selector("[role='dialog'][aria-label='Reprocess']", timeout=10_000)
    scopes = page.locator("[role='dialog'][aria-label='Reprocess'] input[type=checkbox]")
    scopes.nth(0).check()  # choosing scopes only enables the dry-run button
    scopes.nth(2).check()
    page.get_by_role("button", name="Check what would run").click()
    page.wait_for_selector("[data-testid='reprocess-dry-run']", timeout=30_000)
    _settle(page, 800)


def state_prompt_pack_editor(page, ctx: Ctx) -> None:
    pack = os.environ.get("PACK_NAME", "cwlife_pack")
    page.goto(
        f"{ctx.base}/p/cwlife-editors/settings/prompt-packs/{pack}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("textarea", timeout=15_000)
    # Break the required placeholder in a TEXT AREA ONLY (never saved) so the
    # live validation shows a real issue.
    ta = page.locator("textarea").nth(1)
    ta.fill("Class names are listed elsewhere. Label each numbered crop.")
    page.wait_for_selector("[data-testid='config-issue']", timeout=15_000)
    _settle(page, 800)


def state_prompt_pack_test(page, ctx: Ctx) -> None:
    ids = [c["crop_id"] for c in ctx.api("/crops?page_size=3")["crops"][:3]]
    page.goto(
        ctx.url("/settings/prompt-packs/generic_item_v1"),
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='pack-test-panel']", timeout=15_000)
    page.locator("[data-testid='test-crop-ids']").fill(", ".join(ids))
    page.locator("[data-testid='test-run']").click()
    page.wait_for_selector("[data-testid='test-result']", timeout=120_000)
    page.locator("[data-testid='pack-test-panel']").scroll_into_view_if_needed()
    _settle(page, 800)


def state_region_profile_editor(page, ctx: Ctx) -> None:
    prof = os.environ.get("PROFILE_NAME", "cwlife_profile")
    page.goto(
        f"{ctx.base}/p/cwlife-editors/settings/region-profiles/{prof}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("select, input", timeout=15_000)
    _settle(page)


def state_vlm_endpoint_editor(page, ctx: Ctx) -> None:
    page.goto(ctx.url("/settings/models/vlm/env"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("text=API key", timeout=15_000)
    _settle(page)


# Fixed public-sample projects the newer states read from (override via env).
VEHICLES_PROJECT = os.environ.get("VEHICLES_PROJECT", "sample-coco-vehicles")
IMPORT_PROJECT = os.environ.get("IMPORT_PROJECT", "sample-coco-import")


def _multibox_crop_id(base: str) -> str:
    """A wheel item whose boxes are in different states (accepted + rejected)."""
    d = api_get(base, f"/curation/projects/{VEHICLES_PROJECT}/review/regions?page_size=60")
    best = None
    for it in d["items"]:
        states = {b.get("state") for b in it["region_boxes"]}
        if len(it["region_boxes"]) >= 3 and {"accepted", "rejected"} <= states:
            return it["id"]
        if best is None and len(it["region_boxes"]) >= 3:
            best = it["id"]
    if best is None:
        raise RuntimeError("no multi-box item found")
    return best


def _open_multibox_review(page, ctx: Ctx) -> None:
    cid = _multibox_crop_id(ctx.base)
    page.goto(
        f"{ctx.base}/p/{VEHICLES_PROJECT}/review?tab=regions&crop_id={cid}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("text=/\\d+ \\/ \\d+ max/", timeout=20_000)
    _settle(page, 2500)


def state_review_regions_multibox(page, ctx: Ctx) -> None:
    _open_multibox_review(page, ctx)


def state_box_editor(page, ctx: Ctx) -> None:
    _open_multibox_review(page, ctx)
    page.get_by_role("button", name="Edit boxes").click()  # opens edit mode; nothing is saved
    page.wait_for_selector("[data-testid='multibox-canvas']", timeout=10_000)
    page.keyboard.press("Tab")  # select a box
    _settle(page, 800)


def state_region_profile_test(page, ctx: Ctx) -> None:
    ids = [
        api_get(ctx.base, f"/curation/projects/{VEHICLES_PROJECT}/review/regions?page_size=1")[
            "items"
        ][0]["id"]
    ]  # the profile test takes exactly one crop id
    page.goto(
        f"{ctx.base}/p/{VEHICLES_PROJECT}/settings/region-profiles/coco_vehicle_wheels",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='profile-test-panel']", timeout=15_000)
    page.locator("[data-testid='profile-test-crop-id']").fill(", ".join(ids))
    page.locator("[data-testid='test-run']").click()
    page.wait_for_selector("[data-testid='test-result'], [data-testid='test-error']", timeout=180_000)
    if page.locator("[data-testid='test-error']").count():
        raise RuntimeError(page.locator("[data-testid='test-error']").inner_text())
    page.locator("[data-testid='profile-test-panel']").scroll_into_view_if_needed()
    _settle(page, 800)


def state_import_wizard(page, ctx: Ctx) -> None:
    path = os.environ["IMPORT_PREVIEW_PATH"]  # a server path under an allowed root
    page.goto(
        f"{ctx.base}/p/{IMPORT_PROJECT}/datasets/import", wait_until="domcontentloaded", timeout=30_000
    )
    page.get_by_placeholder("/data/source/my_dataset").fill(path)
    page.wait_for_selector("text=Class mapping", timeout=60_000)
    page.get_by_label("Accept suggestions").check()  # client-side choice only
    _settle(page, 1500)


def state_import_job(page, ctx: Ctx) -> None:
    jobs = api_get(ctx.base, f"/curation/projects/{IMPORT_PROJECT}/datasets/imports")["items"]
    jid = next(j["import_id"] for j in jobs if j["status"].startswith("completed"))
    page.goto(
        f"{ctx.base}/p/{IMPORT_PROJECT}/datasets/imports/{jid}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='import-job']", timeout=15_000)
    _settle(page, 1200)


def state_review_imported(page, ctx: Ctx) -> None:
    jobs = api_get(ctx.base, f"/curation/projects/{IMPORT_PROJECT}/datasets/imports")["items"]
    jid = next(j["import_id"] for j in jobs if j["status"].startswith("completed"))
    page.goto(
        f"{ctx.base}/p/{IMPORT_PROJECT}/review?tab=imported&import_id={jid}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("text=/Import imp_/", timeout=20_000)
    _settle(page, 2500)


SHARE_OWNER = os.environ.get("SHARE_OWNER_PROJECT", "model-demo-owner")
SHARE_CONSUMER = os.environ.get("SHARE_CONSUMER_PROJECT", "model-demo-consumer")
SHARE_MODEL = os.environ.get("SHARE_MODEL", "model-demo-owner__model_demo_car")


def state_models_sharing(page, ctx: Ctx) -> None:
    """The owner project's view of its model shared with other projects.

    The consumer's view (with the "from <project>" chip and class mapping) is not
    captured: the backend currently lists the shared model twice there and the
    page's keyed list throws `each_key_duplicate`, so it never leaves "Loading...".
    """
    page.goto(f"{ctx.base}/p/{SHARE_OWNER}/models", wait_until="domcontentloaded", timeout=30_000)
    card = page.locator(f"[data-testid='model-sharing-{SHARE_MODEL}']").first
    card.wait_for(timeout=90_000)
    card.scroll_into_view_if_needed()
    _settle(page, 800)


def state_models_unshare_force(page, ctx: Ctx) -> None:
    """Owner clicks Stop sharing and confirms; the server must answer 409 in_use.

    The one deliberate exception to the read-only guard: this attempts the
    plain (non-force) unshare PUT on the owner's model. It is expected to be
    refused with 409 `in_use` (the consumer's active profile uses the model), so
    nothing changes. The handler below allows ONLY that URL, ONLY without a
    `force` query param, ONLY with {shared: false}; anything else is aborted.
    If the server does not answer 409 the unshare took effect, so the share is
    restored at once and the state fails loudly. "Unshare anyway" is never armed.
    """
    path_re = re.compile(rf"/projects/{re.escape(SHARE_OWNER)}/models/{re.escape(SHARE_MODEL)}/sharing$")
    log: list[tuple[str, str, int | None]] = []
    outcome: dict = {}

    def handler(route) -> None:
        req = route.request
        parts = urlsplit(req.url)
        if req.method == "PUT" and path_re.search(parts.path):
            body = req.post_data or ""
            if "force" in parts.query.lower() or "force" in body.lower() or '"shared":false' not in body.replace(" ", ""):
                print(f"  blocked (force/unexpected) {req.method} {req.url} {body}")
                route.abort()
                return
            resp = route.fetch()
            log.append((f"PUT {parts.path}?{parts.query} body={body}", "", resp.status))
            print(f"  unshare attempt: PUT {parts.path}?{parts.query} {body} -> {resp.status}")
            if resp.status != 409:
                outcome["unexpected"] = resp.status
            route.fulfill(response=resp)
            return
        _read_only(route)

    page.route("**/*", handler)
    page.goto(f"{ctx.base}/p/{SHARE_OWNER}/models", wait_until="domcontentloaded", timeout=30_000)
    toggle = page.locator(f"[data-testid='model-share-toggle-{SHARE_MODEL}']")
    toggle.wait_for(timeout=90_000)
    toggle.click()
    page.wait_for_selector("[data-testid='share-model-dialog']", timeout=10_000)
    page.locator("[data-testid='share-model-confirm']").click()
    try:
        page.wait_for_selector("[data-testid='share-model-in-use']", timeout=15_000)
    finally:
        if outcome.get("unexpected"):
            cur = api_get(ctx.base, f"/curation/projects/{SHARE_OWNER}/models/status")
            m = next(x for x in cur["models"] if x["name"] == SHARE_MODEL)
            print(f"!!! UNSHARE ANSWERED {outcome['unexpected']}: model shared={m['shared']}; restoring")
            if not m["shared"]:
                data = json.dumps({"shared": True, "expected_revision": m["sharing_revision"]}).encode()
                rq = urllib.request.Request(
                    f"{ctx.base}/curation/projects/{SHARE_OWNER}/models/{SHARE_MODEL}/sharing",
                    data=data, method="PUT", headers={"Content-Type": "application/json"},
                )
                print("!!! restore ->", urllib.request.urlopen(rq, timeout=30).status)
    if outcome.get("unexpected"):
        raise RuntimeError(f"unshare answered {outcome['unexpected']}, not 409")
    _settle(page, 600)


def state_vlm_run_picker(page, ctx: Ctx) -> None:
    """The assist bar expanded with the per-run VLM select opened; no run is started."""
    page.goto(f"{ctx.base}/p/{SHARE_OWNER}/dashboard", wait_until="domcontentloaded", timeout=30_000)
    page.get_by_role("button", name=re.compile("assist:")).first.click()
    page.wait_for_selector("[data-testid='vlm-run-picker']", timeout=15_000)
    page.get_by_role("heading", name="Dashboard").click()  # close the class list
    # A native select popup is not part of a screenshot: show its options inline
    # (display only; nothing is picked, no run is started).
    page.locator("[data-testid='vlm-run-select']").evaluate(
        "el => { el.size = el.options.length; }"
    )
    page.keyboard.press("Escape")
    page.locator("[data-testid='vlm-run-picker']").scroll_into_view_if_needed()
    _settle(page, 800)


# name -> function; every state is captured at 1600px only.
STATES = {
    "projects-delete-dry-run": state_projects_delete_dry_run,
    "projects-copy-settings": state_projects_copy_settings,
    "combine-wizard": state_combine_wizard,
    "combine-job": state_combine_job,
    "reprocess-dialog": state_reprocess_dialog,
    "prompt-pack-editor": state_prompt_pack_editor,
    "prompt-pack-test": state_prompt_pack_test,
    "region-profile-editor": state_region_profile_editor,
    "vlm-endpoint-editor": state_vlm_endpoint_editor,
    "review-regions-multibox": state_review_regions_multibox,
    "box-editor": state_box_editor,
    "region-profile-test": state_region_profile_test,
    "import-wizard": state_import_wizard,
    "import-job": state_import_job,
    "review-imported": state_review_imported,
    "models-sharing": state_models_sharing,
    "models-unshare-force": state_models_unshare_force,
    "vlm-run-picker": state_vlm_run_picker,
}


def scoped_route(route: str, project: str | None) -> str:
    """Bare per-project paths get the `/p/<project>` prefix; `/projects*` is global."""
    if project is None or route.startswith("/projects"):
        return route
    return f"/p/{project}{route}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CROPWRIGHT_URL"),
        required="CROPWRIGHT_URL" not in os.environ,
        help="Base URL of a Cropwright instance pointed at a PUBLIC-sample-data "
        "OpenProcessor backend only. Never a real deployment.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("CROPWRIGHT_PROJECT"),
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

    def finish(page, name: str, width: int) -> None:
        out_path = args.out / f"{name}-{width}.png"
        page.screenshot(path=str(out_path), full_page=True)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

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
                    fn(page, ctx)
                    finish(page, name, 1600)
                except Exception as e:  # keep going; a failed state is reported, not faked
                    print(f"  FAILED state {name}: {type(e).__name__}: {str(e)[:300]}")
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
