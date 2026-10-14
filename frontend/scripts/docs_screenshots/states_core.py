"""Scripted capture states for projects, combine, reprocess, prompt packs,
region profiles, the VLM endpoint editor, multi-box review, import and model
sharing."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from urllib.parse import urlsplit

from docs_screenshots.ctx import (
    IMPORT_PROJECT,
    SHARE_MODEL,
    SHARE_OWNER,
    VEHICLES_PROJECT,
    Ctx,
    _settle,
    api_get,
)
from docs_screenshots.guard import _read_only


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
