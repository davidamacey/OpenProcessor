"""Scripted capture states for the large-folder ingest guide, pipeline modes,
audit and the settings lists."""

from __future__ import annotations

from docs_screenshots.ctx import (
    BULK_FAIL_PROJECT,
    BULK_RUN_PROJECT,
    VLM_OFF_PROJECT,
    Ctx,
    _cwlife,
    _settle,
)

# --- Slots for the large-folder ingest guide, pipeline modes, audit and the settings lists ----
# Read-only pages use ctx.project (a public sample project). The slots that need a run in
# progress, a failed run or a deactivated VLM read throwaway cwlife-* projects that
# docs_screenshot_fixtures.py (`bulk`, `vlm-off`) creates; the browser itself never writes.


def _bulk_section(page, ctx: Ctx, slug: str, wait: str):
    page.goto(f"{ctx.base}/p/{slug}/ingest", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector(wait, timeout=30_000)
    panel = page.locator("[data-testid='bulk-ingest']")
    panel.scroll_into_view_if_needed()
    _settle(page, 1500)
    return panel.locator("xpath=..")  # the card around the panel: heading included


def state_ingest_bulk_panel(page, ctx: Ctx):
    # A throwaway project with no run yet, so the panel shows only the roots and the form.
    return _bulk_section(page, ctx, _cwlife(VLM_OFF_PROJECT), "[data-testid='bulk-roots']")


def state_ingest_bulk_run(page, ctx: Ctx):
    # Start the run just before capturing: docs_screenshot_fixtures.py bulk
    sec = _bulk_section(page, ctx, _cwlife(BULK_RUN_PROJECT), "[data-testid='bulk-run']")
    page.locator("[data-testid='bulk-run'] summary", has_text="Where the time went").click()
    page.wait_for_timeout(800)
    return sec


def state_ingest_bulk_failures(page, ctx: Ctx):
    return _bulk_section(page, ctx, _cwlife(BULK_FAIL_PROJECT), "[data-testid='bulk-failures']")


def state_ingest_policy(page, ctx: Ctx) -> None:
    page.goto(ctx.url("/settings/ingest-policy"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("main", timeout=20_000)
    _settle(page, 2000)


def state_vlm_endpoint_off(page, ctx: Ctx):
    # Crop to the active-VLM card: the endpoint table below lists every project of the
    # deployment under "Used by", which must not reach a screenshot.
    page.goto(
        f"{ctx.base}/p/{_cwlife(VLM_OFF_PROJECT)}/settings/models", wait_until="domcontentloaded", timeout=30_000
    )
    page.wait_for_selector("section#active", timeout=20_000)
    _settle(page, 2500)
    return page.locator("section#active")


def state_audit(page, ctx: Ctx) -> None:
    page.goto(ctx.url("/audit"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("main", timeout=20_000)
    _settle(page, 2500)


def state_cluster_detail(page, ctx: Ctx) -> None:
    cl = ctx.api("/clusters?page_size=40")
    rows = [c for c in cl.get("clusters", cl.get("items", [])) if c.get("cluster_id", -1) >= 0]
    cid = max(rows, key=lambda c: min(c.get("size", 0), 60))["cluster_id"]
    page.goto(ctx.url(f"/clusters/{cid}"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='cluster-header-counts']", timeout=20_000)
    _settle(page, 2500)


def state_assist_dialog(page, ctx: Ctx) -> None:
    # The dialog only plans (a dry run); nothing is started or written.
    state_cluster_detail(page, ctx)
    page.get_by_role("button", name="Select page").click()
    page.locator("[data-testid='assist-open']").click()
    page.wait_for_selector("[data-testid='assist-plan-sampled'], [data-testid='assist-plan-error']", timeout=30_000)
    page.locator("[data-testid='assist-choice-representative'] input").check()
    page.wait_for_selector("[data-testid='assist-plan-sampled'], [data-testid='assist-plan-error']", timeout=30_000)
    _settle(page, 1500)


def state_dataset_imports(page, ctx: Ctx) -> None:
    page.goto(ctx.url("/datasets/imports"), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("main", timeout=20_000)
    _settle(page, 2000)


def _settings_list(path: str):
    def state(page, ctx: Ctx) -> None:
        page.goto(ctx.url(path), wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("main", timeout=20_000)
        _settle(page, 2000)

    return state


state_prompt_packs_list = _settings_list("/settings/prompt-packs")
state_region_profiles_list = _settings_list("/settings/region-profiles")
state_open_vocab_list = _settings_list("/settings/open-vocab")
