"""Scripted capture states for the region, embedding and import-lock features."""

from __future__ import annotations

import re

from docs_screenshots.ctx import (
    IMPORT_PROJECT,
    MAIN_PROJECT,
    SHARE_CONSUMER,
    SHARE_OWNER,
    Ctx,
    _life,
    _settle,
    api_get,
)

# --- v0.4.0 feature slots (issue #36) -------------------------------------------------
# MAIN_PROJECT holds the full public COCO sample (VLM labels, embeddings); LIFE_PROJECT is
# a throwaway `cwlife-<epoch>-x` project holding public COCO vehicle photos with a wheel
# region profile, an ingest policy that leaves some detections un-embedded and an
# open-vocabulary set (created by hand: see developer-guide/screenshots.md).


def state_resources_menu(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/p/{MAIN_PROJECT}/dashboard", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='resources-trigger']", timeout=20_000)
    _settle(page, 1500)
    page.locator("[data-testid='resources-trigger']").click()
    page.wait_for_selector("[data-testid='resources-list']", timeout=10_000)
    _settle(page, 600)


def state_wheels_inventory_card(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/p/{_life()}/clusters", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("text=inventory", timeout=30_000)
    _settle(page, 3000)


def state_region_gallery_boxes(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{_life()}/clusters?class=wheel", wait_until="domcontentloaded", timeout=30_000
    )
    page.wait_for_selector("text=/boxes listed/", timeout=30_000)
    _settle(page, 3000)


def state_import_job_actions(page, ctx: Ctx) -> None:
    jobs = api_get(ctx.base, f"/curation/projects/{IMPORT_PROJECT}/datasets/imports")["items"]
    jid = next(j["import_id"] for j in jobs if j["status"].startswith("completed"))
    page.goto(
        f"{ctx.base}/p/{IMPORT_PROJECT}/datasets/imports/{jid}",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='import-job']", timeout=15_000)
    page.wait_for_selector("text=/Cancel import:/", timeout=15_000)
    _settle(page, 1200)


def state_region_profile_testable(page, ctx: Ctx) -> None:
    life = _life()
    first = api_get(ctx.base, f"/curation/projects/{life}/review/regions?page_size=1")["items"][0]["id"]
    page.goto(
        f"{ctx.base}/p/{life}/settings/region-profiles/wheels",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='profile-test-panel']", timeout=20_000)
    page.locator("[data-testid='profile-test-crop-id']").fill(first)
    page.locator("[data-testid='test-run']").click()
    page.wait_for_selector("[data-testid='test-result'], [data-testid='test-error']", timeout=180_000)
    if page.locator("[data-testid='test-error']").count():
        raise RuntimeError(page.locator("[data-testid='test-error']").inner_text())
    page.locator("[data-testid='profile-test-panel']").scroll_into_view_if_needed()
    _settle(page, 800)
    return page.locator("[data-testid='profile-test-panel']")


def state_segmenter_prompt_check(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{_life()}/settings/region-profiles/wheels",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='check-segmenter-prompt']", timeout=20_000)
    page.locator("[data-testid='check-segmenter-prompt']").click()
    page.wait_for_selector("[data-testid='segmenter-prompt-check']", timeout=60_000)
    page.locator("[data-testid='segmenter-prompt-check']").scroll_into_view_if_needed()
    _settle(page, 800)
    return page.locator("[data-testid='profile-group']", has=page.locator("[data-testid='check-segmenter-prompt']"))


def _first_locked_cluster(base: str) -> int:
    d = api_get(base, f"/curation/projects/{IMPORT_PROJECT}/crops?page_size=100")
    counts: dict[int, int] = {}
    for c in d["crops"]:
        if c.get("label_locked") and c.get("cluster_id", -1) >= 0:
            counts[c["cluster_id"]] = counts.get(c["cluster_id"], 0) + 1
    return max(counts, key=lambda k: counts[k])


def state_locked_item_badge(page, ctx: Ctx) -> None:
    cid = _first_locked_cluster(ctx.base)
    page.goto(f"{ctx.base}/p/{IMPORT_PROJECT}/clusters/{cid}", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='label-locked-badge']", timeout=30_000)
    _settle(page, 2000)


def state_reprocess_served_scopes(page, ctx: Ctx) -> None:
    cid = _first_locked_cluster(ctx.base)
    page.goto(f"{ctx.base}/p/{IMPORT_PROJECT}/clusters/{cid}", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='cluster-header-counts']", timeout=15_000)
    _settle(page)
    # The first twelve crops: over 20 the server runs detection as a background job and
    # asks for one scope at a time, which is a different dialog.
    cards = page.locator("[role='button'][aria-pressed]")
    cards.nth(0).click()
    cards.nth(11).click(modifiers=["Shift"])  # a range, anchor to here
    page.wait_for_timeout(500)
    page.locator("[data-testid='reprocess-open']").first.click()
    page.wait_for_selector("[role='dialog'][aria-label='Reprocess']", timeout=10_000)
    scopes = page.locator("[role='dialog'][aria-label='Reprocess'] input[type=checkbox]")
    scopes.nth(0).check()
    scopes.nth(3).check()
    page.get_by_role("button", name="Check what would run").click()
    page.wait_for_selector("[data-testid='reprocess-dry-run']", timeout=30_000)
    _settle(page, 800)


def state_clone_from_project(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{MAIN_PROJECT}/settings/prompt-packs", wait_until="domcontentloaded", timeout=30_000
    )
    page.get_by_role("button", name=re.compile("^Clone")).first.click()
    page.wait_for_selector("[data-testid='clone-from-project']", timeout=15_000)
    # A native select popup is not part of a screenshot: show its options inline
    # (display only; nothing is picked, nothing is submitted).
    # Only the public sample projects are listed in the image: other options are
    # removed from this page's DOM before the shot (the served list is unchanged).
    page.locator("[data-testid='clone-from-project']").evaluate(
        """(el, keep) => {
          for (const o of [...el.options]) {
            if (o.value !== '' && !keep.includes(o.value)) o.remove();
          }
          el.size = el.options.length;
        }""",
        [MAIN_PROJECT, IMPORT_PROJECT, "sample-coco-vehicles-v2", SHARE_OWNER, SHARE_CONSUMER],
    )
    _settle(page, 800)


def state_ingest_policy_preview(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{_life()}/settings/ingest-policy", wait_until="domcontentloaded", timeout=30_000
    )
    page.wait_for_selector("[data-testid='policy-preview-summary']", timeout=30_000)
    _settle(page, 1000)


def state_open_vocab_editor(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{_life()}/settings/open-vocab/vehicle_parts",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='open-vocab-targets']", timeout=20_000)
    _settle(page, 1500)


def state_open_vocab_test(page, ctx: Ctx) -> None:
    life = _life()
    items = api_get(ctx.base, f"/curation/projects/{life}/review/regions?page_size=100")["items"]
    car = max(  # the car crop with the most wheel boxes: a part is surely visible in it
        items, key=lambda it: len(it["region_boxes"])
    )["id"]
    page.goto(
        f"{ctx.base}/p/{life}/settings/open-vocab/vehicle_parts",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='open-vocab-test-panel']", timeout=20_000)
    page.locator("[data-testid='ov-test-crop-id']").fill(car)
    page.locator("[data-testid='ov-test-run']").click()
    page.wait_for_selector("[data-testid='ov-test-result'], [data-testid='ov-test-error']", timeout=180_000)
    if page.locator("[data-testid='ov-test-error']").count():
        raise RuntimeError(page.locator("[data-testid='ov-test-error']").inner_text())
    panel = page.locator("[data-testid='open-vocab-test-panel']")
    panel.scroll_into_view_if_needed()
    _settle(page, 800)
    return panel


def state_region_stage_panel(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/p/{_life()}/ingest", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='region-stage-panel']", timeout=30_000)
    panel = page.locator("[data-testid='region-stage-panel']")
    panel.scroll_into_view_if_needed()
    _settle(page, 1500)
    return panel.locator("xpath=..")  # the worklog card: no server paths in the crop


def state_ingest_sources_panel(page, ctx: Ctx) -> None:
    """The Sources panel with the create form open. Opening the form saves nothing."""
    page.goto(f"{ctx.base}/p/{_life()}/ingest", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("[data-testid='ingest-sources']", timeout=30_000)
    page.get_by_test_id("source-add").click()
    page.get_by_test_id("source-form").wait_for(timeout=10_000)
    panel = page.locator("[data-testid='ingest-sources']")
    panel.scroll_into_view_if_needed()
    _settle(page, 1000)
    return panel.locator("xpath=..")


def state_ingest_source_card(page, ctx: Ctx) -> None:
    """One source card with its served status (counts, triggers, webhook path, last run)."""
    page.goto(f"{ctx.base}/p/{_life()}/ingest", wait_until="domcontentloaded", timeout=30_000)
    card = page.locator("[data-testid='source-card']").first
    card.wait_for(timeout=30_000)
    _settle(page, 2500)
    return card


def state_dashboard_embedding(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/p/{_life()}/dashboard", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("text=Embedding", timeout=30_000)
    _settle(page, 2500)


def state_crop_embedding_row(page, ctx: Ctx) -> None:
    page.goto(
        f"{ctx.base}/p/{_life()}/clusters?mode=matching&embedding_state=not_selected",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_selector("[data-testid='embedding-state-badge']", timeout=30_000)
    _settle(page, 1500)
    card = page.locator("[data-testid='embedding-state-badge']").first.locator("xpath=ancestor::*[contains(@class,'group')][1]")
    card.hover()
    card.locator("button[title^='Details']").first.click()
    page.wait_for_selector("[aria-label='Crop details']", timeout=15_000)
    page.get_by_text("Embedding", exact=True).first.scroll_into_view_if_needed()
    _settle(page, 1200)


def state_run_results_confusion(page, ctx: Ctx) -> None:
    page.goto(f"{ctx.base}/p/{IMPORT_PROJECT}/train", wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("text=Past runs", timeout=30_000)
    _settle(page, 1500)
    page.locator("button[aria-expanded]", has_text="Results").first.click()
    page.wait_for_selector("[data-testid='run-results-title']", timeout=20_000)
    page.get_by_label("Enlarge confusion matrix").first.evaluate(
        "e => e.scrollIntoView({block: 'center'})"
    )
    _settle(page, 2500)
