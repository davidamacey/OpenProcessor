"""Layout/copy fixes from the 2026-09-25 fresh-run findings and the F8
acceptance report, measured in a real browser (jsdom has no layout)."""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint
from playwright.sync_api import expect

NARROW = {"width": 800, "height": 1000}

CLASSES = [
    {
        "class_id": 1,
        "class_name": "widget_a",
        "group": "widgets",
        "hotkey_letter": None,
        "sample_count": 5,
        "validated_count": 5,
        "cluster_size": 5,
        "deprecated": False,
    },
]

EMPTY_PROPOSALS = {
    "total_pending": 0,
    "without_term": 0,
    "top_terms": [],
    "flagged_terms": [],
    "term_rules": None,
}


def _classes_page_stubs(stub) -> None:
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})


def test_breadcrumb_is_not_truncated_at_800(stub, page, app_url):
    """F8 D10 / F-51: the route crumb next to the logo truncated to
    "classe" at 800px."""
    page.set_viewport_size(NARROW)
    _classes_page_stubs(stub)
    page.goto(f"{app_url}/classes")
    crumb = page.locator('nav[aria-label="Breadcrumb"]')
    crumb.wait_for(timeout=ACTION_TIMEOUT_MS)
    # Layout dims need a real paint, not an arbitrary settle sleep.
    wait_for_paint(page)
    dims = crumb.evaluate("el => ({sw: el.scrollWidth, cw: el.clientWidth})")
    assert dims["sw"] <= dims["cw"] + 1, f"breadcrumb clipped at 800px: {dims}"
    link = crumb.locator("a").last
    box = link.evaluate("el => ({sw: el.scrollWidth, cw: el.clientWidth})")
    assert box["sw"] <= box["cw"] + 1, f"crumb link clipped: {box}"
    no_overflow = page.evaluate(
        "() => document.documentElement.scrollWidth <= window.innerWidth + 1"
    )
    assert no_overflow, "page overflows horizontally at 800px"


def test_bakeoff_ranked_table_shows_scroll_cue_at_800(stub, page, app_url):
    """F8 D5: the ranked table cut off after Precision at 800px with no
    sign more columns existed. V-5: the served protocol line renders."""
    from test_bakeoff import COMPARISON, DS_CURRENT, MATRIX, register_discovery

    page.set_viewport_size(NARROW)
    register_discovery(stub)
    stub.on("GET", r"/bakeoff/runs(\?|$)", {"runs": [{
        "job_id": "job-1", "state": "done", "profile": "generic", "datasets": [DS_CURRENT],
        "models": ["run:run-a", "run:run-b"], "started_at": "2026-09-25T00:00:00Z",
        "finished_at": "2026-09-25T00:01:00Z",
    }]})
    stub.on("GET", r"/bakeoff/matrix/job-1$", MATRIX)
    stub.on("GET", r"/bakeoff/results/job-1$", COMPARISON)

    page.goto(f"{app_url}/bakeoff")
    page.locator('[data-job-id="job-1"]').click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("comparison-rows").wait_for(timeout=10000)
    protocol = page.get_by_test_id("comparison-protocol")
    expect(protocol).to_contain_text("bake-off protocol:", timeout=ACTION_TIMEOUT_MS)
    scroller = page.get_by_test_id("comparison-scroll")
    overflows = scroller.evaluate("el => el.scrollWidth > el.clientWidth + 1")
    assert overflows, "fixture no longer overflows at 800px; widen it"
    hint = scroller.locator("xpath=..").get_by_test_id("scroll-hint")
    assert hint.count() == 1, "no scroll cue while columns are hidden"
    scroller.evaluate("el => { el.scrollLeft = el.scrollWidth; }")
    expect(hint).to_have_count(0, timeout=ACTION_TIMEOUT_MS)


def test_settings_unset_sort_is_not_blank(stub, page, app_url):
    """F-69: with no pinned sort and no served default the review-sort
    select rendered blank."""
    from test_curation_settings import METHODS_TODAY, SETTINGS_EMPTY, register

    methods = {
        "strategies": [
            {**s, "default": False} if s["axis"] == "sort" else s
            for s in METHODS_TODAY["strategies"]
        ],
        "flags": {},
    }
    register(stub, methods, 200, SETTINGS_EMPTY)
    page.goto(f"{app_url}/settings")
    sort_select = page.locator("select").nth(1)
    sort_select.wait_for(timeout=ACTION_TIMEOUT_MS)
    shown = sort_select.evaluate("el => el.options[el.selectedIndex]?.text ?? ''")
    assert shown.strip() != "", "the unset sort select renders blank"
    assert "not set" in shown, shown


def test_region_inventory_card_uses_served_display_name(stub, page, app_url):
    """F8 D7: the pinned region inventory card showed the raw class id as
    both title and subtitle instead of the served display name. F-37: the
    class cluster's chip reads cohesion, not purity."""
    from fixtures.wire import REGION_CLASS, REGION_TAB_LABEL, make_item
    from test_visual_fix_pages_narrow import CLASSES as NARROW_CLASSES
    from test_visual_fix_pages_narrow import CLUSTERS

    items = [make_item(crop_id=f"r-{i}", image_id=f"img-{i}") for i in range(4)]
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": NARROW_CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/regions(\?|$)", {"items": items, "total": 4})

    page.goto(f"{app_url}/clusters")
    title = page.get_by_test_id("slot-card-title")
    title.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert title.inner_text().strip() == REGION_TAB_LABEL
    assert REGION_TAB_LABEL != REGION_CLASS
    chip = page.get_by_test_id("cluster-cohesion").first
    text = " ".join(chip.inner_text().split())
    assert "cohesion 3% · n=616" in text, text
    assert "purity" not in text
