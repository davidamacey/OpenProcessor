"""/review and top-nav fixes from docs/design/visual-audit-2026-09-24.md.

- R1: the slot-bound region class (most validated) topped the `/` picker
  and was the first quick-assign chip, so `/` + Enter labeled an item as
  a region.
- R2: at 800px the tab strip clipped "New class proposals"/the region tab
  with no hint, the active tab could sit invisible past the edge, and the
  Confirm/Skip/Discard row sat below the fold.
- Narrow nav: at 800px the primary nav scrolled with no hint that links
  sat off-screen.
- R3: an empty queue said only "Queue empty.".
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint

import struct
import zlib

from fixtures.wire import REGION_CLASS, REGION_TAB_LABEL, REGION_TAB_URL_ID, make_item

CLASSES = [
    {
        "id": 80,
        "name": REGION_CLASS,
        "group": "special",
        "hotkey_letter": None,
        "count": 5614,
        "validated_count": 163,
        "cluster_size": 5614,
        "deprecated": False,
    },
    {
        "id": 1,
        "name": "miata",
        "group": "cars",
        "hotkey_letter": None,
        "count": 35,
        "validated_count": 35,
        "cluster_size": 35,
        "deprecated": False,
    },
    {
        "id": 2,
        "name": "touringbike",
        "group": "bikes",
        "hotkey_letter": None,
        "count": 3,
        "validated_count": 0,
        "cluster_size": 3,
        "deprecated": False,
    },
    {
        "id": 3,
        "name": "zz_old",
        "group": "cars",
        "hotkey_letter": None,
        "count": 0,
        "validated_count": 0,
        "cluster_size": 0,
        "deprecated": True,
    },
]

TABS = {
    "tabs": [
        {"id": "regions", "label": REGION_TAB_LABEL},
        {
            "id": "uncertainty",
            "label": "Uncertainty",
            "description": "High active-learning probe entropy",
        },
        {"id": "new_class_proposals", "label": "New class proposals"},
    ]
}


def review_item(i: int, **over) -> dict:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=None,
        class_name=None,
        proposed_class_id=2,
        proposed_class_name="touringbike",
        label_validated=False,
        label_source="vlm",
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        **over,
    )
    item["reason"] = "needs human review"
    return item


def _png(width: int, height: int) -> bytes:
    """A solid grey PNG — real pixel dimensions, so the source/crop panels
    lay out the way they do with a live image (the default stub image is
    1x1, which hides every "pushed below the fold" layout bug)."""
    row = b"\x00" + b"\x80\x80\x80" * width
    raw = zlib.compress(row * height)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def _stub_review(
    stub,
    *,
    empty_uncertainty: bool = False,
    empty_reason: str | None = None,
    empty_state: dict | None = None,
) -> None:
    source_png = _png(640, 427)
    crop_png = _png(300, 140)
    stub.on("GET", r"thumbnail(/|$|\?)", (200, crop_png, "image/png"))
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    # The review source panel's image (server-burned bbox overlay).
    stub.on("GET", r"/crops/[^/]+/image$", (200, source_png, "image/png"))

    def review_handler(request, _match):
        if empty_uncertainty and "/review/uncertainty" in request.url:
            body = {
                "items": [],
                "total": 0,
                "page": 1,
                "page_size": 30,
                "sort_applied": "atypicality",
                "sort_fallback_reason": "no item has 'probe_pred_entropy' yet",
            }
            if empty_reason is not None:
                body["empty_reason"] = empty_reason
            return (200, body)
        items = [review_item(i) for i in range(3)]
        return (200, {"items": items, "total": 3, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)
    tabs = dict(TABS)
    if empty_state is not None:
        tabs = {**TABS, "empty_state": empty_state}
    stub.on("GET", r"/review/tabs(\?|$)", tabs)


def test_picker_and_quick_assign_never_offer_the_region_class(stub, page, app_url):
    """R1: item-class targets only, the item's own proposal ranked first."""
    _stub_review(stub)
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(f"{app_url}/review")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_role("button", name="Confirm", exact=True).wait_for(timeout=ACTION_TIMEOUT_MS)

    quick = page.locator('button[title^="Assign "]')
    quick.first.wait_for(timeout=10000)
    names = [t.strip() for t in quick.all_inner_texts()]
    assert REGION_CLASS not in names, names
    assert names[0] == "touringbike", names  # the item's own proposal, first

    page.keyboard.press("/")
    search = page.get_by_placeholder("Search all 2 classes…")
    search.wait_for(timeout=5000)
    rows = page.get_by_role("dialog", name="Search classes").locator("li button")
    row_names = [r.split("\n")[0].strip() for r in rows.all_inner_texts()]
    assert all(REGION_CLASS not in r for r in row_names), row_names
    assert "touringbike" in row_names[0], row_names


def test_narrow_review_tabs_and_nav_show_hidden_items(stub, page, app_url):
    """R2 + narrow nav: overflow is signalled, the active tab is scrolled
    into view, the page never overflows, and the actions stay on screen."""
    _stub_review(stub)
    page.set_viewport_size({"width": 800, "height": 1000})
    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)

    # The active (last) tab was invisible past the edge — now scrolled in.
    active = page.get_by_role("button", name=REGION_TAB_LABEL, exact=True)
    active.wait_for(timeout=10000)
    # Scroll-into-view layout needs a real paint, not an arbitrary sleep.
    wait_for_paint(page)
    strip_box = page.get_by_test_id("review-tabs").bounding_box()
    box = active.bounding_box()
    assert strip_box and box
    assert box["x"] >= strip_box["x"] - 1, (box, strip_box)
    assert box["x"] + box["width"] <= strip_box["x"] + strip_box["width"] + 1, (box, strip_box)
    # ... and the tabs scrolled off to the left are signalled.
    assert page.get_by_test_id("review-tabs-more-left").is_visible()

    # The primary nav signals its off-screen links too.
    assert page.get_by_test_id("primary-nav-more-right").is_visible() or page.get_by_test_id(
        "primary-nav-more-left"
    ).is_visible()

    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1, overflow

    # Action row within the viewport without scrolling.
    page.goto(f"{app_url}/review")
    confirm = page.get_by_role("button", name="Confirm", exact=True)
    confirm.wait_for(timeout=ACTION_TIMEOUT_MS)
    cbox = confirm.bounding_box()
    assert cbox and cbox["y"] + cbox["height"] <= 1000, cbox
    assert page.get_by_test_id("review-tabs-more-right").is_visible()


def test_empty_queue_explains_itself_and_dims_the_tab(stub, page, app_url):
    """R3: an empty queue gives the served description + fallback reason."""
    _stub_review(stub, empty_uncertainty=True)
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(f"{app_url}/review?tab=uncertainty")
    empty = page.get_by_test_id("queue-empty")
    empty.wait_for(timeout=ACTION_TIMEOUT_MS)
    text = empty.inner_text()
    assert "The Uncertainty queue is empty." in text, text
    assert "High active-learning probe entropy" in text, text
    assert "no item has 'probe_pred_entropy' yet" in text, text
    assert "Queue empty." not in text
    page.get_by_test_id("tab-empty-count").first.wait_for(timeout=5000)


def test_empty_queue_shows_served_empty_reason_and_links_to_the_probe_control(
    stub, page, app_url
):
    """#36 item 9: GET {API_PREFIX}/review/{tab} serves empty_reason directly
    when total == 0, and /review/tabs serves a top-level empty_state saying
    whether a probe has ever run. The empty message must show the served
    reason (over the raw sort-fallback string) and link to /train when
    empty_state says no probe predictions exist at all."""
    _stub_review(
        stub,
        empty_uncertainty=True,
        empty_reason="no probe predictions — run a probe",
        empty_state={"has_probe_predictions": False, "has_item_scores": True},
    )
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(f"{app_url}/review?tab=uncertainty")
    empty = page.get_by_test_id("queue-empty")
    empty.wait_for(timeout=ACTION_TIMEOUT_MS)
    text = empty.inner_text()
    assert "no probe predictions — run a probe" in text, text
    assert "no item has 'probe_pred_entropy' yet" not in text, text

    link = empty.get_by_role("link", name="Run a probe on /train")
    link.wait_for(timeout=5000)
    assert link.get_attribute("href") == "/train"


_CANDIDATE_BBOX = [0.15, 0.25, 0.55, 0.75]

STATUSES = {
    "statuses": [
        {
            "value": "detected",
            "label": "detected (region visible)",
            "role": "positive",
            "terminal": True,
            "human_writable": True,
            "clears_box": False,
            "wants_reason": False,
        },
        {
            "value": "verify_rejected",
            "label": "rejected (bad detection)",
            "role": "rejected",
            "terminal": True,
            "human_writable": True,
            "clears_box": False,
            "wants_reason": True,
        },
    ],
    "confirm_status": "detected",
    "reject_status": "verify_rejected",
    "false_positive_status": None,
}

VOCABULARY = {
    "detectors": [],
    "region_sources": [],
    "chain_actors": [],
    "text_choices": [],
    "text_rules": None,
    "rejection_reasons": [
        {
            "id": "verifier_no_verdict",
            "label": "Verifier gave no verdict — needs human review",
            "kind": "needs_human",
            "match": "exact",
            "label_template": None,
        }
    ],
}


def test_region_panel_shows_served_rejection_label_and_neutral_text_placeholder(
    stub, page, app_url
):
    """R6: a served machine rejection reason used to sit raw in an editable
    box ("verifier_no_verdict"). R9: the empty region-text field's
    placeholder looked exactly like a reading ("ABC123")."""
    _stub_review(stub)
    stub.on("GET", r"/regions/statuses(\?|$)", STATUSES)
    stub.on("GET", r"/regions/vocabulary(\?|$)", VOCABULARY)
    item = make_item(
        crop_id="crop-r1",
        image_id="img-r1",
        class_id=1,
        class_name="miata",
        thumbnail_url="/curation/crops/crop-r1/thumbnail",
        region_bbox_norm=None,
        region_bbox_in_parent=None,
        region_status="verify_rejected",
        region_rejection_reason="verifier_no_verdict",
        region_text=None,
        region_candidate_bbox_norm=_CANDIDATE_BBOX,
        region_candidate_bbox_in_parent=_CANDIDATE_BBOX,
        region_candidate_score=0.42,
    )
    item["reason"] = "needs human review"
    stub.on(
        "GET",
        r"/review/regions(\?|$)",
        {"items": [item], "total": 1, "page": 1, "page_size": 30},
    )
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(f"{app_url}/review?tab={REGION_TAB_URL_ID}")
    served = page.get_by_test_id("served-rejection-reason")
    served.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert served.inner_text().startswith("Verifier gave no verdict"), served.inner_text()
    values = page.locator("input").evaluate_all("els => els.map(e => e.value)")
    assert "verifier_no_verdict" not in values, values

    placeholders = page.locator("input[placeholder]").evaluate_all(
        "els => els.map(e => e.placeholder)"
    )
    assert "type the text…" in placeholders, placeholders
    assert "ABC123" not in placeholders, placeholders


def test_subject_toggle_default_explains_itself(stub, page, app_url):
    """R11: the subject toggle's first option silently meant "Top 2" on one
    tab and "All ranks" elsewhere — it now carries a tooltip saying which."""
    _stub_review(stub)
    stub.on(
        "GET",
        r"/review/tabs(\?|$)",
        {"tabs": [{"id": "all", "label": "All", "filter_defaults": {"max_rank": 2}}]},
    )
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(f"{app_url}/review")
    top2 = page.get_by_role("button", name="Top 2", exact=True)
    top2.wait_for(timeout=ACTION_TIMEOUT_MS)
    title = top2.get_attribute("title") or ""
    assert "default" in title, title
