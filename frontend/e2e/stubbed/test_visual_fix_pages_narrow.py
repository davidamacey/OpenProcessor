"""Narrow-viewport (800px) layout regressions from the 2026-09-24 visual
audit (docs/design/visual-audit-2026-09-24.md) on /clusters,
/clusters/[id] and /dashboard. jsdom has no layout, so these run in a real
browser: each asserts on measured geometry, not on class names.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS, wait_for_paint

from fixtures.wire import REGION_CLASS, make_box, make_item

NARROW = {"width": 800, "height": 1000}

CLUSTER_ID = 64

CLASSES = [
    {"class_id": CLUSTER_ID, "class_name": "class_b", "kind": "item", "group": "moto", "hotkey_letter": None,
     "sample_count": 641, "validated_count": 0, "cluster_size": 616, "deprecated": False},
    {"class_id": 80, "class_name": REGION_CLASS, "kind": "region", "group": "widgets", "hotkey_letter": None,
     "sample_count": 40, "validated_count": 0, "cluster_size": 44, "deprecated": False},
]

CLUSTERS = {
    "items": [
        {
            "cluster_id": CLUSTER_ID,
            "cluster_kind": "class",
            "size": 616,
            "validated_count": 0,
            "labelled_count": 616,
            "dominant_class_id": CLUSTER_ID,
            "dominant_class_name": "class_b",
            "dominant_count": 616,
            "purity": 0.03,
            "purity_n": 616,
            "purity_basis": "nearest_centroid",
            "purity_tier": "noisy",
            "label_purity": 1.0,
            "labelled_share": 1.0,
            "promotable": False,
            "is_unlabeled": False,
            "representatives": [],
            "n_subclusters": 0,
            "updated_at": None,
        }
    ],
    "total": 1,
    "total_class_clusters": 1,
    "total_candidate_clusters": 0,
    "cluster_id_offset": 10000,
}

CLASS_SOURCES = {
    "class_sources": [
        {"id": "vlm", "label": "Labeled by the VLM", "role": "vlm"},
    ]
}


def _crop(i: int) -> dict:
    return make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=CLUSTER_ID,
        class_name="class_b",
        class_source="vlm",
        label_source="vlm",
        label_validated=False,
        class_validated=False,
        cluster_id=CLUSTER_ID,
        thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
    )


def _no_page_errors(stub) -> None:
    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, errors[:3]


def test_crop_card_class_name_is_readable_at_800(stub, page, app_url):
    """K2: at 800px the card's class name was squeezed to "sp" by the
    served source label ("Labeled by the VLM"); the grid also kept 4+
    columns in ~500px of content."""
    page.set_viewport_size(NARROW)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/class_sources(\?|$)", CLASS_SOURCES)
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)",
            {"total": 6, "page": 1, "page_size": 60, "crops": [_crop(i) for i in range(6)]})

    page.goto(f"{app_url}/p/default/clusters/{CLUSTER_ID}")
    name = page.locator('[data-testid="class-name"]').first
    name.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)  # layout settle for the following overflow/bbox check

    box = name.evaluate(
        "el => ({w: el.clientWidth, sw: el.scrollWidth, text: el.textContent.trim()})"
    )
    assert box["text"] == "class_b", box
    assert box["sw"] <= box["w"], f"class name is truncated at 800px: {box}"

    chip = page.locator('[data-testid="source-chip"]').first
    assert "Labeled by" not in chip.inner_text()
    assert chip.get_attribute("title") == "Label source: Labeled by the VLM — not yet validated"

    card_width = page.locator('[data-testid="class-name"]').first.evaluate(
        "el => el.closest('[aria-pressed]').getBoundingClientRect().width"
    )
    assert card_width >= 140, f"crop cards too narrow at 800px: {card_width}px"
    _no_page_errors(stub)


def test_region_gallery_chips_stay_inside_their_cards_at_800(stub, page, app_url):
    """C3: provenance chips ran past each region card's right edge
    (overflow probe: right=1644 in a 1600 viewport)."""
    page.set_viewport_size(NARROW)
    long_chain = [
        "tag_detector_with_a_very_long_identifier_v11:miss",
        "combined_verify_reject:region_visible_elsewhere_extra_long_reason",
        "tag_segmenter:hit",
    ]
    items = [
        make_item(
            crop_id=f"r-{i}",
            image_id=f"img-{i}",
            class_name="class_b_with_a_long_class_name",
            region_detector_chain=long_chain,
            region_boxes=[
                make_box(
                    "b1",
                    text="65875-LONG-TEXT",
                    text_vlm="65875-LONG-TEXT",
                    text_ocr="65875-L0NG-TEXT",
                    text_disagreement=True,
                )
            ],
            region_count=1,
        )
        for i in range(4)
    ]
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/regions/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/clusters(\?|$)", {"clusters": [], "count": 0})
    rows = [{**i, "row_key": f"{i['crop_id']}#item"} for i in items]
    stub.on("GET", r"/regions(\?|$)", {"items": rows, "total": 4})

    page.goto(f"{app_url}/p/default/clusters?class={REGION_CLASS}")
    page.locator('[data-testid="slot-text-value"]').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)  # layout settle for the following overflow/bbox check

    # C2: the filter chip names the class, not "#80".
    chip = page.locator('[data-testid="class-filter-chip"]')
    assert chip.inner_text().strip() == f"class: {REGION_CLASS}", chip.inner_text()

    overflow = page.evaluate(
        """() => {
          const out = [];
          for (const v of document.querySelectorAll('[data-testid="slot-text-value"]')) {
            const card = v.closest('button');
            const cr = card.getBoundingClientRect();
            for (const el of card.querySelectorAll('*')) {
              const r = el.getBoundingClientRect();
              if (r.width > 0 && r.right > cr.right + 1) {
                out.push({tag: el.tagName, text: (el.textContent || '').slice(0, 40),
                          right: r.right, cardRight: cr.right});
              }
            }
          }
          return out;
        }"""
    )
    assert overflow == [], f"elements overflow their region card: {overflow[:4]}"
    _no_page_errors(stub)


def test_ignored_mode_hides_cluster_grid_controls(stub, page, app_url):
    """C5: Sort/subject/clarity stayed live in the Ignored view and the
    footer still read the cluster grid's "107 / 107 all loaded"."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/clusters(\?|$)", CLUSTERS)
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 60, "crops": []})

    stub.on("GET", r"/regions(\?|$)", {"items": [], "total": 0})

    page.goto(f"{app_url}/p/default/clusters")
    sort = page.locator('label:has-text("Sort") select')
    sort.wait_for(timeout=ACTION_TIMEOUT_MS)

    page.locator("button:has-text('Ignored')").first.click()
    page.get_by_text("Nothing ignored.").wait_for(timeout=10000)

    assert page.locator('label:has-text("Sort") select').count() == 0
    footer = page.locator('[data-testid="clusters-footer-count"]').inner_text()
    assert "ignored crops" in footer, footer
    _no_page_errors(stub)


def test_models_updated_label_does_not_overlap_description(stub, page, app_url):
    """M1: at 1600px "Updated just now" ran into the description line."""
    page.set_viewport_size({"width": 1600, "height": 1000})
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/models/status", {"models": []})

    page.goto(f"{app_url}/p/default/models")
    status = page.locator('[data-testid="models-status"]')
    status.get_by_text("Updated", exact=False).wait_for(timeout=ACTION_TIMEOUT_MS)

    boxes = page.evaluate(
        """() => {
          const p = document.querySelector('header p');
          const s = document.querySelector('[data-testid="models-status"] span');
          const a = p.getBoundingClientRect(), b = s.getBoundingClientRect();
          return {pRight: a.right, sLeft: b.left, sLines: Math.round(b.height / 16)};
        }"""
    )
    assert boxes["sLeft"] >= boxes["pRight"], boxes
    assert boxes["sLines"] <= 1, f"'Updated ...' wraps: {boxes}"
    _no_page_errors(stub)


DASHBOARD_STATS = {
    "as_of": "2026-09-24T00:00:00Z",
    "total_crops": 3,
    "validated": 0,
    "test_holdout": 0,
    "by_source": [],
    "labeled": {"by_human": 0, "by_vlm": 0, "by_classifier": 0, "other": 0},
    "regions": {"total_detected": 0, "by_detector": 0, "by_segmenter": 0, "by_human": 0},
    "unlabeled": {"pending_detection": 0, "pending_verification": 0, "no_label_source": 0},
    "in_progress": {"region_drain_total_unfinished": 0},
    "clusters": {"last_run_at": None, "cluster_count": 0, "residual_count": 0,
                 "noise_count": 0, "method": None},
}

COMPLETED_JOB = {
    "job_id": "job-1",
    "status": "completed",
    "stage": "finalize",
    "processed": 0,
    "total": 0,
    "started_at": 1,
    "finished_at": 2,
    "error": None,
    "result": {
        "stages": {
            "cluster_residuals": {"status": "success", "method": "ivf", "n_clusters": 1},
            "auto_promote": {"skipped": True, "reason": "disabled by default"},
        }
    },
    "args": {},
    "eta_seconds": None,
    "elapsed_seconds": 1,
}


def test_dashboard_recluster_card_stacks_at_800(stub, page, app_url):
    """D3: at 800px the recluster description was squeezed into a ~60px
    column (one word per line). D5: the last-run summary is a stage table,
    not a JSON dump."""
    page.set_viewport_size(NARROW)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/stats/classes(\?|$)", {"classes": []})
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on("GET", r"/stats/dataset(\?|$)", DASHBOARD_STATS)
    stub.on("GET", r"/crops(\?|$)", {"total": 0, "page": 1, "page_size": 20, "crops": []})
    stub.on("GET", r"/pipeline/auto_label/status", COMPLETED_JOB)

    page.goto(f"{app_url}/p/default/dashboard")
    desc = page.locator('[data-testid="recluster-description"]')
    desc.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)  # layout settle for the following overflow/bbox check

    # Stacked: the description spans its card's full content width.
    ratio = desc.evaluate(
        "el => el.getBoundingClientRect().width"
        " / el.closest('section').getBoundingClientRect().width"
    )
    assert ratio >= 0.9, f"recluster description squeezed to {ratio:.0%} of its card at 800px"

    page.get_by_text("Last run summary").click()
    table = page.locator('[data-testid="last-run-stages"]')
    table.wait_for(timeout=5000)
    text = " ".join(table.inner_text().split())
    assert "disabled by default" in text, text
    assert '"stages"' not in text
    _no_page_errors(stub)
