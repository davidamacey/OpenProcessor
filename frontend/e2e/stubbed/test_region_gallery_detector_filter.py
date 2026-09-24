"""W0 naming-sweep finding m9: the /clusters plate gallery's Detector
filter used to hardcode `lpr_nanov11_640`/`sam3`/`paddleocr_det_trt`/
`human` `<option>`s (`SlotGallery.svelte`). It now renders the served
`GET {API_PREFIX}/regions/vocabulary` response's filterable detectors instead —
this proves the real browser-rendered `<select>` reflects a vocabulary
this test controls, not a hardcoded list baked into the build.
"""

from __future__ import annotations

CLASSES = [
    {
        "id": 1,
        "name": "license_plate",
        "group": "vehicle",
        "hotkey_letter": "l",
        "count": 40,
        "validated_count": 12,
        "cluster_size": 44,
        "deprecated": False,
    },
]

VOCABULARY = {
    "detectors": [
        {"id": "acme_anpr_v3", "label": "ACME ANPR v3", "role": "detector", "filterable": True},
        {"id": "acme_segmenter", "label": "ACME Segmenter", "role": "segmenter", "filterable": True},
        # Not filterable — must NOT appear as a <select> option.
        {"id": "acme_classifier", "label": "ACME Classifier", "role": "classifier", "filterable": False},
    ],
    "region_sources": [],
    "chain_actors": [],
}


def test_region_gallery_detector_filter_lists_served_filterable_detectors(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/regions/vocabulary(\?|$)", VOCABULARY)
    stub.on("GET", r"/regions/clusters(\?|$)", {"clusters": [], "count": 0})
    stub.on("GET", r"/regions(\?|$)", {"items": [], "total": 0})
    # Not the plate gallery's own view, but /clusters/+page.svelte's
    # class-filter effect fires unconditionally on mount before
    # `isLicensePlateFilter` settles — stub it too so this test stays
    # fail-closed-clean rather than accumulating unrelated `unhandled` hits.
    stub.on("GET", r"/clusters(\?|$)", {"clusters": [], "count": 0})

    page.goto(f"{app_url}/clusters?class=license_plate")

    # The Detector select specifically — /clusters also renders an
    # unrelated cluster-sort <select>, so `.first` would be ambiguous.
    select = page.locator('label:has-text("Detector") select')
    select.wait_for(timeout=15000)
    page.wait_for_timeout(300)

    option_values = select.locator("option").evaluate_all(
        "opts => opts.map(o => o.value)"
    )
    option_labels = select.locator("option").evaluate_all(
        "opts => opts.map(o => o.textContent.trim())"
    )

    assert option_values == ["", "acme_anpr_v3", "acme_segmenter"], option_values
    assert option_labels == ["any", "ACME ANPR v3", "ACME Segmenter"], option_labels

    # Neither the non-filterable served detector nor any of the old
    # hardcoded ids/labels this deployment used to bake in ever appear.
    full_text = select.evaluate("el => el.textContent")
    for stale in ("ACME Classifier", "LPR", "SAM3", "Paddle det", "lpr_nanov11_640", "sam3"):
        assert stale not in full_text, f"stale/unfilterable option leaked: {stale!r}"
