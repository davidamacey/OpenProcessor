"""OpenProcessor main 840beb8 adoption: `GET {API_PREFIX}/review/tabs`
gains `filter_specs` — a self-describing enum filter per tab (e.g. the
Plates tab's `region_status`: all / detected only / verifier-rejected
candidates only). `/review`'s filter bar renders one generic `<select>`
per served spec (no tab/param-specific markup), and picking a value
forwards it as `?region_status=` on `GET {API_PREFIX}/review/regions` —
proves the real browser-rendered control actually drives the query, not
just that it renders (mirrors test_region_gallery_status_filter.py's pattern for
the /clusters plate gallery's equivalent control).

Also covers the companion wording rule (backend live-check, 2026-09-24):
a slot-tab item's own `region_rejection_reason` is rendered through the
served `rejection_reasons` vocabulary, never the generic per-item
`reason` string (which always reads "verifier rejected this candidate
(...)" — wrong wording for a `needs_human` item like
`verifier_no_verdict`, which never received a verdict at all).
"""

from __future__ import annotations

from fixtures.wire import make_item

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

REVIEW_TABS = {
    "tabs": [
        {
            "id": "regions",
            "label": "License plates",
            "filters": ["text", "region_status"],
            "filter_defaults": {"region_status": "all"},
            "filter_specs": [
                {
                    "param": "region_status",
                    "kind": "enum",
                    "label": "Status",
                    "options": [
                        {"value": "all", "label": "All (accepted + rejected candidates)"},
                        {"value": "detected", "label": "Detected only"},
                        {
                            "value": "verify_rejected",
                            "label": "Verifier-rejected candidates only",
                        },
                    ],
                },
            ],
        },
    ]
}

# openprocessor fix #29 / 840beb8: labeled region_rejection_reason vocabulary.
REJECTION_REASONS = [
    {
        "id": "region_visible_elsewhere",
        "label": "Verifier: the box is wrong (region is elsewhere)",
        "kind": "model_verdict",
        "match": "exact",
        "label_template": None,
    },
    {
        "id": "sanity_reject:",
        "label": "Box failed the geometry check",
        "kind": "automatic",
        "match": "prefix",
        "label_template": "Box failed the geometry check ({detail})",
    },
    {
        "id": "verifier_no_verdict",
        "label": "Verifier gave no verdict — needs human review",
        "kind": "needs_human",
        "match": "exact",
        "label_template": None,
    },
]

VOCABULARY = {
    "detectors": [
        {"id": "sam3", "label": "SAM3", "role": "segmenter", "filterable": True},
    ],
    "region_sources": [],
    "chain_actors": [],
    "text_choices": [],
    "text_rules": None,
    "rejection_reasons": REJECTION_REASONS,
}

_CANDIDATE_BBOX = [0.15, 0.25, 0.55, 0.75]


def needs_human_item() -> dict:
    # A verify_rejected candidate the verifier never actually judged —
    # the live backend fact this adoption is pinned against (18 total
    # under region_status=verify_rejected: 13 model_verdict + 5
    # verifier_no_verdict).
    item = make_item(
        crop_id="plate-needs-human-1",
        image_id="img-1",
        class_id=1,
        class_name="license_plate",
        thumbnail_url="/curation/crops/plate-needs-human-1/thumbnail",
        region_bbox_norm=None,
        region_bbox_in_parent=None,
        region_status="verify_rejected",
        region_rejection_reason="verifier_no_verdict",
        region_verified=False,
        region_validated=False,
        region_auto_confirmed=False,
        region_candidate_bbox_norm=_CANDIDATE_BBOX,
        region_candidate_bbox_in_parent=_CANDIDATE_BBOX,
        region_candidate_score=0.42,
        region_candidate_detector="sam3",
        region_candidate_detector_version="3.0.0",
        region_candidate_source="segmenter",
    )
    # Every /review/{tab} row also carries the generic `reason` key (m4
    # mismatches) — the backend fact this test pins: it always reads
    # "verifier rejected this candidate (...)" even for a needs_human
    # item, so the frontend must not render it verbatim once
    # region_rejection_reason is present.
    item["reason"] = "verifier rejected this candidate (verifier_no_verdict) — needs human review"
    return item


def test_region_status_filter_forwards_the_param_and_needs_human_reason_never_reads_rejected(
    stub, page, app_url
):
    region_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/tabs(\?|$)", REVIEW_TABS)
    stub.on("GET", r"/regions/vocabulary(\?|$)", VOCABULARY)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(request, _match):
        region_calls.append(request.url)
        return (
            200,
            {"items": [needs_human_item()], "total": 1, "page": 1, "page_size": 30},
        )

    stub.on("GET", r"/review/regions(\?|$)", review_handler)

    page.goto(f"{app_url}/review?tab=plates")
    counter = page.get_by_test_id("queue-counter")
    counter.first.wait_for(timeout=15000)
    page.wait_for_timeout(500)

    # The served label renders (not a raw param name), and the label
    # ("Needs review") is never worded as a rejection for this
    # needs_human item — the raw `reason` string ("verifier rejected
    # this candidate (...)") must not appear anywhere on the page.
    page.get_by_text("Needs review", exact=False).first.wait_for(timeout=10000)
    assert "candidate · needs review" in page.content()
    assert "verifier rejected this candidate" not in page.content()
    page.get_by_text("Verifier gave no verdict", exact=False).first.wait_for(timeout=10000)

    select = page.locator('label:has-text("Status") select')
    select.wait_for(timeout=10000)
    option_values = select.locator("option").evaluate_all("opts => opts.map(o => o.value)")
    assert option_values == ["all", "detected", "verify_rejected"], option_values

    region_calls.clear()
    select.select_option("verify_rejected")
    page.wait_for_timeout(500)

    assert region_calls, "picking a status must trigger a fresh GET {API_PREFIX}/review/regions call"
    assert any("region_status=verify_rejected" in url for url in region_calls), region_calls

    errors = [c for c in stub.console_errors if c.startswith("pageerror")]
    assert not errors, f"no pageerror expected in the region_status filter flow: {errors[:3]}"


def test_region_status_from_the_url_reaches_the_queue_request(stub, page, app_url):
    # Live regression (2026-09-24): loading /review?tab=plates&region_status=
    # verify_rejected directly showed the unfiltered queue. _filter() only
    # forwards params the tab's served filter_specs declare, and the refetch
    # effect keyed on the raw URL-seeded values, so when /review/tabs landed
    # after the first queue fetch nothing refetched with the param.
    region_calls: list[str] = []

    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/tabs(\?|$)", REVIEW_TABS)
    stub.on("GET", r"/regions/vocabulary(\?|$)", VOCABULARY)
    stub.on("GET", r"/crops/[^/]+/image$", (200, b"", "image/jpeg"))

    def review_handler(request, _match):
        region_calls.append(request.url)
        return (
            200,
            {"items": [needs_human_item()], "total": 1, "page": 1, "page_size": 30},
        )

    stub.on("GET", r"/review/regions(\?|$)", review_handler)

    page.goto(f"{app_url}/review?tab=plates&region_status=verify_rejected")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=15000)
    page.wait_for_timeout(1500)

    assert region_calls, "the plates queue must be fetched"
    assert "region_status=verify_rejected" in region_calls[-1], region_calls
    select = page.locator('label:has-text("Status") select')
    assert select.input_value() == "verify_rejected"
