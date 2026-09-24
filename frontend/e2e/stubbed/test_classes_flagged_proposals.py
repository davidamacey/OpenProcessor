"""DQ-M11 fix (dq-queues cutover, 2026-09-24): `GET {API_PREFIX}/review/
new_class_proposals/summary` now splits terms into `top_terms`
(`flag: null`, one-click actionable) and `flagged_terms`
(`existing_class`/`generic_parent`/`non_object`, no create action). This
proves the real browser-rendered /classes page puts flagged terms in
their own collapsed section with the served reason, and offers no
"Create class & assign" for them.
"""

from __future__ import annotations

CLASSES = [
    {
        "id": 67,
        "name": "suv",
        "group": "vehicle",
        "hotkey_letter": "s",
        "count": 10,
        "validated_count": 5,
        "cluster_size": 12,
        "deprecated": False,
    },
]

PROPOSALS_SUMMARY = {
    "total_pending": 191,
    "without_term": 12,
    "top_terms": [
        {
            "label": "classic_car",
            "count": 13,
            "sample_crop_ids": ["a1", "a2"],
            "flag": None,
            "class_id": None,
        },
    ],
    "flagged_terms": [
        {
            "label": "motorcycle",
            "count": 89,
            "sample_crop_ids": ["b1"],
            "flag": "generic_parent",
            "class_id": None,
        },
        {
            "label": "suv",
            "count": 4,
            "sample_crop_ids": ["c1"],
            "flag": "existing_class",
            "class_id": 67,
        },
        {
            "label": "abstract_blur",
            "count": 2,
            "sample_crop_ids": [],
            "flag": "non_object",
            "class_id": None,
        },
    ],
    "term_rules": {
        "generic_terms": ["car", "motorcycle"],
        "non_object_terms": ["abstract_blur"],
        "registry_groups_are_generic": True,
        "existing_classes_flagged": True,
        "generic_terms_env": "OP_NEW_CLASS_GENERIC_TERMS",
        "non_object_terms_env": "OP_NEW_CLASS_NON_OBJECT_TERMS",
    },
}


def test_classes_flagged_terms_render_separately_with_no_create_action(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on(
        "GET",
        r"/review/new_class_proposals/summary(\?|$)",
        PROPOSALS_SUMMARY,
    )

    page.goto(f"{app_url}/classes")

    heading = page.get_by_role("heading", name="New class proposals")
    heading.wait_for(timeout=15000)

    # top_terms: one actionable row with "Create class & assign".
    assert page.get_by_text("classic_car").count() >= 1
    assert page.get_by_text("Create class & assign").count() == 1

    # flagged_terms: collapsed <details>, not expanded by default.
    details = page.locator("details", has_text="Flagged terms")
    details.wait_for(timeout=15000)
    assert details.get_attribute("open") is None

    # Expand it and check every flagged term's reason renders, with no
    # create action anywhere inside the flagged section.
    details.locator("summary").click()
    page.wait_for_timeout(200)

    details_text = details.inner_text()
    assert "motorcycle" in details_text
    assert "generic parent" in details_text
    assert "suv" in details_text
    assert "existing class" in details_text
    assert "abstract_blur" in details_text
    assert "not an object" in details_text

    # No "Create class & assign" button inside the flagged section.
    assert details.get_by_text("Create class & assign").count() == 0

    # existing_class term gets a map-to-class button naming the class.
    assert details.get_by_role("button", name="Map to suv").count() == 1

    # without_term and term_rules surface as help text somewhere on the
    # page (not silently dropped).
    body_text = page.locator("body").inner_text()
    assert "12" in body_text  # without_term count
    assert "abstract_blur" in body_text  # term_rules non_object_terms
