"""DQ-M11 fix (dq-queues cutover, 2026-09-24): `GET {API_PREFIX}/review/
new_class_proposals/summary` now splits terms into `top_terms`
(`flag: null`, one-click actionable) and `flagged_terms`
(`existing_class`/`generic_parent`/`non_object`, no create action). This
proves the real browser-rendered /classes page puts flagged terms in
their own collapsed section with the served reason, and offers no
"Create class & assign" for them.
"""

from __future__ import annotations

import re

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


def test_classes_flagged_terms_are_listed_with_no_create_action(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on(
        "GET",
        r"/review/new_class_proposals/summary(\?|$)",
        PROPOSALS_SUMMARY,
    )
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})

    page.goto(f"{app_url}/classes")

    # F-53: the proposals list sits below the registry in a collapsed
    # <details>; open it first.
    section = page.get_by_test_id("proposals-section")
    section.wait_for(timeout=15000)
    section.locator("summary").click()

    # top_terms: one actionable row with "Create class & assign".
    assert page.get_by_text("classic_car").count() >= 1
    assert page.get_by_text("Create class & assign").count() == 1

    # Visual audit 2026-09-24 L2: flagged terms share the one list (sorted
    # by count) instead of a collapsed, action-less <details>; each shows
    # its served reason and never a create action.
    rows = page.get_by_test_id("proposal-row")
    rows.first.wait_for(timeout=15000)
    for label, reason in (
        ("motorcycle", "generic parent"),
        ("suv", "existing class"),
        ("abstract_blur", "not an object"),
    ):
        row = rows.filter(
            has=page.locator("div.text-sm", has_text=re.compile(rf"^{label}$"))
        ).first
        text = row.inner_text()
        assert reason in text, (label, text)
        assert row.get_by_text("Create class & assign").count() == 0, label

    # existing_class term gets a map-to-class button naming the class;
    # the other flagged terms get "map to existing".
    assert page.get_by_role("button", name="Map to suv").count() == 1
    assert rows.filter(has_text="motorcycle").first.locator("select").count() == 1

    # without_term and term_rules surface as help text somewhere on the
    # page (not silently dropped).
    body_text = page.locator("body").inner_text()
    assert "12" in body_text  # without_term count
    assert "abstract_blur" in body_text  # term_rules non_object_terms


def test_classes_table_fits_800px(stub, page, app_url):
    """Visual audit 2026-09-24 L5: at 800px the class table was 932px wide
    and scrolled sideways; the ID and Added columns now hide below lg."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", PROPOSALS_SUMMARY)
    stub.on(
        "GET",
        r"/test_holdout/stats(\?|$)",
        {"total": 5, "by_class": [{"key": CLASSES[0]["id"], "doc_count": 5}]},
    )
    page.set_viewport_size({"width": 800, "height": 1000})
    page.goto(f"{app_url}/classes")
    table = page.locator("table").first
    table.wait_for(timeout=15000)
    page.get_by_test_id("validated-test-suffix").first.wait_for(timeout=10000)
    width = table.evaluate("t => t.scrollWidth")
    container = table.evaluate("t => t.parentElement.clientWidth")
    assert width <= container + 1, (width, container)
    headers = [h.strip().lower() for h in table.locator("thead th:visible").all_inner_texts()]
    assert "id" not in headers and "added" not in headers, headers
    assert "validated" in headers, headers


def test_class_table_keeps_its_height_under_a_long_proposals_list(stub, page, app_url):
    """With ~60 proposal rows above it, the flex-1/overflow-auto table
    section used to collapse to almost nothing (seen live at 800px)."""
    many = dict(PROPOSALS_SUMMARY)
    many["top_terms"] = [
        {
            "label": f"term_{i:02d}",
            "count": 1,
            "sample_crop_ids": [],
            "flag": None,
            "class_id": None,
        }
        for i in range(60)
    ]
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", many)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    page.set_viewport_size({"width": 800, "height": 1000})
    page.goto(f"{app_url}/classes")
    table = page.locator("table").first
    table.wait_for(timeout=15000)
    section = page.get_by_test_id("proposals-section")
    section.wait_for(timeout=15000)
    section.locator("summary").click()
    page.get_by_test_id("proposal-row").nth(59).wait_for(timeout=10000)
    # F-53: the registry renders first and is never squeezed by the list
    # (no inner scroll pane: its section shows the whole table).
    dims = table.evaluate(
        "t => ({top: t.getBoundingClientRect().top,"
        " proposalsTop: document.querySelector('[data-testid=proposals-section]')"
        ".getBoundingClientRect().top,"
        " sectionH: t.closest('section').clientHeight, tableH: t.scrollHeight})"
    )
    assert dims["top"] < dims["proposalsTop"], dims
    assert dims["sectionH"] >= dims["tableH"], dims
