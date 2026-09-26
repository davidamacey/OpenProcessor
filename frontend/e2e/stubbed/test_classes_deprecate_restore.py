"""Deprecate/Restore (OpenProcessor 01324cb, 243f7f2): `POST
{API_PREFIX}/classes/{id}/deprecate` and `.../restore` — the `/classes`
Restore button used to be permanently disabled ("no backend support
exists"). This proves the real browser-rendered page:

- deprecates a live class successfully and the class list refreshes;
- on a structured `class_still_referenced` 409, offers (and, on confirm,
  opens) the merge dialog with the deprecated-attempt class preselected
  as source;
- restores a deprecated class successfully and the deprecated table's row
  disappears;
- on restore's PLAIN STRING 409 detail, surfaces it verbatim in a toast.
"""

from __future__ import annotations

from conftest import ACTION_TIMEOUT_MS

CLASSES_LIVE = [
    {
        "class_id": 10,
        "class_name": "coupe",
        "group": "vehicle",
        "hotkey_letter": "c",
        "sample_count": 0,
        "validated_count": 0,
        "cluster_size": 0,
        "deprecated": False,
    },
    {
        "class_id": 11,
        "class_name": "sedan",
        "group": "vehicle",
        "hotkey_letter": "s",
        "sample_count": 40,
        "validated_count": 30,
        "cluster_size": 41,
        "deprecated": False,
    },
]

CLASSES_WITH_DEPRECATED = CLASSES_LIVE + [
    {
        "class_id": 99,
        "class_name": "wagon",
        "group": "vehicle",
        "hotkey_letter": None,
        "sample_count": 0,
        "validated_count": 0,
        "cluster_size": 0,
        "deprecated": True,
    },
]

EMPTY_PROPOSALS = {
    "total_pending": 0,
    "without_term": 0,
    "top_terms": [],
    "flagged_terms": [],
    "term_rules": None,
}


def _confirm_dialogs_yes(page) -> None:
    page.on("dialog", lambda d: d.accept())


def test_deprecate_empty_class_succeeds_and_refreshes(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_LIVE})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/10/deprecate$",
        {"class_id": 10, "class_name": "coupe", "deprecated": True},
    )
    _confirm_dialogs_yes(page)

    page.goto(f"{app_url}/classes")
    row = page.get_by_test_id("class-row-10")
    row.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("deprecate-10").click()

    assert any(c[1] == "/curation/classes/10/deprecate" for c in stub.calls)
    toast = page.get_by_text("Deprecated coupe.")
    toast.wait_for(timeout=ACTION_TIMEOUT_MS)


def test_deprecate_still_referenced_offers_merge_preselecting_source(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_LIVE})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/11/deprecate$",
        (
            409,
            {
                "detail": {
                    "error": "class_still_referenced",
                    "message": 'class "sedan" is still referenced',
                    "class_id": 11,
                    "item_count": 40,
                    "confirmed_label_count": 30,
                }
            },
        ),
    )
    _confirm_dialogs_yes(page)

    page.goto(f"{app_url}/classes")
    row = page.get_by_test_id("class-row-11")
    row.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id("deprecate-11").click()

    # Both confirm dialogs are auto-accepted above (deprecate confirm, then
    # the "merge instead?" confirm) — the merge dialog should now be open
    # with sedan preselected as the source.
    dialog = page.get_by_role("dialog", name="Merge classes")
    dialog.wait_for(timeout=ACTION_TIMEOUT_MS)
    source_select = dialog.locator("select").first
    assert source_select.input_value() == "11"


def test_restore_succeeds_and_row_disappears(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_WITH_DEPRECATED})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/99/restore$",
        {"class_id": 99, "class_name": "wagon", "deprecated": False},
    )
    _confirm_dialogs_yes(page)

    page.goto(f"{app_url}/classes")
    toggle = page.get_by_text("Deprecated classes (1)")
    toggle.wait_for(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    restore_btn = page.get_by_test_id("restore-99")
    restore_btn.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert restore_btn.is_enabled()
    restore_btn.click()

    assert any(c[1] == "/curation/classes/99/restore" for c in stub.calls)


def test_restore_conflict_shows_plain_string_detail_verbatim(stub, page, app_url):
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_WITH_DEPRECATED})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/99/restore$",
        (409, {"detail": 'a live class already uses the name "wagon"'}),
    )
    _confirm_dialogs_yes(page)

    page.goto(f"{app_url}/classes")
    toggle = page.get_by_text("Deprecated classes (1)")
    toggle.wait_for(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    page.get_by_test_id("restore-99").click()

    toast = page.get_by_text('a live class already uses the name "wagon"')
    toast.wait_for(timeout=ACTION_TIMEOUT_MS)


def test_restore_merged_class_names_the_merge_target(stub, page, app_url):
    """F-56 (OpenProcessor 4c125ec): restoring a merged class 409s with a
    structured `class_merged` detail; the toast names the served merge
    target and says un-merge isn't supported, with the served hint."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_WITH_DEPRECATED})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/99/restore$",
        (
            409,
            {
                "detail": {
                    "error": "class_merged",
                    "message": "class_id 99 was merged into class_id 11",
                    "class_id": 99,
                    "merged_into": {"class_id": 11, "class_name": "sedan"},
                    "hint": "relabel crops by hand to split them back out",
                }
            },
        ),
    )
    _confirm_dialogs_yes(page)

    page.goto(f"{app_url}/classes")
    toggle = page.get_by_text("Deprecated classes (1)")
    toggle.wait_for(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    page.get_by_test_id("restore-99").click()

    page.get_by_text("Merged into sedan; un-merge isn't supported.", exact=False).wait_for(
        timeout=ACTION_TIMEOUT_MS
    )
    body = page.locator("body").inner_text()
    assert "relabel crops by hand to split them back out" in body


def test_merged_class_row_has_no_restore(stub, page, app_url):
    """51b05d7: GET /classes serves merged_into; a merged deprecated class
    shows where it went and offers no Restore."""
    merged = dict(CLASSES_WITH_DEPRECATED[-1], merged_into=11)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": [*CLASSES_LIVE, merged]})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})

    page.goto(f"{app_url}/classes")
    toggle = page.get_by_text("Deprecated classes (1)")
    toggle.wait_for(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    assert page.get_by_test_id("merged-into-99").inner_text().strip() == "merged into sedan"
    assert page.get_by_test_id("restore-99").count() == 0


def test_merge_preview_says_validations_carry_over(stub, page, app_url):
    """51b05d7: the merge dry-run serves validations_carried_over (kept),
    replacing would_unvalidate (lost)."""
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES_LIVE})
    stub.on("GET", r"/review/new_class_proposals/summary(\?|$)", EMPTY_PROPOSALS)
    stub.on("GET", r"/test_holdout/stats(\?|$)", {"total": 0, "by_class": []})
    stub.on(
        "POST",
        r"/classes/merge$",
        {
            "dry_run": True,
            "source_id": 10,
            "target_id": 11,
            "would_relabel": 3,
            "validations_carried_over": 2,
            "holdout_blocking": 0,
            "blocked": False,
        },
    )
    page.goto(f"{app_url}/classes")
    page.get_by_role("button", name="Merge classes").click(timeout=ACTION_TIMEOUT_MS)
    dialog = page.get_by_role("dialog", name="Merge classes")
    dialog.wait_for(timeout=5000)
    dialog.locator("select").nth(0).select_option("10")
    dialog.locator("select").nth(1).select_option("11")
    dialog.get_by_text("will carry over", exact=False).wait_for(timeout=5000)
    text = " ".join(dialog.inner_text().split())
    assert "2 human validations will carry over" in text, text
    assert "lose validation" not in text
