"""OpenProcessor W10 leftovers (docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
§5.5): the `imported` review tab, URL-seeded `import_id` / `combine_conflict`
filters, lock badges, and image Reprocess from a browse card. Payloads follow
the vendored contract in the neutral widget domain; the fail-closed stub means
any request the page makes that a test did not expect fails the test.

  1. `test_imported_tab_and_import_id_chip` — the tab appears when served, the
     URL-seeded `import_id` chip is shown and sent, removing it refetches.
  2. `test_imported_tab_absent_when_not_served` — no tab; a `?tab=imported`
     link falls back to All with a visible reason.
  3. `test_url_filter_not_sent_when_the_tab_does_not_serve_it`.
  4. `test_combine_conflict_chip_from_the_url_is_sent`.
  5. `test_lock_badge_only_on_a_locked_label`.
  6. `test_image_reprocess_from_a_cluster_card` — body asserted, served items
     adopted into the grid.
  7. `test_import_provenance_rows_in_details`.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from conftest import ACTION_TIMEOUT_MS, expect_handled
from playwright.sync_api import expect

from fixtures.wire import make_item, review_tab, review_tabs
from test_dataset_import import FORMATS, serve_cluster

CLASSES = [
    {
        "class_id": 2,
        "class_name": "widget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 10,
        "validated_count": 5,
        "cluster_size": 3,
        "deprecated": False,
    },
    {
        "class_id": 5,
        "class_name": "gadget",
        "kind": "item",
        "group": None,
        "hotkey_letter": None,
        "sample_count": 4,
        "validated_count": 1,
        "cluster_size": 4,
        "deprecated": False,
    },
]

EMPTY_STATE = {
    "has_probe_predictions": True,
    "has_item_scores": True,
    "has_imported_labels": True,
}

IMPORT_ID = "imp_20260927T120000_1a2b3c4d"


def tabs_with_imported(**imported_over: Any) -> dict[str, Any]:
    return review_tabs(
        review_tab("all", "All", filters=["class_id", "source", "combine_conflict", "import_id"]),
        review_tab(
            "imported",
            "Imported labels",
            description="Items whose labels came from a dataset import",
            filters=["class_id", "source", "import_id", "dataset_split", "on_negative_frame"],
            **imported_over,
        ),
        empty_state=EMPTY_STATE,
    )


def review_item(i: int, **over: Any) -> dict[str, Any]:
    item = make_item(
        crop_id=f"crop-{i}",
        image_id=f"img-{i}",
        class_id=2,
        class_name="widget",
        label_validated=True,
        label_source="import",
        **over,
    )
    item["reason"] = "imported"
    return item


def serve_review(stub: Any, tabs: dict[str, Any]) -> list[dict[str, list[str]]]:
    """Registers the shell reads and a `/review/{tab}` handler that records
    each request's path and query."""
    seen: list[dict[str, list[str]]] = []
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    stub.on("GET", r"/review/tabs(\?|$)", tabs)

    def review_handler(request: Any, _m: Any):
        u = urlparse(request.url)
        seen.append({"path": [u.path], **parse_qs(u.query)})
        items = [review_item(i) for i in range(2)]
        return (200, {"items": items, "total": 2, "page": 1, "page_size": 30})

    stub.on("GET", r"/review/(?!tabs)", review_handler)
    return seen


def review_requests(seen: list[dict[str, list[str]]], tab: str) -> list[dict[str, list[str]]]:
    return [s for s in seen if s["path"][0].endswith(f"/review/{tab}")]


def test_imported_tab_and_import_id_chip(stub, page, app_url):
    seen = serve_review(stub, tabs_with_imported())

    page.goto(f"{app_url}/p/default/review?tab=imported&import_id={IMPORT_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    tab = page.get_by_role("button", name="Imported labels")
    expect(tab).to_have_attribute("data-active", "true", timeout=ACTION_TIMEOUT_MS)
    chip = page.get_by_test_id("filter-chip-import-id")
    expect(chip).to_contain_text(f"Import {IMPORT_ID}")
    first = review_requests(seen, "imported")
    assert first, seen
    assert first[0]["import_id"] == [IMPORT_ID], first[0]

    # Removing the chip refetches without the filter and clears the URL.
    with page.expect_request(lambda r: "/review/imported" in r.url and "import_id" not in r.url):
        chip.click()
    expect(chip).to_have_count(0)
    assert "import_id" not in page.url, page.url


def test_imported_tab_absent_when_not_served(stub, page, app_url):
    serve_review(stub, review_tabs(review_tab("all", "All"), empty_state=EMPTY_STATE))

    page.goto(f"{app_url}/p/default/review?tab=all")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role("button", name="All", exact=True).first).to_be_visible()
    assert page.get_by_role("button", name="Imported", exact=False).count() == 0

    # A bookmark to the tab falls back to All, with the reason shown.
    page.goto(f"{app_url}/p/default/review?tab=imported")
    notice = page.get_by_test_id("tab-unavailable")
    expect(notice).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(notice).to_contain_text("imported")
    expect(page.get_by_role("button", name="All", exact=True).first).to_have_attribute(
        "data-active", "true"
    )


def test_url_filter_not_sent_when_the_tab_does_not_serve_it(stub, page, app_url):
    # `all` serves no `import_id` here.
    seen = serve_review(
        stub, review_tabs(review_tab("all", "All", filters=["class_id", "source"]), empty_state=EMPTY_STATE)
    )
    page.goto(f"{app_url}/p/default/review?tab=all&import_id={IMPORT_ID}")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    reqs = review_requests(seen, "all")
    assert reqs, seen
    assert all("import_id" not in r for r in reqs), reqs
    expect(page.get_by_test_id("filter-chip-import-id")).to_have_count(0)


def test_combine_conflict_chip_from_the_url_is_sent(stub, page, app_url):
    seen = serve_review(stub, tabs_with_imported())
    page.goto(f"{app_url}/p/default/review?tab=all&combine_conflict=true")
    page.get_by_test_id("queue-counter").first.wait_for(timeout=ACTION_TIMEOUT_MS)
    chip = page.get_by_test_id("filter-chip-combine-conflict")
    expect(chip).to_contain_text("Combine conflicts only", timeout=ACTION_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=ACTION_TIMEOUT_MS)
    reqs = review_requests(seen, "all")
    assert reqs and reqs[0]["combine_conflict"] == ["true"], reqs

    # Switching tab clears it from the state and the URL.
    with page.expect_request(lambda r: "/review/imported" in r.url):
        page.get_by_role("button", name="Imported labels").click()
    expect(chip).to_have_count(0)
    assert "combine_conflict" not in page.url, page.url
    imported = review_requests(seen, "imported")
    assert imported and all("combine_conflict" not in r for r in imported), imported


def serve_locked_cluster(stub: Any, locked_ids: set[str]) -> None:
    serve_cluster(stub)
    stub.on("GET", r"(?<!/stats)/classes(\?|$)", {"classes": CLASSES})
    crops = [
        make_item(
            crop_id=f"crop-{i}",
            image_id=f"img-{i}",
            class_id=2,
            class_name="widget",
            cluster_id=2,
            label_validated=False,
            label_source="model_suggestion",
            label_locked=f"crop-{i}" in locked_ids,
            thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        )
        for i in range(3)
    ]
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": crops})


def test_lock_badge_only_on_a_locked_label(stub, page, app_url):
    serve_locked_cluster(stub, {"crop-1"})
    page.goto(f"{app_url}/p/default/clusters/2")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    badges = page.get_by_test_id("label-locked-badge")
    expect(badges).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    expect(badges.first).to_have_attribute("title", "Label locked")


def test_image_reprocess_from_a_cluster_card(stub, page, app_url):
    serve_locked_cluster(stub, set())
    stub.on("GET", r"/datasets/formats(\?|$)", FORMATS)
    bodies: list[tuple[str, Any]] = []

    def reprocess(request: Any, _m: Any):
        bodies.append((urlparse(request.url).path, request.post_data_json))
        updated = make_item(
            crop_id="crop-0",
            image_id="img-0",
            class_id=5,
            class_name="gadget",
            cluster_id=2,
            label_validated=False,
            label_source="model_suggestion",
            thumbnail_url="/curation/crops/crop-0/thumbnail",
        )
        return (
            200,
            {
                "dry_run": False,
                "scopes": [{"scope": "detect", "selected": 1, "locked_skipped": 0, "queued": 1, "breakdown": []}],
                "job": None,
                "items": [updated],
            },
        )

    stub.on("POST", r"/images/img-0/reprocess$", reprocess)

    page.goto(f"{app_url}/p/default/clusters/2")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    names = page.get_by_test_id("class-name")
    expect(names).to_have_text(["widget", "widget", "widget"], timeout=ACTION_TIMEOUT_MS)

    card = page.locator('[role="button"][aria-pressed]').first
    card.hover()
    card.get_by_role("button", name="Expand").click()
    page.get_by_test_id("card-reprocess-image").get_by_test_id("reprocess-open").click()
    dialog = page.get_by_role("dialog", name="Reprocess")
    expect(dialog.get_by_role("heading", name="Reprocess image")).to_be_visible()
    dialog.get_by_label("Detect", exact=True).check()
    with expect_handled(page, lambda r: r.method == "POST" and r.url.endswith("/images/img-0/reprocess")):
        dialog.get_by_role("button", name="Reprocess", exact=True).click()
    assert [b for _p, b in bodies] == [{"scopes": ["detect"], "dry_run": False}], bodies
    assert bodies[0][0].endswith("/images/img-0/reprocess"), bodies
    expect(dialog.get_by_test_id("reprocess-result")).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # The served item replaced the grid's copy of that crop only.
    dialog.get_by_role("button", name="Close").click()
    page.get_by_role("button", name="Close", exact=True).click()
    expect(names).to_have_text(["gadget", "widget", "widget"], timeout=ACTION_TIMEOUT_MS)


def test_import_provenance_rows_in_details(stub, page, app_url):
    serve_locked_cluster(stub, set())
    stub.on("GET", r"/datasets/formats(\?|$)", FORMATS)
    stub.on("GET", r"/crops/[^/]+/history$", {"crop_id": "crop-0", "entries": []})
    imported = make_item(
        crop_id="crop-0",
        image_id="img-0",
        class_id=2,
        class_name="widget",
        cluster_id=2,
        label_validated=True,
        label_source="import",
        label_locked=True,
        import_ids=[IMPORT_ID],
        dataset_split="train",
        imported_at="2026-09-27T12:00:03Z",
        on_negative_frame=True,
        proposal_chain=["import:" + IMPORT_ID],
        thumbnail_url="/curation/crops/crop-0/thumbnail",
    )
    others = [
        make_item(
            crop_id=f"crop-{i}",
            image_id=f"img-{i}",
            class_id=2,
            class_name="widget",
            cluster_id=2,
            thumbnail_url=f"/curation/crops/crop-{i}/thumbnail",
        )
        for i in (1, 2)
    ]
    stub.on("GET", r"/crops(\?|$)", {"total": 3, "page": 1, "page_size": 60, "crops": [imported, *others]})

    page.goto(f"{app_url}/p/default/clusters/2")
    page.wait_for_selector("img", timeout=ACTION_TIMEOUT_MS)
    card = page.locator('[role="button"][aria-pressed]').first
    card.hover()
    card.get_by_role("button", name="Show crop details").click()
    expect(page.get_by_test_id("import-label-locked")).to_have_text("Locked", timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id("import-split")).to_have_text("train")
    link = page.get_by_test_id("import-ids").get_by_role("link", name=IMPORT_ID)
    expect(link).to_have_attribute("href", f"/p/default/datasets/imports/{IMPORT_ID}")
    expect(page.get_by_test_id("import-negative-frame")).to_be_visible()
    expect(page.get_by_test_id("import-proposal-chain")).to_contain_text("import:" + IMPORT_ID)
