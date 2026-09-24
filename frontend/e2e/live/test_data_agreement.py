"""Live read-only tier: the UI shows what the API actually serves.

Every test here reads a number or a set of labels off the real rendered
page and compares it against a direct `GET {API_PREFIX}/...` call. A
concurrent actor (another agent's UI-write smoke test, per this task's
brief) can change dataset-derived counts between our API read and the
browser's — `agrees_with_retry` (conftest.py) re-reads the API once on
a mismatch and accepts either value, so this suite only goes red on a
genuine frontend/backend disagreement, not a race with someone else's
writes.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from conftest import agrees_with_retry, api_get, wait_for_stable_text
from fixtures.wire import REGION_TAB_URL_ID

TOTAL_RE = re.compile(r"·\s*([\d,]+)\s*total")  # "1 / 18 loaded · 18 total"


def _queue_total(page: Any) -> int:
    text = page.locator('[data-testid="queue-counter"]').inner_text()
    m = TOTAL_RE.search(text)
    assert m, f"queue-counter text didn't match the expected '· N total' shape: {text!r}"
    return int(m.group(1).replace(",", ""))


def test_dashboard_cluster_count_agrees_with_stats_dataset(guarded_page: Any, live_url: str) -> None:
    page = guarded_page.page
    page.goto(f"{live_url}/dashboard", wait_until="domcontentloaded")
    # `data-testid="dataset-cluster-count"` (DatasetStats.svelte) is only
    # in source as of this change — the deployment at CROPWRIGHT_LIVE_URL
    # is a prior build without it (this tier is instructed not to rebuild
    # or restart the container), so select by the stable dt/dd label
    # pairing instead, which works against both the old and new build.
    dt = page.locator('dt:has-text("Clusters (total now)")')
    dt.wait_for(timeout=15_000)
    dd = dt.locator("xpath=following-sibling::dd[1]")
    # SSE-pushed; the first snapshot frame can take a beat past DOM mount.
    page.wait_for_function(
        """
        () => {
          const dt = Array.from(document.querySelectorAll('dt'))
            .find((el) => el.textContent.includes('Clusters (total now)'));
          const dd = dt && dt.nextElementSibling;
          const text = dd && dd.textContent.trim();
          return !!text && text !== '0';
        }
        """,
        timeout=15_000,
    )
    displayed = int(dd.inner_text().strip().replace(",", ""))

    ok, first, second = agrees_with_retry(
        live_url, "/stats/dataset", ("clusters", "cluster_count"), displayed
    )
    assert ok, (
        f"dashboard 'Clusters (total now)' showed {displayed}, but "
        f"GET {{API_PREFIX}}/stats/dataset clusters.cluster_count was "
        f"{first} (and, on retry, {second})"
    )


def test_region_queue_total_agrees_with_review_regions(guarded_page: Any, live_url: str) -> None:
    page = guarded_page.page
    page.goto(f"{live_url}/review?tab={REGION_TAB_URL_ID}", wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="queue-counter"]', timeout=15_000)
    page.wait_for_function(
        """
        () => {
          const el = document.querySelector('[data-testid="queue-counter"]');
          return !!el && /total/.test(el.textContent);
        }
        """,
        timeout=15_000,
    )
    wait_for_stable_text(page, '[data-testid="queue-counter"]')
    displayed = _queue_total(page)

    ok, first, second = agrees_with_retry(live_url, "/review/regions", ("total",), displayed)
    assert ok, (
        f"/review?tab={REGION_TAB_URL_ID} queue-counter showed total={displayed}, but "
        f"GET {{API_PREFIX}}/review/regions total was {first} (and, on retry, {second})"
    )


def test_region_queue_total_agrees_with_filtered_region_status(guarded_page: Any, live_url: str) -> None:
    page = guarded_page.page
    page.goto(
        f"{live_url}/review?tab={REGION_TAB_URL_ID}&region_status=verify_rejected",
        wait_until="domcontentloaded",
    )
    page.wait_for_selector('[data-testid="queue-counter"]', timeout=15_000)
    page.wait_for_function(
        """
        () => {
          const el = document.querySelector('[data-testid="queue-counter"]');
          return !!el && /total/.test(el.textContent);
        }
        """,
        timeout=15_000,
    )
    # ?region_status= only takes effect once GET {API_PREFIX}/review/tabs's
    # filter_specs has loaded — the counter briefly shows the *unfiltered*
    # total first (see CLAUDE.md's "Served per-tab filters"). Wait for it
    # to settle before reading it.
    wait_for_stable_text(page, '[data-testid="queue-counter"]')
    displayed = _queue_total(page)

    ok, first, second = agrees_with_retry(
        live_url, "/review/regions?region_status=verify_rejected", ("total",), displayed
    )
    assert ok, (
        f"/review?tab={REGION_TAB_URL_ID}&region_status=verify_rejected queue-counter showed "
        f"total={displayed}, but GET {{API_PREFIX}}/review/regions?"
        f"region_status=verify_rejected total was {first} (and, on retry, {second})"
    )


def test_regions_filter_spec_select_matches_served_options(guarded_page: Any, live_url: str) -> None:
    tabs = api_get(live_url, "/review/tabs")["tabs"]
    regions_tab = next((t for t in tabs if t["id"] == "regions"), None)
    if regions_tab is None or not regions_tab.get("filter_specs"):
        pytest.skip("backend serves no filter_specs for the regions tab right now")
    specs = regions_tab["filter_specs"]

    page = guarded_page.page
    page.goto(f"{live_url}/review?tab={REGION_TAB_URL_ID}", wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="queue-counter"]', timeout=15_000)

    for spec in specs:
        select = page.locator(f'label:has-text("{spec["label"]}") select')
        select.wait_for(timeout=15_000)
        rendered_labels = select.locator("option").all_inner_texts()
        served_labels = [opt["label"] for opt in spec["options"]]
        assert rendered_labels == served_labels, (
            f"filter_specs[{spec['param']}]: rendered options {rendered_labels} != "
            f"served options {served_labels}"
        )


def _resolve_reason_label(reason_id: str, vocab_entries: list[dict]) -> str | None:
    """Mirror `regionVocabularyStore.rejectionReasonLabel` (src/lib/
    stores/regionVocabulary.svelte.ts): exact match first, then
    longest-prefix among `match: "prefix"` entries, filling
    `label_template`'s `{detail}`. Returns None (never the raw id) when
    nothing in the vocabulary matches — a free-text human reason, which
    this live dataset isn't exercising today.
    """
    for entry in vocab_entries:
        if entry["match"] == "exact" and entry["id"] == reason_id:
            return entry["label"]
    best: dict | None = None
    for entry in vocab_entries:
        if entry["match"] == "prefix" and reason_id.startswith(entry["id"]):
            if best is None or len(entry["id"]) > len(best["id"]):
                best = entry
    if best is None:
        return None
    detail = reason_id[len(best["id"]) :]
    template = best.get("label_template")
    return template.format(detail=detail) if template else best["label"]


def test_rejection_reason_label_renders_for_a_live_item(guarded_page: Any, live_url: str) -> None:
    vocab = api_get(live_url, "/regions/vocabulary")
    vocab_entries = vocab.get("rejection_reasons", [])
    if not vocab_entries:
        pytest.skip("backend serves no rejection_reasons vocabulary right now")

    rejected = api_get(live_url, "/review/regions?region_status=verify_rejected&page_size=50")
    items = rejected.get("items", [])
    if not items:
        pytest.skip("no verify_rejected items in the live dataset right now")

    expected_labels = {
        _resolve_reason_label(item["region_rejection_reason"], vocab_entries)
        for item in items
        if item.get("region_rejection_reason")
    }
    expected_labels.discard(None)
    if not expected_labels:
        pytest.skip("no verify_rejected item resolves to a served rejection_reasons label")

    page = guarded_page.page
    page.goto(
        f"{live_url}/review?tab={REGION_TAB_URL_ID}&region_status=verify_rejected",
        wait_until="domcontentloaded",
    )
    page.wait_for_selector('[data-testid="queue-counter"]', timeout=15_000)
    reason_dd = page.locator(
        'xpath=//dt[contains(text(),"Needs review") or contains(text(),"Rejection")]'
        "/following-sibling::dd[1]"
    )
    reason_dd.wait_for(timeout=15_000)
    displayed = reason_dd.inner_text().strip()

    assert displayed, "the reason/needs-review row rendered empty text"
    assert displayed in expected_labels, (
        f"displayed rejection/needs-review label {displayed!r} is not one of the "
        f"labels {sorted(expected_labels)} the served rejection_reasons vocabulary "
        f"maps this cohort's items to"
    )
