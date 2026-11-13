"""Links from the Data quality pages and the Dashboard open the Workspace with a filter in the URL:
`audit_outcome` (who was right on an audited crop) and `near_dup=collapsed` (one photo per
near-duplicate group). The Workspace applies both, shows them, and sends them to the lists."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_gallery_view import serve as serve_gallery


def test_a_quality_link_filters_the_gallery_by_outcome_and_collapsed_duplicates(
    stub, page, app_url
):
    queries: list[dict] = []
    serve_gallery(stub, queries)
    page.goto(
        f'{app_url}/p/default/workspace?view=gallery&audit_outcome=both_wrong&near_dup=collapsed'
    )
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(
        page.get_by_test_id('item-filter-chip').filter(has_text='Audit: Both wrong')
    ).to_be_visible()
    expect(page.get_by_role('button', name='Duplicates: One per group')).to_be_visible()
    sent = queries[-1]
    assert sent['audit_outcome'] == ['both_wrong'], sent
    assert sent['near_dup'] == ['collapsed'], sent


def test_the_outcome_filter_reaches_the_detections_list_and_clears_with_its_chip(
    stub, page, app_url
):
    seen: list[dict] = []

    def crops(request, _match):
        seen.append(parse_qs(urlparse(request.url).query))
        return (200, {'total': 0, 'next_cursor': None, 'limit': 60, 'crops': []})

    serve_gallery(stub, [])
    stub.on('GET', r'/crops(\?|$)', crops)
    page.goto(f'{app_url}/p/default/workspace?view=detections&audit_outcome=agree')
    chip = page.get_by_test_id('item-filter-chip').filter(has_text='Audit: Agree')
    expect(chip).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert seen[-1]['audit_outcome'] == ['agree'], seen[-1]
    chip.click()
    expect(chip).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    assert 'audit_outcome' not in page.url, page.url
