"""With a filter that leaves none of the located photos, the map says so and offers one action;
the counts under the filter are the server's ("12 of 63 located photos match")."""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_map_view import CANVAS, _cells, _meta, _serve_map


def _serve(stub, page, app_url, matching: int, unlocated: int):
    _serve_map(stub, page, app_url, _meta(app_url, n_located=63, n_without_location=28))

    def cells(_request, _m):
        body = _cells()
        body['n_located_matching'] = matching
        body['n_unlocated_matching'] = unlocated
        return (200, body, 'application/geo+json')

    stub.on('GET', r'/map/cells(\?|$)', cells)


def test_the_header_counts_the_photos_that_match_the_filter(stub, page, app_url):
    _serve(stub, page, app_url, matching=12, unlocated=9)
    page.goto(f'{app_url}/p/default/workspace?view=map&class_name=widget')
    page.locator(CANVAS).wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('map-located')).to_have_text(
        '12 of 63 located photos match', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('map-without-location')).to_have_text('9 without location')
    expect(page.get_by_test_id('map-none-match')).to_have_count(0)


def test_no_match_shows_an_overlay_with_clear_filters(stub, page, app_url):
    _serve(stub, page, app_url, matching=0, unlocated=0)
    page.goto(f'{app_url}/p/default/workspace?view=map&class_name=widget')
    overlay = page.get_by_test_id('map-none-match')
    expect(overlay).to_contain_text(
        'None of the 63 located photos match the filter.', timeout=ACTION_TIMEOUT_MS
    )
    page.get_by_test_id('map-clear-filters').click()
    page.wait_for_function(
        '() => !location.search.includes("class_name")', timeout=ACTION_TIMEOUT_MS
    )
