"""One primary per view state: exactly one visible filled accent button in each Workspace view,
with nothing selected. (A selection swaps the toolbar for the bulk bar with its own one primary;
that is pinned in test_cluster_scope_toolbar.)"""

from __future__ import annotations

import pytest
from fixtures.contract import cluster_card, cluster_list
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_cluster_scope_toolbar import CLASSES, UIDS
from test_gallery_view import serve as serve_gallery
from test_map_view import CANVAS, _meta, _serve_map
from test_table_view import serve as serve_table


PRIMARY = '[data-testid=workspace-view] .btn-primary:visible'


def _common(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 60, 'crops': []})


@pytest.mark.parametrize('view', ['gallery', 'detections', 'table', 'clusters'])
def test_each_list_view_has_exactly_one_primary(stub, page, app_url, view):
    serve_gallery(stub, [])
    _common(stub)
    stub.on(
        'GET',
        r'/clusters(\?|$)',
        cluster_list(cluster_card(UIDS[0], size=3, dominant_class_name='widget')),
    )
    serve_table(stub)
    page.goto(f'{app_url}/p/default/workspace?view={view}')
    expect(
        page.locator('[data-testid=workspace-view] button, [data-testid=workspace-view] a').first
    ).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.locator(PRIMARY)).to_have_count(1, timeout=ACTION_TIMEOUT_MS)


def test_the_map_has_exactly_one_primary(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    page.goto(f'{app_url}/p/default/workspace?view=map')
    page.locator(CANVAS).wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.locator(PRIMARY)).to_have_count(1)
