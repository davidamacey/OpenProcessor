"""Opening the Workspace's VLM menu, which exists only while a VLM is served active."""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


ACTIVE_VLM = {'source': 'saved', 'active': {'name': 'local', 'revision': 1}}
WORKSPACE = '/p/default/workspace?view=clusters'


def open_vlm_menu(stub, page, app_url: str, path: str = WORKSPACE) -> None:
    stub.on('GET', r'/vlm/endpoints/active(\?|$)', ACTIVE_VLM)
    page.goto(f'{app_url}{path}')
    toggle = page.get_by_test_id('vlm-menu-toggle')
    expect(toggle).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    toggle.click()
    expect(toggle).to_have_attribute('aria-expanded', 'true')


def serve_workspace_reads(stub) -> None:
    """The clusters view's reads other than /classes (the test serves its own), with empty data, for tests that only need the menu."""
    from test_cluster_discard import CLUSTERS

    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 60, 'crops': []})
