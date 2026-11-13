"""The header follows the loop, and the breadcrumb is only for object pages.

Dashboard | Data (Ingest, Classes) | Workspace | Data quality | Model (Export, Train, Bake-off,
Models) | Settings. Section and tab pages show no breadcrumb; the image editor's last crumb is the
served file name.
"""

from __future__ import annotations

from fixtures.image_editor import Editor
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_app_shell import DESKTOP, _stub_classes


def test_header_order_follows_the_loop_and_export_is_in_model(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/classes')
    nav = page.get_by_test_id('primary-nav')
    nav.wait_for(timeout=ACTION_TIMEOUT_MS)
    order = nav.locator('> a, > div > button, > button').all_inner_texts()
    assert [t.strip() for t in order] == [
        'Dashboard',
        'Data',
        'Workspace',
        'Data quality',
        'Model',
        'Settings',
    ]
    expect(nav.get_by_role('link', name='Workspace')).to_have_attribute(
        'href', '/p/default/workspace'
    )
    nav.get_by_role('button', name='Model').click()
    expect(nav.get_by_role('menuitem')).to_have_text(['Export', 'Train', 'Bake-off', 'Models'])


def test_no_breadcrumb_on_a_section_page(stub, page, app_url):
    _stub_classes(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/classes')
    page.get_by_test_id('primary-nav').wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('breadcrumb')).to_have_count(0)


def test_the_editor_breadcrumb_shows_the_file_name(stub, page, app_url):
    page.set_viewport_size(DESKTOP)
    Editor(stub).open(page, app_url)
    crumb = page.get_by_test_id('breadcrumb')
    expect(crumb).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(crumb).to_contain_text('Images')
    expect(crumb.locator('[aria-current="page"]')).not_to_have_text('img-1')
