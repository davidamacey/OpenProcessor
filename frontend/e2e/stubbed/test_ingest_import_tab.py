"""Import dataset is a real Ingest tab, and the import job page sits under Ingest.

- the tab renders inside the Ingest layout: same width, the tab strip and the Pipeline rail
- the job page is `/ingest/imports/{id}` with the Import tab current and the file/import name last
  in the breadcrumb
- the old `/datasets/*` routes are the app's 404
"""

from __future__ import annotations

import copy

from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_app_shell import DESKTOP
from test_dataset_import import (
    IMPORT_ID,
    completed_job,
    serve_import_page,
    serve_ingest_page,
    serve_job_sequence,
)


def _serve(stub) -> None:
    serve_import_page(stub)
    serve_ingest_page(stub)
    stub.on(
        'GET', r'/datasets/imports(\?|$)', {'items': [], 'total': 0, 'page': 1, 'page_size': 50}
    )


def test_the_import_tab_renders_inside_the_ingest_layout(stub, page, app_url):
    _serve(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/ingest')
    page.get_by_test_id('ingest-pipeline-rail').wait_for(timeout=ACTION_TIMEOUT_MS)
    upload_width = page.get_by_test_id('ingest-pipeline-rail').bounding_box()['x']
    tab = page.locator('[data-tab="import"]')
    expect(tab).to_have_text('Import dataset')
    tab.click()
    expect(page).to_have_url(f'{app_url}/p/default/ingest?tab=import')
    expect(page.get_by_role('heading', level=1)).to_have_text('Ingest')
    expect(tab).to_have_attribute('aria-current', 'page')
    expect(page.get_by_test_id('import-source')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('ingest-pipeline-rail')).to_be_visible()
    assert page.get_by_test_id('ingest-pipeline-rail').bounding_box()['x'] == upload_width
    expect(page.get_by_role('heading', name='Past imports')).to_be_visible()


def test_the_job_page_is_under_ingest_with_the_tab_current(stub, page, app_url):
    _serve(stub)
    serve_job_sequence(stub, [completed_job()])
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/ingest/imports/{IMPORT_ID}')
    expect(page.get_by_test_id('job-status')).to_have_text('Done', timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-tab="import"]')).to_have_attribute('aria-current', 'page')
    expect(page.get_by_test_id('ingest-pipeline-rail')).to_be_visible()
    crumb = page.get_by_test_id('breadcrumb')
    expect(crumb).to_contain_text('Ingest')
    expect(crumb).to_contain_text('widgets_v1')
    page.get_by_role('link', name='Import dataset').first.click()
    expect(page).to_have_url(f'{app_url}/p/default/ingest?tab=import')


def test_the_old_dataset_routes_are_the_404(stub, page, app_url):
    serve_import_page(stub)
    stub.on(
        'GET',
        r'/datasets/imports(\?|$)',
        copy.deepcopy({'items': [], 'total': 0, 'page': 1, 'page_size': 50}),
    )
    for path in ('datasets', 'datasets/import', 'datasets/imports'):
        page.goto(f'{app_url}/p/default/{path}')
        expect(page.get_by_text('Page not found', exact=False)).to_be_visible(
            timeout=ACTION_TIMEOUT_MS
        )
