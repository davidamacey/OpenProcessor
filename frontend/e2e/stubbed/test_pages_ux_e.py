"""Screenshots (1600 and 430 px) of the Ingest tabs, Data quality and Settings, from the stubbed
harness, with the structure each page promises: URL-bound tabs, the Pipeline rail, the sidebar."""

from __future__ import annotations

from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_ingest import _base_ingest_stubs


FOLDER = '/data/archive'


def test_ingest_tabs_and_pipeline_rail(stub, page, app_url):
    _base_ingest_stubs(stub, status_total=120, batch_source_roots=[FOLDER])
    page.goto(f'{app_url}/p/default/ingest')
    expect(page.get_by_test_id('ingest-pipeline-rail')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    tabs = page.get_by_test_id('tabs')
    for name in ('Upload', 'Server folder', 'Sources', 'Activity'):
        expect(tabs.get_by_role('link', name=name)).to_be_visible()
    expect(tabs.get_by_role('link', name='Upload')).to_have_attribute('aria-current', 'page')
    save_screens(page, 'workflow_ux-e', 'ingest-upload', desktop=1600)
    tabs.get_by_role('link', name='Activity').click()
    expect(page).to_have_url(f'{app_url}/p/default/ingest?tab=activity')
    expect(page.get_by_test_id('ingest-pipeline-rail')).to_be_visible()
    save_screens(page, 'workflow_ux-e', 'ingest-activity', desktop=1600)


def test_data_quality_tabs(stub, page, app_url):
    stub.on(
        'GET',
        r'/audit/report(\?|$)',
        {
            'audited': 0,
            'pending': 0,
            'min_per_class': 30,
            'detector': [],
            'vlm': [],
            'confusion': {},
            'outcomes': {},
        },
    )
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    page.goto(f'{app_url}/p/default/quality/audit')
    expect(page.get_by_role('heading', name='Data quality', level=1)).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('audit-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    save_screens(page, 'workflow_ux-e', 'data-quality-audit-empty', desktop=1600)


def test_settings_sidebar_and_search(stub, page, app_url):
    stub.on('GET', r'/settings(\?|$)', {'defaults': {}, 'updated_at': None, 'updated_by': None})
    stub.on('GET', r'/methods(\?|$)', {'strategies': [], 'flags': {}})
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/scores/coverage(\?|$)', {'coverage': {}})
    stub.on('GET', r'/scores/status(\?|$)', {'status': 'idle'})
    page.goto(f'{app_url}/p/default/settings')
    nav = page.get_by_test_id('sidebar-nav')
    expect(nav.get_by_role('link', name='General')).to_have_attribute(
        'aria-current', 'page', timeout=ACTION_TIMEOUT_MS
    )
    page.keyboard.press('/')
    expect(page.get_by_test_id('settings-search')).to_be_focused()
    save_screens(page, 'workflow_ux-e', 'settings-general', desktop=1600)
    page.get_by_test_id('settings-search').fill('serving')
    expect(nav.get_by_role('link')).to_have_count(1)
    stub.on('GET', r'/serving_profiles(\?|$)', {'items': []})
    stub.on(
        'GET',
        r'/serving_profiles/schema$',
        {'kind': 'serving_profile', 'axes': [], 'json_schema': {}},
    )
    stub.on('GET', r'/serving_profiles/active$', {'active': None})
    nav.get_by_role('link', name='Serving profiles').click()
    expect(page).to_have_url(f'{app_url}/p/default/settings/presets/serving-profiles')
