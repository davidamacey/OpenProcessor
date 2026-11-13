"""`/p/{project}/quality`: Accuracy audit, Near-duplicates and Label history as
tabs of one page. Also saves full-page screenshots under artifacts_local/screens/workflow_3e/.
"""

from __future__ import annotations

import re

import pytest
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_label_ops import serve as serve_label_ops
from test_near_duplicates import _serve as serve_near_dups


@pytest.fixture(autouse=True)
def _classes(stub):
    serve_label_ops(stub)
    serve_near_dups(stub)


TABS = [
    ('audit', 'Accuracy audit'),
    ('near-duplicates', 'Near-duplicates'),
    ('history', 'Label history'),
]


def test_bare_quality_lands_on_the_audit_tab(stub, page, app_url):
    page.goto(f'{app_url}/p/default/quality')
    expect(page).to_have_url(re.compile(r'/p/default/quality/audit$'), timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-tab="audit"]')).to_have_attribute('aria-current', 'page')


def test_each_tab_opens_its_own_route(stub, page, app_url):
    page.goto(f'{app_url}/p/default/quality/audit')
    for slug, label in TABS:
        tab = page.locator(f'[data-tab="{slug}"]')
        expect(tab).to_have_text(label, timeout=ACTION_TIMEOUT_MS)
        tab.click()
        expect(page).to_have_url(re.compile(rf'/p/default/quality/{slug}$'))
        expect(tab).to_have_attribute('aria-current', 'page')
        for other, _ in TABS:
            if other != slug:
                expect(page.locator(f'[data-tab="{other}"]')).not_to_have_attribute(
                    'aria-current', 'page'
                )


def test_nav_has_one_quality_entry(stub, page, app_url):
    page.goto(f'{app_url}/p/default/quality/history')
    nav = page.get_by_role('navigation').first
    expect(nav.get_by_role('link', name='Quality')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    for gone in ('Audit', 'Label ops'):
        expect(nav.get_by_role('link', name=gone, exact=True)).to_have_count(0)


def test_screens(stub, page, app_url):
    for slug, _ in TABS:
        page.goto(f'{app_url}/p/default/quality/{slug}')
        expect(page.get_by_role('navigation', name='Data quality')).to_be_visible(
            timeout=ACTION_TIMEOUT_MS
        )
        save_screens(page, 'workflow_3e', f'quality-{slug}')
