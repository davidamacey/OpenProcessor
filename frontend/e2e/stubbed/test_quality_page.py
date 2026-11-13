"""`/p/{project}/quality`: Ready to train, Accuracy audit and Label history as tabs of one
page. Also saves full-page screenshots under artifacts_local/screens/uxr-d/.
"""

from __future__ import annotations

import re

import pytest
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_audit_gallery import _serve as serve_audit
from test_label_ops import serve as serve_label_ops
from test_quality_ready import _serve as serve_ready


@pytest.fixture(autouse=True)
def _classes(stub):
    serve_label_ops(stub)
    serve_ready(stub)
    serve_audit(stub)


TABS = [
    ('ready', 'Ready to train'),
    ('audit', 'Accuracy audit'),
    ('history', 'Label history'),
]


def test_bare_quality_lands_on_ready_to_train(stub, page, app_url):
    page.goto(f'{app_url}/p/default/quality')
    expect(page).to_have_url(re.compile(r'/p/default/quality/ready$'), timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-tab="ready"]')).to_have_attribute('aria-current', 'page')


def test_each_tab_opens_its_own_route(stub, page, app_url):
    page.goto(f'{app_url}/p/default/quality/ready')
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
        expect(page.get_by_test_id('tabs')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
        save_screens(page, 'uxr-d', f'quality-{slug}')
