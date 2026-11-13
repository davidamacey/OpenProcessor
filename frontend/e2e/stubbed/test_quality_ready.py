"""Data quality › Ready to train: the served readiness checklist.

- rows render the served label, severity (icon and word) and message, and each link opens the
  served Workspace (or page) address
- `/quality` lands on Ready to train
- the near-duplicates route is the app's 404
"""

from __future__ import annotations

import re
from typing import Any

from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_app_shell import DESKTOP


READINESS: dict[str, Any] = {
    'verdict': 'blocked',
    'computed_at': '2026-11-12T18:20:00Z',
    'checks': [
        {
            'id': 'test_holdout',
            'label': 'Test set',
            'severity': 'block',
            'message': '2 classes have under 5 test examples (bottle 0, book 0).',
            'count': 2,
            'link': {'section': 'export', 'query': None},
            'classes': [
                {'class_name': 'bottle', 'count': 0},
                {'class_name': 'book', 'count': 0},
            ],
        },
        {
            'id': 'class_balance',
            'label': 'Class balance',
            'severity': 'warn',
            'message': '4 classes have under 500 checked examples.',
            'count': 4,
            'link': {'section': 'workspace', 'query': 'view=clusters'},
            'classes': [{'class_name': 'car', 'count': 74}],
        },
        {
            'id': 'machine_unchecked',
            'label': 'Detector labels not checked',
            'severity': 'warn',
            'message': '1,131 items carry only a detector class.',
            'count': 1131,
            'link': {'section': 'workspace', 'query': 'view=review&queue=check_machine'},
            'classes': [],
        },
        {
            'id': 'audit',
            'label': 'Accuracy audit',
            'severity': 'warn',
            'message': 'No accuracy audit has been run.',
            'count': None,
            'link': {'section': 'quality/audit', 'query': None},
            'classes': [],
        },
        {
            'id': 'near_dup',
            'label': 'Near-duplicate groups',
            'severity': 'ok',
            'message': 'Near-duplicate groups are up to date.',
            'count': None,
            'link': None,
            'classes': [],
        },
    ],
    'coverage': [
        {'class_name': 'car', 'validated': 74, 'machine_only': 10, 'detector_only': 412, 'test': 3},
        {
            'class_name': 'person',
            'validated': 300,
            'machine_only': 0,
            'detector_only': 380,
            'test': 9,
        },
    ],
}


def _serve(stub: Any, body: dict[str, Any] | None = None) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on('GET', r'/quality/readiness(\?|$)', body or READINESS)


def test_rows_render_served_labels_severities_and_links(stub, page, app_url):
    _serve(stub)
    page.set_viewport_size(DESKTOP)
    page.goto(f'{app_url}/p/default/quality/ready')
    rows = page.get_by_test_id('readiness-row')
    expect(rows).to_have_count(5, timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('readiness-verdict')).to_contain_text('Not ready')
    first = rows.nth(0)
    expect(first).to_contain_text('Test set')
    expect(first.get_by_test_id('readiness-severity')).to_have_text('Blocking')
    expect(first.get_by_role('link')).to_have_attribute('href', '/p/default/export')
    expect(rows.nth(1).get_by_test_id('readiness-severity')).to_have_text('Warning')
    expect(rows.nth(1).get_by_role('link')).to_have_attribute(
        'href', '/p/default/workspace?view=clusters'
    )
    expect(rows.nth(2).get_by_role('link')).to_have_attribute(
        'href', '/p/default/workspace?view=review&queue=check_machine'
    )
    expect(rows.nth(3).get_by_role('link')).to_have_attribute('href', '/p/default/quality/audit')
    expect(rows.nth(4).get_by_test_id('readiness-severity')).to_have_text('OK')
    expect(rows.nth(4).get_by_role('link')).to_have_count(0)
    expect(page.get_by_test_id('readiness-coverage').get_by_role('row')).to_have_count(3)
    save_screens(page, 'uxr-d', 'quality-ready')


def test_bare_quality_lands_on_ready_to_train(stub, page, app_url):
    _serve(stub)
    page.goto(f'{app_url}/p/default/quality')
    expect(page).to_have_url(re.compile(r'/p/default/quality/ready$'), timeout=ACTION_TIMEOUT_MS)
    expect(page.locator('[data-tab="ready"]')).to_have_attribute('aria-current', 'page')


def test_the_near_duplicates_route_is_the_404(stub, page, app_url):
    _serve(stub)
    page.goto(f'{app_url}/p/default/quality/near-duplicates')
    expect(page.get_by_role('heading', name='Page not found')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
