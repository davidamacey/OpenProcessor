"""`/projects` pages by the served cursor, searches on the server and sorts by the served options.
Set PROJECTS_SHOTS=<dir> to save screenshots at 1600 and 430 px."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qs, urlparse

from fixtures.wire import project, projects_response
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


API_PREFIX = '/curation'
ALL = [project(API_PREFIX, 'default', is_default=True, deletable=False)] + [
    project(API_PREFIX, f'survey-{i:02d}', display_name=f'Survey {i:02d}') for i in range(60)
]
SEEN: list[dict[str, list[str]]] = []


def _served(request, _match):
    query = parse_qs(urlparse(request.url).query)
    SEEN.append(query)
    q = query.get('q', [''])[0].lower()
    limit = int(query['limit'][0]) if 'limit' in query else None
    hit = [p for p in ALL if q in p['slug'] or q in p['display_name'].lower()]
    if query.get('sort') == ['updated_at:desc']:
        hit = hit[::-1]
    start = int(query['cursor'][0]) if 'cursor' in query else 0
    end = start + limit if limit else len(hit)
    body = projects_response(API_PREFIX, hit[start:end])
    body['total'] = len(hit)
    body['next_cursor'] = str(end) if end < len(hit) else None
    body['sort'] = query.get('sort', ['name:asc'])[0]
    return (200, body)


def test_projects_page_pages_searches_and_sorts(stub, page, app_url):
    stub.on('GET', rf'^{re.escape(API_PREFIX)}/projects$', _served)
    shots = os.environ.get('PROJECTS_SHOTS')
    if shots:
        page.set_viewport_size({'width': 1600, 'height': 900})
    page.goto(f'{app_url}/projects')
    page.get_by_test_id('project-row-default').wait_for(timeout=ACTION_TIMEOUT_MS)
    pager = page.get_by_test_id('projects-pager')
    assert 'Showing 50 of 61' in pager.inner_text()
    width = page.viewport_size['width']
    if shots:
        page.screenshot(path=f'{shots}/projects_page_{width}.png', full_page=True)

    page.get_by_test_id('projects-more').click()
    page.get_by_test_id('project-row-survey-59').wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'Showing 61 of 61' in pager.inner_text()
    assert page.get_by_test_id('projects-more').count() == 0

    page.get_by_test_id('projects-search').fill('survey-4')
    expect(pager).to_contain_text('Showing 10 of 10', timeout=ACTION_TIMEOUT_MS)
    assert SEEN[-1]['q'] == ['survey-4']

    page.get_by_test_id('projects-sort').select_option('updated_at:desc')
    page.wait_for_function(
        "() => document.querySelector('[data-testid=projects-table] tbody tr')"
        "?.getAttribute('data-testid') === 'project-row-survey-49'",
        timeout=ACTION_TIMEOUT_MS,
    )
    assert SEEN[-1]['sort'] == ['updated_at:desc']

    page.get_by_test_id('projects-status-filter').select_option('archived')
    expect(page.get_by_test_id('projects-status-filter')).to_have_value('archived')
    assert SEEN[-1]['status'] == ['archived']
    if shots:
        page.screenshot(path=f'{shots}/projects_page_filtered_{width}.png', full_page=True)
        page.set_viewport_size({'width': 430, 'height': 900})
        page.screenshot(path=f'{shots}/projects_page_filtered_430.png', full_page=True)
