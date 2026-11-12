"""The top-bar project picker: a bounded, scrollable list the server filters as the operator
types (`GET /projects?q=&limit=`), with listbox keyboard navigation. Set PICKER_SHOTS=<dir>
to save screenshots at 1280 and 430 px."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qs, urlparse

from fixtures.wire import project, projects_response
from harness.constants import ACTION_TIMEOUT_MS


API_PREFIX = '/curation'
ALL = [project(API_PREFIX, 'default', is_default=True, deletable=False)] + [
    project(API_PREFIX, f'survey-{i:02d}', display_name=f'Survey {i:02d}') for i in range(60)
]


def _page_of_projects(request, _match):
    query = parse_qs(urlparse(request.url).query)
    q = query.get('q', [''])[0].lower()
    limit = int(query['limit'][0]) if 'limit' in query else None
    hit = [p for p in ALL if q in p['slug'] or q in p['display_name'].lower()]
    body = projects_response(API_PREFIX, hit[:limit] if limit else hit)
    body['total'] = len(hit)
    return (200, body)


def test_picker_is_bounded_scrollable_and_searchable(stub, page, app_url):
    stub.on('GET', rf'^{re.escape(API_PREFIX)}/projects$', _page_of_projects)
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/review/(?!tabs)',
        (200, {'items': [], 'total': 0, 'next_cursor': None, 'limit': 30}),
    )
    page.goto(f'{app_url}/p/default/review')
    page.get_by_test_id('project-switcher-trigger').click()
    page.get_by_test_id('project-switcher-menu').wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('project-option-survey-00').wait_for(timeout=ACTION_TIMEOUT_MS)

    listbox = page.get_by_test_id('project-switcher-list')
    assert listbox.get_by_role('option').count() == 25
    assert 'Showing 25 of 61' in page.get_by_test_id('project-switcher-more').inner_text()
    # The list scrolls inside its own box and never runs past the viewport.
    box = listbox.bounding_box()
    viewport = page.viewport_size
    assert box['y'] + box['height'] <= viewport['height']
    assert listbox.evaluate('el => el.scrollHeight > el.clientHeight')
    shots = os.environ.get('PICKER_SHOTS')
    if shots:
        page.screenshot(path=f'{shots}/picker_open_{viewport["width"]}.png')

    page.get_by_test_id('project-switcher-filter').fill('survey-4')
    page.get_by_test_id('project-option-survey-40').wait_for(timeout=ACTION_TIMEOUT_MS)
    assert listbox.get_by_role('option').count() == 10
    if shots:
        page.screenshot(path=f'{shots}/picker_filtered_{viewport["width"]}.png')

    if shots:
        page.set_viewport_size({'width': 430, 'height': 900})
        listbox.wait_for()
        box = listbox.bounding_box()
        assert box['x'] >= 0 and box['x'] + box['width'] <= 430
        page.screenshot(path=f'{shots}/picker_filtered_430.png')

    page.keyboard.press('ArrowDown')
    page.keyboard.press('Enter')
    page.wait_for_url(re.compile(r'/p/survey-41/review'), timeout=ACTION_TIMEOUT_MS)
