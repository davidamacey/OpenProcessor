"""The served next steps: the Dashboard card and the empty Workspace show the
loop in the served order, the first undone step is the current one, and each
link lands on its section. Also saves screenshots under
artifacts_local/screens/workflow_3e/.
"""

from __future__ import annotations

import re

from fixtures.contract import cluster_list, fill
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_dashboard_stats import serve_dashboard


def _step(id_, label, detail, done, section, query=None):
    return {
        'id': id_,
        'label': label,
        'detail': detail,
        'done': done,
        'section': section,
        'query': query,
    }


FRESH = [
    _step('ingest', 'Ingest images', '120 images ingested', True, 'ingest'),
    _step('classes', 'Create classes from the detector', 'No classes yet', False, 'classes'),
    _step('cluster', 'Cluster', 'No partition yet', False, 'workspace', 'view=clusters'),
]


def serve_next_steps(stub, steps):
    serve_dashboard(stub)
    stub.on('GET', r'/review/new_class_proposals/summary(\?|$)', (404, {'detail': 'Not Found'}))
    nxt = next((s for s in steps if not s['done']), None)
    stub.on(
        'GET',
        r'/next_steps(\?|$)',
        fill('WorkflowSteps', steps=steps, next=nxt, workspace_default_view='gallery'),
    )


def test_dashboard_lists_the_served_steps_and_marks_the_next(stub, page, app_url):
    serve_next_steps(stub, FRESH)
    page.goto(f'{app_url}/p/default/dashboard')
    rows = page.get_by_test_id('next-step')
    expect(rows).to_have_count(3, timeout=ACTION_TIMEOUT_MS)
    expect(rows.nth(0)).to_have_attribute('data-done', 'true')
    expect(rows.nth(1)).to_contain_text('No classes yet')
    expect(page.get_by_test_id('next-step-current')).to_have_text(
        'Create classes from the detector'
    )
    save_screens(page, 'workflow_3e', 'dashboard-next-steps')


def test_each_link_lands_on_its_section(stub, page, app_url):
    serve_next_steps(stub, FRESH)
    page.goto(f'{app_url}/p/default/dashboard')
    page.get_by_test_id('next-step-current').click(timeout=ACTION_TIMEOUT_MS)
    expect(page).to_have_url(re.compile(r'/p/default/classes$'))
    serve_next_steps(stub, FRESH)
    stub.on('GET', r'/clusters(\?|$)', cluster_list())
    page.goto(f'{app_url}/p/default/dashboard')
    page.get_by_role('link', name='Cluster', exact=True).click(timeout=ACTION_TIMEOUT_MS)
    expect(page).to_have_url(re.compile(r'/p/default/workspace\?view=clusters'))


def test_empty_workspace_shows_the_same_steps(stub, page, app_url):
    serve_next_steps(stub, FRESH)
    stub.on('GET', r'/crops(\?|$)', {'total': 0, 'next_cursor': None, 'limit': 30, 'crops': []})
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    expect(page.get_by_test_id('view-state-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('next-step')).to_have_count(3)
    save_screens(page, 'workflow_3e', 'workspace-empty-next-steps')


def test_empty_clusters_view_shows_the_same_steps(stub, page, app_url):
    serve_next_steps(stub, FRESH)
    stub.on('GET', r'/clusters(\?|$)', cluster_list())
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.get_by_test_id('clusters-empty')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('next-step')).to_have_count(3)
    save_screens(page, 'workflow_3e', 'workspace-empty-clusters-next-steps')
