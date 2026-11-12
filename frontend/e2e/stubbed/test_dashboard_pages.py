"""The Dashboard: one primary button decided by the served next step, the adopt dialog (dry run,
then confirm, then the remaining count and the undo link) and a failed read as an ErrorState."""

from __future__ import annotations

import re

from fixtures.contract import fill, job
from harness.constants import ACTION_TIMEOUT_MS
from harness.screens import save_screens
from playwright.sync_api import expect
from test_dashboard_stats import serve_dashboard
from test_next_steps import _step, serve_next_steps


ADOPT = _step('classes', 'Create classes from the detector', '29 classes missing', False, 'classes')
ADOPT['action'] = 'adopt_detector_classes'
STEPS = [
    _step('ingest', 'Ingest images', '120 images ingested', True, 'ingest'),
    ADOPT,
    _step('cluster', 'Cluster', 'No partition yet', False, 'workspace', 'view=clusters'),
]
DRY = fill(
    'AdoptDetectorResponse',
    dry_run=True,
    items=40,
    classes_to_create=['bus', 'car'],
    blocked=[],
    blocked_items=0,
    remaining=0,
)
REAL = {**DRY, 'dry_run': False, 'remaining': 5, 'op_id': 'op-9', 'job_id': 'job-9'}
SUMMARY = fill(
    'DetectionsSummary',
    total=120,
    embedding=fill('EmbeddingBreakdown', embedded=100, not_embedded=20),
)


def _serve(stub, steps=STEPS):
    serve_next_steps(stub, steps)
    stub.on('GET', r'/detections/summary(\?|$)', SUMMARY)
    stub.on('GET', r'/vlm/endpoints/active(\?|$)', (404, {'detail': 'Not Found'}))


def test_one_primary_button_and_it_is_the_served_actions(stub, page, app_url):
    _serve(stub)
    page.goto(f'{app_url}/p/default/dashboard')
    expect(page.get_by_test_id('next-step-action')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('next-step-action')).to_have_class(re.compile('btn-primary'))
    assert page.locator('.btn-primary').count() == 1
    save_screens(page, 'workflow_ux-e', 'dashboard', desktop=1600)


def test_adopt_dialog_dry_run_then_confirm(stub, page, app_url):
    _serve(stub)
    calls = []

    def adopt(request, _match):
        body = request.post_data_json
        calls.append(body['dry_run'])
        return (200, DRY if body['dry_run'] else REAL)

    stub.on('POST', r'/label_ops/adopt_detector(\?|$)', adopt)
    stub.on(
        'GET',
        r'/jobs/job-9(\?|$)',
        job('job-9', result={'op_id': 'op-9', 'updated': 40, 'conflicts': 0, 'complete': True}),
    )
    page.goto(f'{app_url}/p/default/dashboard')
    page.get_by_test_id('next-step-action').click(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('adopt-items')).to_have_text('40', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('adopt-create')).to_contain_text('bus, car')
    assert calls == [True]
    save_screens(page, 'workflow_ux-e', 'dashboard-adopt-dry-run', desktop=1600)
    page.get_by_role('button', name='Adopt 40 items').click()
    expect(page.get_by_test_id('adopt-result')).to_contain_text(
        '5 more are left', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('adopt-undo-link')).to_have_attribute(
        'href', '/p/default/quality/history'
    )
    assert calls == [True, False]


def test_failed_detections_read_is_an_error_state_with_retry(stub, page, app_url):
    serve_dashboard(stub)
    stub.on('GET', r'/detections/summary(\?|$)', (404, {'detail': 'Not Found'}))
    page.goto(f'{app_url}/p/default/dashboard')
    err = page.get_by_test_id('detections-error').get_by_test_id('error-state')
    expect(err).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(err).to_contain_text('Not Found')
    stub.on('GET', r'/detections/summary(\?|$)', SUMMARY)
    err.get_by_role('button', name='Retry').click()
    expect(page.get_by_test_id('detections-total')).to_contain_text('120 detections')
