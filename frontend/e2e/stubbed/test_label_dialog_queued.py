"""A label whose job waits in the queue shows its served state ("Queued, 2 ahead"), never a bare
"Working…", and can run in the background; the end shows as a toast with Undo."""

from __future__ import annotations

from fixtures.contract import job, label_op_accepted
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_cluster_scope_toolbar import CLASSES, UIDS, _open, _serve


def test_a_queued_label_shows_the_jobs_ahead_and_runs_in_the_background(stub, page, app_url):
    _serve(stub, [])
    state = {'status': 'queued'}

    def label(request, _match):
        if (request.post_data_json or {}).get('dry_run'):
            return (200, {'dry_run': True, 'selected': 3})
        return (202, label_op_accepted('op-q', 'job-q'))

    def read_job(_request, _match):
        if state['status'] == 'queued':
            return job('job-q', status='queued', queue='interactive', queue_ahead=2)
        return job(
            'job-q',
            result={'op_id': 'op-q', 'updated': 3, 'conflicts': 0, 'complete': True},
        )

    stub.on('PUT', r'/crops/batch_label$', label)
    stub.on('GET', r'/jobs/job-q$', read_job)
    _open(page, app_url, UIDS[0])
    page.get_by_test_id('label-cluster').click(timeout=ACTION_TIMEOUT_MS)
    page.get_by_role('combobox', name='Search classes').fill(CLASSES[0]['class_name'])
    page.keyboard.press('Enter')
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    # Keep this dialog open: leave the cluster list's auto-advance off for this flow.
    page.get_by_test_id('open-next-after-label').uncheck()
    page.get_by_role('dialog').get_by_role('button', name='Apply').click()
    job_line = page.get_by_test_id('selection-job')
    expect(job_line).to_contain_text('Queued, 2 ahead', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_role('dialog').get_by_role('button', name='Working…')).to_have_count(0)
    page.get_by_test_id('selection-background').click()
    expect(page.get_by_role('dialog')).to_have_count(0)
    state['status'] = 'completed'
    expect(page.get_by_test_id('toast-action')).to_have_text('Undo', timeout=ACTION_TIMEOUT_MS)
