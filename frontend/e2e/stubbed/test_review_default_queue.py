"""Review always runs on a served queue. With none in the URL it opens the served `default_tab`;
an item a person or an import already validated is never asked to be confirmed; and "Accept
confident labels…" shows the served dry run by class, applies one recorded operation and offers
Undo."""

from __future__ import annotations

from fixtures.contract import job
from fixtures.wire import make_item, review_tab, review_tabs
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.label_ops import stub_label_op_undo
from playwright.sync_api import expect
from test_review_view import CLASSES


def item(i: int, **over) -> dict:
    fields = {
        'crop_id': f'crop-{i}',
        'image_id': f'img-{i}',
        'class_id': 1,
        'class_name': 'ducati',
        'proposed_class_id': 1,
        'proposed_class_name': 'ducati',
        'label_validated': False,
        'class_validated': False,
        'label_source': 'model_suggestion',
        'thumbnail_url': f'/curation/crops/crop-{i}/thumbnail',
    }
    return make_item(**{**fields, **over})


def queue_body(items: list[dict]) -> dict:
    return {
        'items': [dict(i, reason='') for i in items],
        'total': len(items),
        'next_cursor': None,
        'limit': 30,
        'sort_applied': 'detector_confidence:asc',
        'sort_fallback_reason': None,
        'empty_reason': None,
        'hidden': {'test_holdout': 0, 'excluded': 0},
        'join_truncated': False,
    }


def serve(stub, items: list[dict], queues: list[str]) -> list[str]:
    seen: list[str] = []
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on(
        'GET',
        r'/review/tabs(\?|$)',
        review_tabs(
            review_tab('to_label', 'To label', count=0),
            review_tab('check_machine', 'Check machine labels', count=len(items)),
        ),
    )

    def review(request, match):
        seen.append(request.url)
        return (200, queue_body(items))

    for q in queues:
        stub.on('GET', rf'/review/{q}(\?|$)', review)
    stub.on('GET', r'/crops(\?|$)', {'total': 99, 'next_cursor': None, 'limit': 48, 'crops': []})
    stub.on('GET', r'/crops/[^/]+/history$', {'crop_id': 'crop-0', 'entries': []})
    stub.on('GET', r'/crops/(crop-\d+)$', lambda r, m: (200, item(int(m.group(1)[5:]))))
    return seen


def test_no_queue_opens_the_served_default_tab(stub, page, app_url):
    seen = serve(stub, [item(0), item(1)], ['check_machine'])
    page.goto(review_url(app_url))
    page.wait_for_url('**queue=check_machine*', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('queue-counter').first).to_contain_text(
        '#1 of 2', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('review-queue-label')).to_have_text('Check machine labels')
    # The Detections order was never read: only the served queue.
    assert seen and all('/review/check_machine' in u for u in seen), seen


def test_a_validated_item_has_no_confirm(stub, page, app_url):
    validated = item(0, class_validated=True, label_validated=True, label_source='human')
    serve(stub, [validated, item(1)], ['check_machine'])
    page.goto(review_url(app_url, 'check_machine'))
    expect(page.get_by_test_id('review-validated')).to_contain_text(
        'Already validated', timeout=ACTION_TIMEOUT_MS
    )
    actions = page.get_by_test_id('review-actions')
    expect(actions.get_by_role('button', name='Confirm')).to_have_count(0)
    expect(actions.get_by_test_id('review-change-class')).to_be_visible()
    primary = page.locator('[data-testid=workspace-view] .btn-primary:visible')
    expect(primary).to_have_count(1)
    expect(primary).to_contain_text('Next')
    # The next, unvalidated item still asks to be confirmed.
    page.get_by_test_id('review-next').click()
    expect(
        page.get_by_test_id('review-actions').get_by_role('button', name='Confirm')
    ).to_be_visible(timeout=ACTION_TIMEOUT_MS)


def test_accept_confident_labels_shows_the_dry_run_applies_one_op_and_offers_undo(
    stub, page, app_url
):
    serve(stub, [item(0), item(1)], ['check_machine'])
    bodies: list[dict] = []
    undo_calls: list[dict] = []

    def accept(request, _match):
        body = request.post_data_json or {}
        bodies.append(body)
        result = {
            'dry_run': body['dry_run'],
            'items': 412,
            'by_class': [
                {'class_name': 'car', 'count': 300},
                {'class_name': 'person', 'count': 112},
            ],
            'held_back': {
                'holdout': 30,
                'audit_sample': 4,
                'excluded': 0,
                'locked': 185,
                'deprecated_class': 0,
            },
            'remaining': 0,
        }
        if body['dry_run']:
            return (200, {**result, 'op_id': None, 'job_id': None})
        return (200, {**result, 'op_id': 'op-acc', 'job_id': 'job-acc'})

    stub.on('POST', r'/label_ops/accept_machine$', accept)
    stub.on(
        'GET',
        r'/jobs/job-acc$',
        job(
            'job-acc', result={'op_id': 'op-acc', 'updated': 412, 'conflicts': 0, 'complete': True}
        ),
    )
    stub_label_op_undo(stub, 'op-acc', restored=412, calls=undo_calls)
    stub.on(
        'GET',
        r'/audit/report(\?|$)',
        {
            'audited': 40,
            'pending': 0,
            'min_per_class': 10,
            'detector': [
                {
                    'name': 'car',
                    'n': 30,
                    'correct': 27,
                    'precision': 0.9,
                    'ci_low': 0.74,
                    'ci_high': 0.97,
                    'insufficient_sample': False,
                }
            ],
            'vlm': [],
            'confusion': {},
            'outcomes': {},
        },
    )
    page.goto(review_url(app_url, 'check_machine'))
    page.get_by_test_id('accept-confident-open').click(timeout=ACTION_TIMEOUT_MS)
    total = page.get_by_test_id('accept-total')
    expect(total).to_contain_text('412', timeout=ACTION_TIMEOUT_MS)
    table = page.get_by_test_id('accept-by-class')
    expect(table).to_contain_text('car')
    expect(table).to_contain_text('300')
    expect(table).to_contain_text('90%')  # the audited precision of the detector's class
    expect(table).to_contain_text('not audited')  # person has no audit sample
    expect(page.get_by_test_id('accept-held-back')).to_contain_text('185')
    assert bodies[0]['dry_run'] is True
    assert bodies[0]['min_confidence'] == 0.95
    page.get_by_role('button', name='Accept 412 labels').click()
    toast = page.get_by_test_id('toast-action')
    expect(toast).to_have_text('Undo', timeout=ACTION_TIMEOUT_MS)
    assert [b['dry_run'] for b in bodies] == [True, False]
    toast.click()
    for _ in range(100):
        if undo_calls:
            break
        page.wait_for_timeout(100)
    assert undo_calls
