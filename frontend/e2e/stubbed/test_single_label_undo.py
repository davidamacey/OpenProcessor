"""A single label write is a served label operation, so its undo survives a reload.

On `/review`, Enter labels an item: `PUT /crops/{id}/label` answers the `op_id` of the
operation it recorded. After a reload the page's own undo stack is empty (Z finds nothing),
but the write is in the served Label history (Quality page, History tab) and is undone there
by `POST /label_ops/{op_id}/undo`.
"""

from __future__ import annotations

from fixtures.contract import fill
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.label_ops import stub_crop_label, stub_label_op_undo
from harness.waits import wait_for_paint
from playwright.sync_api import expect
from test_label_ops import list_page
from test_review_actions import CLASSES, METHODS, review_item


AT = '2026-11-12T10:00:00+00:00'


def served_op(op_id: str) -> dict:
    return fill(
        'LabelOp',
        op_id=op_id,
        kind='label',
        at=AT,
        finished_at=AT,
        class_name='ducati',
        label_source='human',
        target='crop_ids',
        selected=1,
        count=1,
        conflicts=0,
        current=1,
        status='written',
        undone=False,
        undoable=True,
        last_undo=None,
    )


def test_single_label_undo_survives_a_reload(stub, page, app_url):
    rows: list[dict] = []
    label_calls: list[tuple[str, dict]] = []
    undo_calls: list[dict] = []
    legacy: list[str] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', METHODS)
    stub.on(
        'GET',
        r'/review/(?!tabs)',
        (
            200,
            {
                'items': [review_item(i) for i in range(3)],
                'total': 3,
                'next_cursor': None,
                'limit': 30,
            },
        ),
    )
    stub.on('GET', r'/label_ops(\?|$)', lambda request, _m: list_page(rows, request))
    stub_label_op_undo(stub, 'op-crop-0', calls=undo_calls)
    stub.on(
        'POST',
        r'/crops/[^/]+/label/undo(_batch)?$',
        lambda request, match: legacy.append(match.string) or (200, {}),
    )
    stub_crop_label(stub, calls=label_calls)

    page.goto(review_url(app_url, 'to_label'))
    page.get_by_test_id('queue-counter').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    with page.expect_response(
        lambda r: r.request.method == 'PUT' and r.url.endswith('/label'), timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press('Enter')
    assert [c for c, _ in label_calls] == ['crop-0']
    rows.append(served_op('op-crop-0'))  # the server recorded it

    page.reload()
    page.get_by_test_id('queue-counter').first.wait_for(timeout=ACTION_TIMEOUT_MS)
    wait_for_paint(page)
    page.keyboard.press('z')
    expect(page.get_by_text('Nothing to undo.').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert undo_calls == [], 'the reloaded page has no client-side undo stack'

    page.goto(f'{app_url}/p/default/quality/history')
    expect(page.get_by_test_id('label-op-row')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)
    page.get_by_role(
        'button', name='Details for Labelled 1 item as ducati at 2026-11-12 10:00:00 UTC'
    ).click()
    page.get_by_test_id('label-op-undo').click()
    expect(page.get_by_test_id('label-op-confirm')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('label-op-confirm').click()
    expect(page.get_by_test_id('label-op-result')).to_contain_text(
        'Undo finished', timeout=ACTION_TIMEOUT_MS
    )
    assert undo_calls == [{'dry_run': True}, {'dry_run': False}], undo_calls
    assert legacy == [], f'the undo goes through the served operation: {legacy}'
