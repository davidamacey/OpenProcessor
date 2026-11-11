"""Undo batching: `/clusters/[uid]` bulk label (class hotkey over several
selected crops) followed by ONE Z.

The bulk label is a recorded label operation: `PUT {API_PREFIX}/crops/batch_label`
answers 202 `LabelOpAccepted`, the client follows `GET /jobs/{id}`, and Z sends
exactly one `POST /label_ops/{op_id}/undo` (queued, then followed) -- not N
per-crop undos and not the legacy `undo_batch` route.
"""

from __future__ import annotations

from fixtures.contract import cluster_card, cluster_list
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.label_ops import stub_label_op, stub_label_op_undo
from playwright.sync_api import expect


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'ducati',
        'kind': 'item',
        'group': 'moto',
        'hotkey_letter': 'k',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
]

CLUSTER_UID = 'cl-fixture-0001'
CLUSTERS = cluster_list(cluster_card(CLUSTER_UID, size=3))


def crop(i: int) -> dict:
    return make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=1,
        class_name='ducati',
        cluster={
            'partition_id': 'p1',
            'uid': CLUSTER_UID,
            'state': 'assigned',
            'score': 0.9,
            'margin': 0.1,
            'assigned_by': 'run',
        },
        label_validated=False,
        label_source='model_suggestion',
        thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
    )


def test_cluster_bulk_label_then_single_undo_queues_one_op_undo(stub, page, app_url):
    label_calls: list[tuple[str, dict]] = []
    undo_calls: list[dict] = []
    unexpected: list[str] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'page': 1, 'page_size': 60, 'crops': [crop(i) for i in range(3)]},
    )
    stub_label_op(
        stub, r'/crops/batch_label$', 'op-label-1', updated=2, method='PUT', calls=label_calls
    )
    stub_label_op_undo(stub, 'op-label-1', restored=2, calls=undo_calls)
    # Fail-closed proof: one Z over a 2-crop write must not fall back to the
    # per-crop or legacy batch undo routes.
    for pattern in (r'/crops/([^/]+)/label/undo$', r'/crops/label/undo_batch$'):
        stub.on(
            'POST', pattern, lambda request, match: unexpected.append(match.string) or (200, {})
        )

    page.goto(f'{app_url}/p/default/clusters/{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)

    # Select two crops, then bulk-label them via the class hotkey (no drag).
    page.locator('img').nth(0).click()
    page.keyboard.down('Shift')
    page.locator('img').nth(1).click()
    page.keyboard.up('Shift')
    expect(page.get_by_text('2 selected').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    with page.expect_response(
        lambda r: r.request.method == 'PUT' and r.url.endswith('/batch_label'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('k')

    # The undo entry is recorded once the job has been followed, and the
    # success toast follows it: pressing Z before that finds nothing to undo.
    expect(page.get_by_text('Labeled 2').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    assert len(label_calls) == 1, f'bulk label should PUT exactly once: {label_calls}'
    _, label_body = label_calls[0]
    assert len(label_body.get('crop_ids', [])) == 2, (
        f'expected 2 crops in the bulk label request: {label_body}'
    )

    with page.expect_response(
        lambda r: r.request.method == 'POST' and r.url.endswith('/label_ops/op-label-1/undo'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('z')
    expect(page.get_by_text('Reverted 2').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    assert undo_calls == [{'dry_run': False}], (
        f'Z should queue exactly one undo by op id: {undo_calls}'
    )
    assert unexpected == [], f'no per-crop or batch undo route expected: {unexpected}'

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected in the bulk-label/undo flow: {errors[:3]}'
