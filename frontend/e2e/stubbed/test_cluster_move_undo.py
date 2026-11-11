"""M5 (interactive pass): `/clusters/[uid]` M (move selected to another
cluster) followed by Z must undo the move.

The move is a recorded label operation: `POST {API_PREFIX}/crops/move` answers
202 `LabelOpAccepted`, the client follows `GET /jobs/{id}`, and Z undoes it by
operation id (`POST /label_ops/{op_id}/undo`, also queued and followed). It
never falls back to the per-crop label undo routes.
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

SOURCE_UID = 'cl-source-0001'
TARGET_UID = 'cl-target-0002'
CLUSTERS = cluster_list(cluster_card(SOURCE_UID, size=2), cluster_card(TARGET_UID, size=5))


def crop(i: int) -> dict:
    return make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=1,
        class_name='ducati',
        cluster={
            'partition_id': 'p1',
            'uid': SOURCE_UID,
            'state': 'assigned',
            'score': 0.9,
            'margin': 0.1,
            'assigned_by': 'run',
        },
        label_validated=False,
        label_source='model_suggestion',
        thumbnail_url=f'/curation/crops/crop-{i}/thumbnail',
    )


def test_move_then_undo_follows_the_job_and_undoes_by_op_id(stub, page, app_url):
    move_calls: list[tuple[str, dict]] = []
    undo_calls: list[dict] = []
    unexpected: list[str] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 2, 'page': 1, 'page_size': 60, 'crops': [crop(i) for i in range(2)]},
    )
    stub_label_op(stub, r'/crops/move$', 'op-move-1', calls=move_calls)
    stub_label_op_undo(stub, 'op-move-1', calls=undo_calls)
    # Fail-closed proof: a move's undo must not use the per-crop label routes.
    for pattern in (r'/crops/([^/]+)/label/undo$', r'/crops/label/undo_batch$'):
        stub.on(
            'POST', pattern, lambda request, match: unexpected.append(match.string) or (200, {})
        )

    page.goto(f'{app_url}/p/default/clusters/{SOURCE_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)

    page.locator('img').first.click()
    expect(page.get_by_text('1 selected').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    # M opens the move picker, a select of the served clusters.
    page.keyboard.press('m')
    target = page.get_by_test_id('move-target')
    target.wait_for(timeout=ACTION_TIMEOUT_MS)
    assert 'Assign class to selected' in page.get_by_test_id('move-relabel-hint').inner_text()
    assert page.get_by_test_id('assign-selected').inner_text().strip() == 'Assign class to selected'
    target.select_option(TARGET_UID)
    with page.expect_response(
        lambda r: r.request.method == 'POST' and r.url.endswith('/crops/move'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('Enter')

    # The undo entry is recorded once the job has been followed, and the
    # success toast follows it: pressing Z before that finds nothing to undo.
    expect(page.get_by_text('Moved 1 crop').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    assert len(move_calls) == 1, f'expected exactly one move request: {move_calls}'
    _, move_body = move_calls[0]
    assert move_body.get('cluster_uid') == TARGET_UID
    assert 'cluster_id' not in move_body
    assert len(move_body.get('crop_ids', [])) == 1, move_body

    with page.expect_response(
        lambda r: r.request.method == 'POST' and r.url.endswith('/label_ops/op-move-1/undo'),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('z')
    expect(page.get_by_text('Reverted 1').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    assert undo_calls == [{'dry_run': False}], (
        f'Z should queue one real undo by op id: {undo_calls}'
    )
    assert unexpected == [], f"a move's undo must not use the per-crop routes: {unexpected}"

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected in the move/undo flow: {errors[:3]}'
