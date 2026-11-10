"""Region-cluster bulk triage: "Triage from a cluster
goes through batch_box_state, never the item-level batch_status, which
would flip every sibling box." Once a curator opens a region cluster
bucket, the same Confirm/No-<region>/Mark-false-positive bulk toolbar
must call POST {API_PREFIX}/regions/batch_box_state with per-box targets,
not POST {API_PREFIX}/regions/batch_status.

No fixed sleeps (parallel e2e, CLAUDE.md): every wait is a real selector
or `page.expect_request`.
"""

from __future__ import annotations

from fixtures.wire import REGION_CLASS, make_box, make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.waits import expect_handled


CLASSES = [
    {
        'class_id': 1,
        'class_name': REGION_CLASS,
        'kind': 'region',
        'group': 'widgets',
        'hotkey_letter': 'l',
        'sample_count': 40,
        'validated_count': 12,
        'cluster_size': 44,
        'deprecated': False,
    },
]

STATUSES = {
    'statuses': [],
    'confirm_status': 'detected',
    'reject_status': 'no_region_visible',
    'false_positive_status': 'false_positive',
    'box_states': [
        {
            'value': 'accepted',
            'label': 'accepted',
            'role': 'accepted',
            'human_writable': True,
            'exported': True,
            'dashed': False,
            'dim': False,
            'badge': None,
        },
    ],
}

REGION_CLUSTER_UID = 'aaaa0007'

# One card of `GET /regions/clusters`: boxes grouped by cluster_uid.
CLUSTER_CARD = {
    'cluster_uid': REGION_CLUSTER_UID,
    'size': 2,
    'box_count': 2,
    'validated_count': 0,
    'representative_crop_ids': [],
    'representative_thumb_urls': [],
}


def row(crop_id: str, box_id: str) -> dict:
    """A `/regions` row: the item (its boxes in `region_boxes`) plus the one box the row is."""
    return {
        **make_item(
            crop_id=crop_id,
            image_id=f'img-{crop_id}',
            class_id=1,
            class_name=REGION_CLASS,
            region_status='detected',
            region_boxes=[make_box(box_id, state='proposed', cluster_uid=REGION_CLUSTER_UID)],
        ),
        'region_box_id': box_id,
        'row_key': f'{crop_id}#{box_id}',
    }


ROW_1 = row('c1', 'b1')
ROW_2 = row('c2', 'b2')


def test_cluster_triage_uses_batch_box_state_with_per_box_targets(stub, page, app_url):
    box_state_calls: list[dict] = []
    batch_status_calls: list[dict] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/regions/statuses(\?|$)', STATUSES)
    stub.on(
        'GET',
        r'/regions/clusters(\?|$)',
        {'clusters': [CLUSTER_CARD], 'count': 1},
    )
    stub.on('GET', r'(?<!/regions)/clusters(\?|$)', {'clusters': [], 'count': 0})

    def regions_handler(request, match):
        return (200, {'items': [ROW_1, ROW_2], 'total': 2, 'total_rows': 2})

    stub.on('GET', r'/regions(\?|$)', regions_handler)

    def box_state_handler(request, match):
        body = request.post_data_json
        box_state_calls.append(body)
        return (
            200,
            {
                'updated': 2,
                'invalid': [],
                'conflicts': [],
                'items': [
                    {'id': 'c1', 'class_id': 1},
                    {'id': 'c2', 'class_id': 1},
                ],
            },
        )

    stub.on('POST', r'/regions/batch_box_state(\?|$)', box_state_handler)

    def batch_status_handler(request, match):
        batch_status_calls.append(request.post_data_json)
        return (200, {'updated': 2, 'conflicts': [], 'invalid': [], 'items': []})

    stub.on('POST', r'/regions/batch_status(\?|$)', batch_status_handler)

    page.goto(f'{app_url}/p/default/clusters?class={REGION_CLASS}')

    cluster_card = page.locator('button:has-text("aaaa0007")')
    cluster_card.wait_for(timeout=ACTION_TIMEOUT_MS)
    cluster_card.click()

    select_all = page.get_by_role('button', name='Select all')
    select_all.wait_for(timeout=ACTION_TIMEOUT_MS)
    select_all.click()

    verify_button = page.get_by_role('button', name='✓ Verify', exact=True)
    verify_button.wait_for(timeout=ACTION_TIMEOUT_MS)
    with expect_handled(
        page,
        lambda r: r.method == 'POST' and '/regions/batch_box_state' in r.url,
        timeout=ACTION_TIMEOUT_MS,
    ):
        verify_button.click()

    assert not batch_status_calls, (
        'cluster triage must never call the item-level batch_status route '
        f'(would flip every sibling box): {batch_status_calls}'
    )
    assert len(box_state_calls) == 1
    body = box_state_calls[0]
    assert body['state'] == 'accepted'
    targets = sorted(body['targets'], key=lambda t: t['crop_id'])
    assert targets == [
        {'crop_id': 'c1', 'box_id': 'b1'},
        {'crop_id': 'c2', 'box_id': 'b2'},
    ]

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected in the cluster-triage flow: {errors[:3]}'

    stub.assert_fail_closed()
