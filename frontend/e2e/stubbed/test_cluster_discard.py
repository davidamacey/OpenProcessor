"""New coverage (test audit recommendation 5):
`/clusters/[uid]` D discards the current selection.

Single selection -> `POST {API_PREFIX}/crops/{id}/discard` (discardCrop).
Multi selection -> `POST {API_PREFIX}/crops/discard_batch`
(discardCropsBatch). See src/routes/clusters/[id]/+page.svelte's `reg('d', ...)`.
"""

from __future__ import annotations

from fixtures.contract import cluster_card, cluster_list
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
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


def test_cluster_single_discard(stub, page, app_url):
    discard_calls: list[tuple[str, str]] = []

    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [crop(i) for i in range(3)]},
    )

    def discard_handler(request, match):
        discard_calls.append((request.method, match.string))
        return (200, crop(1))

    stub.on('POST', r'/crops/([^/]+)/discard$', discard_handler)
    # Fail-closed proof this test relies on: batch discard is registered
    # too, so a UI bug that always calls the batch endpoint (even for one
    # selection) would still be handled, not 501 — the assertion below on
    # `discard_calls` catches that class of bug instead.
    stub.on(
        'POST',
        r'/crops/discard_batch$',
        lambda req, m: (200, {'items': [], 'discarded': 0, 'conflicts': [], 'not_found': []}),
    )

    page.goto(f'{app_url}/p/default/clusters/{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)

    page.locator('img').nth(1).click()
    expect(page.get_by_text('1 selected').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)

    discard_calls.clear()
    with page.expect_response(
        lambda r: r.request.method == 'POST' and '/discard' in r.url, timeout=ACTION_TIMEOUT_MS
    ):
        page.keyboard.press('d')

    assert len(discard_calls) == 1, (
        f'D on a single selection should call the single discard endpoint: {discard_calls}'
    )
    method, path = discard_calls[0]
    assert method == 'POST'
    assert path.endswith('/discard')
    assert 'discard_batch' not in path

    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, f'no pageerror expected: {errors[:3]}'
