"""An open cluster: one primary ("Label all N as…") and no bulk bar until something is selected,
then one primary ("Label as…"); More and Refine hold every rarely used control; and the label
loop is a few keys: L, a class (typed once, then a recent one by its digit), Enter, and the next
cluster opens."""

from __future__ import annotations

import re

from fixtures.contract import cluster_card, cluster_list, fill, job, label_op_accepted
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'widget',
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': None,
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
]
UIDS = ['aaaa0001', 'bbbb0002']


def crop(i: int, uid: str) -> dict:
    return make_item(
        crop_id=f'crop-{uid}-{i}',
        image_id=f'img-{uid}-{i}',
        class_id=None,
        class_name=None,
        cluster={
            'partition_id': 'p1',
            'uid': uid,
            'state': 'assigned',
            'score': 0.9,
            'margin': 0.1,
            'assigned_by': 'run',
        },
        label_validated=False,
        label_source='model_suggestion',
        thumbnail_url=f'/curation/crops/crop-{uid}-{i}/thumbnail',
    )


def _serve(stub, writes: list[dict]) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    cards = [
        cluster_card(uid, display_no=i + 1, size=3, dominant_class_name=None, labelled_count=0)
        for i, uid in enumerate(UIDS)
    ]
    stub.on('GET', r'/clusters(\?|$)', cluster_list(*cards))

    def crops(request, _match):
        uid = next((u for u in UIDS if u in request.url), UIDS[0])
        return (
            200,
            {
                'total': 3,
                'next_cursor': None,
                'limit': 60,
                'crops': [crop(i, uid) for i in range(3)],
            },
        )

    stub.on('GET', r'/crops(\?|$)', crops)

    def label(request, _match):
        body = request.post_data_json or {}
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 3})
        writes.append(body)
        return (202, label_op_accepted(f'op-{len(writes)}', f'job-{len(writes)}'))

    stub.on('PUT', r'/crops/batch_label$', label)
    for n in (1, 2):
        stub.on(
            'GET',
            rf'/jobs/job-{n}$',
            job(
                f'job-{n}',
                result={'op_id': f'op-{n}', 'updated': 3, 'conflicts': 0, 'complete': True},
            ),
        )


def _open(page, app_url, uid=UIDS[0]) -> None:
    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{uid}')
    expect(page.get_by_test_id('label-cluster')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    # The class registry has loaded: a picker opened earlier would offer to create the class.
    expect(page.get_by_test_id('workspace-rail')).to_contain_text(
        'widget', timeout=ACTION_TIMEOUT_MS
    )


def test_no_selection_has_one_primary_and_no_bulk_bar(stub, page, app_url):
    _serve(stub, [])
    _open(page, app_url)
    primary = page.locator('[data-testid=workspace-view] .btn-primary:visible')
    expect(primary).to_have_count(1)
    expect(primary).to_contain_text(re.compile(r'Label all \d+ as'))
    expect(page.get_by_test_id('bulk-bar')).to_have_count(0)


def test_a_selection_swaps_the_row_for_the_bulk_bar_with_one_primary(stub, page, app_url):
    _serve(stub, [])
    _open(page, app_url)
    page.locator('[data-crop-id]').first.click()
    expect(page.get_by_test_id('bulk-bar')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    primary = page.locator('[data-testid=workspace-view] .btn-primary:visible')
    expect(primary).to_have_count(1)
    expect(primary).to_contain_text('Label as')
    expect(page.get_by_test_id('cluster-actions')).to_have_count(0)


def test_more_and_refine_hold_every_rare_control(stub, page, app_url):
    _serve(stub, [])
    stub.on('GET', r'/clusters/aaaa0001/facets$', fill('ClusterFacets', cluster_uid=UIDS[0]))
    _open(page, app_url)
    page.get_by_test_id('scope-more').click()
    menu = page.get_by_role('menu', name='More')
    for name in (
        'Flag selected as a new class',
        "Accept the VLM's suggestions on this page",
        'Select a diverse sample',
        'Undo last change',
    ):
        expect(menu.get_by_role('menuitem', name=name)).to_be_visible()
    page.keyboard.press('Escape')
    page.get_by_test_id('scope-refine').click()
    refine = page.get_by_test_id('cluster-detail-refine')
    for text in ('Label source', 'Subject', 'Minimum clarity', 'Order'):
        expect(refine).to_contain_text(text)
    # One Sort in the header; the facet block's own sort is gone.
    expect(page.get_by_test_id('scope-sort')).to_have_count(1)
    expect(page.get_by_test_id('facet-sort')).to_have_count(0)


def test_the_label_loop_is_a_few_keys_per_cluster(stub, page, app_url):
    writes: list[dict] = []
    _serve(stub, writes)
    _open(page, app_url)
    keys: list[str] = []

    def press(key: str) -> None:
        keys.append(key)
        page.keyboard.press(key)

    # First cluster: the class is typed once.
    press('l')
    expect(page.get_by_role('combobox', name='Search classes')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    for ch in 'widget':
        press(ch)
    press('Enter')
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    press('Enter')
    # The write is queued and followed in the background; the next cluster opens at once.
    page.wait_for_url(f'**scope=cluster%3A{UIDS[1]}*', timeout=ACTION_TIMEOUT_MS)

    # Second cluster: the recent class is one key.
    keys.clear()
    press('l')
    expect(page.get_by_role('combobox', name='Search classes')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    press('1')
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    press('Enter')
    for _ in range(100):
        if len(writes) == 2:
            break
        page.wait_for_timeout(100)
    assert keys == ['l', '1', 'Enter'], keys
    assert all(w['class_id'] == 1 for w in writes), writes


def test_a_mixed_cluster_warns_before_the_whole_cluster_is_labelled(stub, page, app_url):
    _serve(stub, [])
    mixed = cluster_card(
        UIDS[0],
        display_no=1,
        size=3,
        dominant_class_name=None,
        labelled_count=0,
        flags=['mixed_detector_classes'],
        detector_top_class='cup',
        detector_top_share=0.5,
    )
    stub.on('GET', r'/clusters(\?|$)', cluster_list(mixed))
    stub.on('GET', rf'/clusters/{UIDS[0]}$', mixed)
    _open(page, app_url)
    page.get_by_test_id('label-cluster').click()
    page.get_by_role('combobox', name='Search classes').fill('widget')
    page.keyboard.press('Enter')
    warning = page.get_by_test_id('selection-warning')
    expect(warning).to_contain_text('This cluster is mixed', timeout=ACTION_TIMEOUT_MS)
    expect(warning).to_contain_text('cup')
    expect(warning).to_contain_text('Or choose a class below to label all of them')
