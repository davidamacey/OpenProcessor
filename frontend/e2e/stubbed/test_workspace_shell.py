"""The Workspace shell: one page, one filter, one selection across views.

- a cluster card opens in place (scope chip), the class picker assigns the selection through
  the server's dry run and only then writes `PUT /crops/batch_label`
- the filter, queue chip and selection survive a switch to Detections and back, and the bulk bar
  keeps stating the server's summary
- Back restores the previous view
- at 430 px the rail and the details are drawers and the filter bar is one button
"""

from __future__ import annotations

import re

from fixtures.contract import cluster_card, cluster_list, label_op_accepted
from fixtures.wire import make_item
from harness.constants import ACTION_TIMEOUT_MS
from harness.label_ops import stub_label_op
from playwright.sync_api import expect


CLASSES = [
    {
        'class_id': 1,
        'class_name': 'widget',
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': 'k',
        'sample_count': 10,
        'validated_count': 5,
        'deprecated': False,
    },
    {
        'class_id': 2,
        'class_name': 'gadget',
        'kind': 'item',
        'group': 'g',
        'hotkey_letter': 'j',
        'sample_count': 4,
        'validated_count': 1,
        'deprecated': False,
    },
]
CLUSTER_UID = 'cl-fixture-0001'
CLUSTERS = cluster_list(cluster_card(CLUSTER_UID, size=3, dominant_class_name='widget'))


def crop(i: int) -> dict:
    return make_item(
        crop_id=f'crop-{i}',
        image_id=f'img-{i}',
        class_id=1,
        class_name='widget',
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


def serve(stub) -> None:
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [crop(i) for i in range(3)]},
    )


def test_cluster_opens_in_place_and_assigns_through_a_dry_run(stub, page, app_url):
    serve(stub)
    label_calls: list[dict] = []
    stub_label_op(stub, r'/crops/batch_label$', 'op-label-1', updated=2, method='PUT')

    def label(request, _match):
        body = request.post_data_json or {}
        label_calls.append(body)
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 2})
        return (202, label_op_accepted('op-label-1', 'job-op-label-1'))

    stub.on('PUT', r'/crops/batch_label$', label)

    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    page.get_by_text('Cluster', exact=False).first.wait_for(timeout=ACTION_TIMEOUT_MS)
    page.locator('[data-testid=workspace-view] li button').first.click()
    expect(page.get_by_test_id('scope-chip')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    assert (
        f'scope=cluster%3A{CLUSTER_UID}' in page.url or f'scope=cluster:{CLUSTER_UID}' in page.url
    )

    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    page.locator('img').nth(0).click()
    page.keyboard.down('Shift')
    page.locator('img').nth(1).click()
    page.keyboard.up('Shift')
    expect(page.get_by_test_id('bulk-summary')).to_have_text(
        '2 detections in 2 images', timeout=ACTION_TIMEOUT_MS
    )

    page.get_by_test_id('assign-selected').click()
    page.get_by_role('combobox', name='Search classes').fill('widget')
    page.get_by_role('dialog').get_by_role('option', name=re.compile('widget')).first.click()
    expect(page.get_by_test_id('selection-count')).to_contain_text('2', timeout=ACTION_TIMEOUT_MS)
    assert [c.get('dry_run') for c in label_calls] == [True]

    with page.expect_response(
        lambda r: (
            r.request.method == 'PUT'
            and r.url.endswith('/batch_label')
            and not (r.request.post_data_json or {}).get('dry_run')
        ),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.get_by_role('button', name='Apply').click()
    assert label_calls[-1]['selection']['crop_ids'] == ['crop-0', 'crop-1']
    errors = [c for c in stub.console_errors if c.startswith('pageerror')]
    assert not errors, errors[:3]


def test_class_letter_then_enter_labels_a_pure_cluster(stub, page, app_url):
    serve(stub)
    label_calls: list[dict] = []
    stub_label_op(stub, r'/crops/batch_label$', 'op-label-2', updated=3, method='PUT')

    def label(request, _match):
        body = request.post_data_json or {}
        label_calls.append(body)
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 3})
        return (202, label_op_accepted('op-label-2', 'job-op-label-2'))

    stub.on('PUT', r'/crops/batch_label$', label)

    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    page.locator('[data-testid=workspace-view] li button').first.click()
    expect(page.get_by_test_id('scope-chip')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)

    page.keyboard.press('k')
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    assert [c.get('dry_run') for c in label_calls] == [True]
    assert label_calls[0]['selection']['filter']['cluster_uids'] == [CLUSTER_UID]

    with page.expect_response(
        lambda r: (
            r.request.method == 'PUT'
            and r.url.endswith('/batch_label')
            and not (r.request.post_data_json or {}).get('dry_run')
        ),
        timeout=ACTION_TIMEOUT_MS,
    ):
        page.keyboard.press('Enter')
    assert label_calls[-1]['selection']['filter']['cluster_uids'] == [CLUSTER_UID]
    assert label_calls[-1]['class_id'] == 1


def test_filter_queue_and_selection_survive_a_view_switch(stub, page, app_url):
    serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=detections&class_name=widget')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    for i in range(3):
        page.locator('img').nth(i).click(modifiers=['Control'])
    expect(page.get_by_test_id('bulk-summary')).to_have_text(
        '3 detections in 3 images', timeout=ACTION_TIMEOUT_MS
    )

    page.get_by_test_id('view-tab-clusters').click()
    expect(page).to_have_url(re.compile('view=clusters'))
    assert 'class_name=widget' in page.url
    expect(page.get_by_test_id('item-filter-chip').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('view-tab-detections').click()
    expect(page.get_by_test_id('bulk-summary')).to_have_text(
        '3 detections in 3 images', timeout=ACTION_TIMEOUT_MS
    )


def test_comma_and_period_step_between_clusters(stub, page, app_url):
    serve(stub)
    second = 'cl-fixture-0002'
    stub.on(
        'GET', r'/clusters(\?|$)', cluster_list(cluster_card(CLUSTER_UID), cluster_card(second))
    )
    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('.')
    expect(page).to_have_url(re.compile(f'scope=cluster(:|%3A){second}'), timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press(',')
    expect(page).to_have_url(
        re.compile(f'scope=cluster(:|%3A){CLUSTER_UID}'), timeout=ACTION_TIMEOUT_MS
    )


def test_slash_opens_the_class_picker_and_space_toggles_the_focused_card(stub, page, app_url):
    serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    thumb = page.locator('button[aria-pressed]').first
    thumb.focus()
    page.keyboard.press('Space')
    expect(page.get_by_test_id('bulk-count')).to_have_text('1 selected', timeout=ACTION_TIMEOUT_MS)
    page.locator('button[aria-pressed]').nth(1).focus()
    page.keyboard.press('Space')
    expect(page.get_by_test_id('bulk-count')).to_have_text('2 selected', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('/')
    expect(page.get_by_role('combobox', name='Search classes')).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )


def test_the_filter_bar_states_the_served_total_and_what_is_hidden(stub, page, app_url):
    serve(stub)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {
            'total': 3,
            'next_cursor': None,
            'limit': 60,
            'crops': [crop(i) for i in range(3)],
            'hidden': {'test_holdout': 2, 'excluded': 1},
        },
    )
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    bar = page.get_by_test_id('workspace-filter-bar')
    expect(bar).to_contain_text('3 matching', timeout=ACTION_TIMEOUT_MS)
    expect(bar).to_contain_text('2 test holdout and 1 ignored not listed')


def test_back_restores_the_previous_view(stub, page, app_url):
    serve(stub)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    page.get_by_test_id('view-tab-detections').wait_for(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('view-tab-detections').click()
    expect(page.get_by_test_id('detections-view')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.go_back()
    expect(page).to_have_url(re.compile('view=clusters'))
    expect(page.get_by_test_id('detections-view')).to_have_count(0)


def test_rail_and_details_are_drawers_at_430(stub, page, app_url):
    serve(stub)
    page.set_viewport_size({'width': 430, 'height': 900})
    page.goto(f'{app_url}/p/default/workspace?view=detections')
    page.get_by_test_id('view-bar').wait_for(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('workspace-rail')).to_be_hidden()
    expect(page.get_by_test_id('details-panel')).to_have_count(0)
    expect(page.get_by_test_id('filters-toggle')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')

    page.get_by_test_id('rail-toggle').click()
    expect(page.get_by_test_id('workspace-rail')).to_be_visible()
    page.get_by_test_id('workspace-rail').get_by_role('button', name='Close').click()
    page.get_by_test_id('details-toggle').click()
    expect(page.get_by_test_id('details-panel')).to_be_visible()
