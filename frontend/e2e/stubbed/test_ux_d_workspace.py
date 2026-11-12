"""Workspace and Review fixes of UX wave 2 in a real browser (stubbed API).

- details follow the view: a focus the next view cannot show is parked, Back restores it
- the lightbox opens from a card's hover control and Shift+Space, steps with the arrows, closes on Esc
- the map fills the Workspace height
- Review asks the preview route and the Tight/Context toggle changes `pad`
- cluster cards show the served display name, a detector name marked as the detector's
- the Gallery's Duplicates control sends the served `near_dup` filter
"""

from __future__ import annotations

import re

import pytest
from fixtures.contract import cluster_card, cluster_list
from harness.constants import ACTION_TIMEOUT_MS, review_url
from harness.screens import shot
from playwright.sync_api import expect
from test_gallery_view import serve as serve_gallery
from test_map_view import CANVAS, _meta, _open, _serve_map
from test_review_view import item as review_item, serve as serve_review
from test_table_view import serve as serve_table


def _crops(stub, page, app_url) -> None:
    serve_review(stub)
    stub.on('GET', r'/crops/crop-\d+/history$', {'crop_id': 'crop-0', 'entries': []})
    stub.on(
        'GET',
        r'/crops/locate(\?|$)',
        {
            'crop_id': 'crop-1',
            'in_view': True,
            'index': 1,
            'total': 6,
            'cursor_before': None,
            'cursor_after': None,
            'reason': None,
            'reason_message': None,
        },
    )


def _review_queue(stub) -> None:
    located = {
        'crop_id': 'crop-1',
        'in_queue': True,
        'index': 1,
        'total': 6,
        'cursor_before': None,
        'cursor_after': None,
        'reason': None,
        'sort_applied': 'detector_confidence:desc',
        'sort_fallback_reason': None,
    }
    page_ = {
        'items': [dict(review_item(i), reason='no class yet') for i in range(6)],
        'total': 6,
        'next_cursor': None,
        'limit': 30,
        'sort_applied': 'detector_confidence:desc',
        'sort_fallback_reason': None,
        'empty_reason': None,
        'hidden': {'test_holdout': 0, 'excluded': 0},
        'join_truncated': False,
    }
    stub.on('GET', r'/review/to_label(\?|$)', page_)
    stub.on('GET', r'/review/to_label/locate(\?|$)', located)


@pytest.mark.parametrize(
    ('start', 'target', 'item_survives'),
    [
        ('detections', 'gallery', False),
        ('detections', 'table', True),
        ('detections', 'map', False),
    ],
)
def test_an_item_focus_follows_the_view_it_fits(stub, page, app_url, start, target, item_survives):
    _crops(stub, page, app_url)
    serve_gallery(stub, [])
    _serve_map(stub, page, app_url, _meta(app_url))
    serve_table(stub)
    page.goto(f'{app_url}/p/default/workspace?view={start}&focus=item%3Acrop-1')
    expect(page.get_by_test_id('details-panel')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id(f'view-tab-{target}').click()
    page.wait_for_url(f'**view={target}**')
    survives = 'focus=item' in page.url.replace('%3A', ':')
    assert survives == item_survives, page.url
    if not item_survives:
        page.go_back()
        page.wait_for_url(f'**view={start}**')
        assert 'focus=item' in page.url.replace('%3A', ':'), page.url


def test_an_image_focus_is_parked_when_clusters_open_and_comes_back(stub, page, app_url):
    serve_gallery(stub, [])
    stub.on('GET', r'/clusters(\?|$)', cluster_list(cluster_card('cl-1')))
    page.goto(f'{app_url}/p/default/workspace?view=gallery&focus=image%3Aimg-1')
    expect(page.get_by_test_id('image-details')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('view-tab-clusters').click()
    page.wait_for_url('**view=clusters**')
    assert 'focus=image' not in page.url.replace('%3A', ':'), page.url
    expect(page.get_by_test_id('image-details')).to_have_count(0)
    page.get_by_test_id('view-tab-gallery').click()
    page.wait_for_url('**view=gallery**')
    assert 'focus=image:img-1' in page.url.replace('%3A', ':'), page.url


def test_lightbox_opens_steps_and_closes(stub, page, app_url):
    serve_gallery(stub, [])
    stub.on('GET', r'/images/img-\d+/file', (200, b'', 'image/jpeg'))
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    card = page.get_by_test_id('image-card').nth(0)
    card.hover()
    card.get_by_test_id('image-card-lightbox').click()
    box = page.get_by_test_id('lightbox')
    expect(box).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    expect(box.locator('img')).to_have_attribute('src', re.compile('max_dim=2048'))
    shot(page, 'lightbox', widths=(1600, 430))
    page.keyboard.press('+')
    expect(page.get_by_test_id('lightbox-zoom')).to_have_text('150%')
    page.keyboard.press('Escape')
    expect(box).to_have_count(0)
    page.get_by_test_id('image-card-select').nth(1).focus()
    page.keyboard.press('Shift+Space')
    expect(page.get_by_test_id('lightbox')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('ArrowRight')
    expect(page.get_by_test_id('lightbox').locator('img')).to_have_attribute(
        'src', re.compile('img-2')
    )


def test_the_map_fills_the_workspace_height(stub, page, app_url):
    _serve_map(stub, page, app_url, _meta(app_url))
    page.set_viewport_size({'width': 1600, 'height': 1000})
    _open(page, app_url)
    canvas = page.locator('[data-testid="map-canvas"]').bounding_box()
    view = page.get_by_test_id('workspace-view').bounding_box()
    assert canvas and view
    assert canvas['height'] >= 0.8 * view['height'], (canvas, view)
    expect(page.get_by_test_id('map-cell-list')).to_be_visible()
    shot(page, 'map', widths=(1600, 430))
    assert page.locator(CANVAS).count() == 1


def test_review_asks_the_preview_route_and_the_toggle_changes_pad(stub, page, app_url):
    _crops(stub, page, app_url)
    _review_queue(stub)
    urls: list[str] = []
    stub.on(
        'GET',
        r'/crops/[^/]+/preview',
        lambda r, m: (urls.append(r.url), (200, b'', 'image/jpeg'))[1],
    )
    page.goto(review_url(app_url, 'to_label', 'crop-1'))
    expect(page.get_by_test_id('review-crop-panel')).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    page.wait_for_timeout(300)
    assert urls and 'pad=0.6' in urls[-1] and 'outline=true' in urls[-1], urls
    shot(page, 'review-context', widths=(1600, 430))
    page.get_by_role('radio', name='Tight').click()
    page.wait_for_timeout(300)
    assert 'pad=0' in urls[-1] and 'pad=0.6' not in urls[-1], urls
    shot(page, 'review-tight', widths=(1600, 430))


def test_cluster_cards_show_the_served_name(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': []})
    stub.on(
        'GET',
        r'/clusters(\?|$)',
        cluster_list(
            cluster_card(
                'cl-det',
                dominant_class_name=None,
                display_name='bus',
                display_name_kind='detector',
                near_dup_group_count=2,
            ),
            cluster_card(
                'cl-none', dominant_class_name=None, display_name='', display_name_kind='none'
            ),
        ),
    )
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    name = page.get_by_test_id('cluster-name')
    expect(name).to_have_text('bus', timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_text('from the detector')).to_be_visible()
    expect(page.get_by_test_id('cluster-near-dups')).to_have_text('2 duplicate groups')
    expect(page.get_by_text('needs label')).to_have_count(1)
    shot(page, 'clusters-names', widths=(1600, 430))


def test_gallery_duplicates_control_sends_the_served_filter(stub, page, app_url):
    queries: list[dict] = []
    serve_gallery(stub, queries)
    page.goto(f'{app_url}/p/default/workspace?view=gallery')
    expect(page.get_by_test_id('image-card').first).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    with page.expect_request(lambda r: '/images?' in r.url and 'near_dup=collapsed' in r.url):
        page.get_by_role('radio', name='One per group').click()
    with page.expect_request(lambda r: r.url.endswith('/images')):
        page.get_by_role('radio', name='All images').click()
    shot(page, 'gallery-duplicates', widths=(1600, 430))


def test_a_cluster_is_labelled_by_keyboard_and_focus_moves_to_the_next(stub, page, app_url):
    from fixtures.contract import label_op_accepted
    from harness.label_ops import stub_label_op
    from test_workspace_shell import CLUSTER_UID, serve as serve_shell

    second = 'cl-fixture-0002'
    serve_shell(stub)
    stub.on(
        'GET', r'/clusters(\?|$)', cluster_list(cluster_card(CLUSTER_UID), cluster_card(second))
    )
    calls: list[dict] = []
    stub_label_op(stub, r'/crops/batch_label$', 'op-key', updated=3, method='PUT')

    def label(request, _match):
        body = request.post_data_json or {}
        calls.append(body)
        if body.get('dry_run'):
            return (200, {'dry_run': True, 'selected': 3})
        return (202, label_op_accepted('op-key', 'job-op-key'))

    stub.on('PUT', r'/crops/batch_label$', label)
    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    page.wait_for_selector('img', timeout=ACTION_TIMEOUT_MS)
    page.locator('body').click(position={'x': 5, 'y': 400})
    page.keyboard.press('l')
    box = page.get_by_role('combobox', name='Search classes')
    expect(box).to_be_focused(timeout=ACTION_TIMEOUT_MS)
    page.keyboard.type('widget')
    page.keyboard.press('Enter')
    expect(page.get_by_test_id('selection-subject')).to_have_text(
        'Acting on every item in this cluster.', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('selection-count')).to_contain_text('3', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Enter')
    expect(page).to_have_url(re.compile(f'scope=cluster(:|%3A){second}'), timeout=ACTION_TIMEOUT_MS)
    assert [c.get('dry_run') for c in calls][:1] == [True]
    assert any(not c.get('dry_run') for c in calls)
    shot(page, 'label-cluster-by-key', widths=(1600, 430))
