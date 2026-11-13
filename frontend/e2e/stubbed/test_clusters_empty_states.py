"""The cluster list tells the truth about what the server served. A pending activation shows
the Activate panel (never a client-built inventory card), the grid fills after an activation
without a reload, and the keyboard reaches the cards: arrows move, Enter opens, `.` opens the
first."""

from __future__ import annotations

from fixtures.contract import cluster_card, cluster_list, fill, job
from fixtures.wire import make_box, make_item
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_cluster_discard import CLASSES


UIDS = ['aaaa0001', 'bbbb0002', 'cccc0003']


def _cards() -> list[dict]:
    return [
        cluster_card(uid, display_no=i + 1, size=10 - i, dominant_class_name='cup')
        for i, uid in enumerate(UIDS)
    ]


def _region_rows() -> list[dict]:
    """Region rows that the old grid turned into a pinned inventory card."""
    boxes = [make_box(f'b{n}') for n in (1, 2)]
    item = make_item(crop_id='r-1', image_id='img-1', region_boxes=boxes)
    return [{**item, 'region_box_id': b['box_id'], 'row_key': f'r-1#{b["box_id"]}'} for b in boxes]


def _partition(serving: bool) -> dict:
    return fill(
        'ClusterPartitionView',
        id='part-pending-1',
        target='item',
        level='item',
        method='hdbscan',
        status='ready',
        serving=serving,
        n_clusters=46,
        n_items=279,
        created_at='2026-11-11T10:00:00Z',
    )


def _serve_pending(stub) -> dict:
    served = {'now': False}
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/regions(\?|$)', {'items': _region_rows(), 'total': 2})

    def clusters(_request, _match):
        if served['now']:
            return cluster_list(*_cards())
        return cluster_list(
            empty_reason='pending_activation',
            empty_message='A clustering run finished but is not serving yet.',
            pending_partition={'partition_id': 'part-pending-1', 'clusters': 46, 'items': 279},
        )

    stub.on('GET', r'/clusters(\?|$)', clusters)
    stub.on(
        'GET',
        r'/clusters/partitions(\?|$)',
        lambda _r, _m: fill('ClusterPartitionList', partitions=[_partition(served['now'])]),
    )

    def activate(_request, _match):
        served['now'] = True
        return (202, {'job_id': 'job-act'})

    stub.on('PUT', r'/clusters/activation$', activate)
    stub.on('GET', r'/jobs/job-act$', job('job-act', 'completed'))
    return served


def test_pending_activation_shows_the_activate_panel_and_no_inventory_card(stub, page, app_url):
    _serve_pending(stub)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    empty = page.get_by_test_id('clusters-empty')
    expect(empty).to_contain_text('Clusters are ready to activate', timeout=ACTION_TIMEOUT_MS)
    expect(empty).to_contain_text('46 clusters over 279 items')
    expect(page.get_by_test_id('cluster-run-activate')).to_be_visible()
    # The served empty state wins: no client-built card stands in front of it.
    expect(page.get_by_test_id('slot-card-title')).to_have_count(0)
    # Activation is the one primary here.
    expect(page.locator('[data-testid=workspace-view] .btn-primary:visible')).to_have_count(1)


def test_the_grid_fills_after_an_activation_without_a_reload(stub, page, app_url):
    _serve_pending(stub)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    page.get_by_test_id('cluster-run-activate').click(timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('cluster-uid')).to_have_count(3, timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('clusters-empty')).to_have_count(0)


def test_arrow_keys_move_between_cards_and_enter_opens_one(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', cluster_list(*_cards()))
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    cards = page.locator('[data-grid-item]')
    expect(cards).to_have_count(3, timeout=ACTION_TIMEOUT_MS)
    cards.first.focus()
    page.keyboard.press('ArrowRight')
    expect(cards.nth(1)).to_be_focused()
    page.keyboard.press('Enter')
    page.wait_for_url(f'**scope=cluster%3A{UIDS[1]}*', timeout=ACTION_TIMEOUT_MS)


def test_dot_opens_the_first_cluster_from_the_list(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', cluster_list(*_cards()))
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.locator('[data-grid-item]')).to_have_count(3, timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('.')
    page.wait_for_url(f'**scope=cluster%3A{UIDS[0]}*', timeout=ACTION_TIMEOUT_MS)


def test_the_primary_button_opens_the_first_cluster(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', cluster_list(*_cards()))
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    primary = page.get_by_test_id('open-first-cluster')
    expect(primary).to_be_visible(timeout=ACTION_TIMEOUT_MS)
    primary.click()
    page.wait_for_url(f'**scope=cluster%3A{UIDS[0]}*', timeout=ACTION_TIMEOUT_MS)
