"""The cluster list renders what the server returns: the served sort options and
flag vocabulary, the served order, the served quality fields, the truncation
notice, and a sort/filter choice that survives a reload through the URL.

The stub plays the server: it orders and narrows the cards from the query it
receives, so the page can only show the order it was sent.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from fixtures.contract import cluster_card, cluster_list, fill
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect


METHODS = {'strategies': [], 'flags': {}}
CLASSES: list[dict] = []

SORT_OPTIONS = [
    {'id': 'size_desc', 'label': 'Largest first'},
    {'id': 'needs_review_desc', 'label': 'Most to review'},
    {'id': 'size_asc', 'label': 'Smallest first'},
]
FLAG_VOCAB = [
    {'id': 'unvalidated', 'label': 'Unvalidated', 'description': 'No member is validated.'},
    {
        'id': 'mixed_detector_classes',
        'label': 'Mixed detector classes',
        'description': 'The detector gave different classes.',
    },
    {
        'id': 'label_disagreement',
        'label': 'Label disagreement',
        'description': 'The labelled members name different classes.',
    },
]

# The card shows the served display_no, so each card gets its own.
UIDS = {1: 'aaaa0001', 2: 'bbbb0002', 3: 'cccc0003'}


def card(cid: int, size: int, review: int, flags: list[str]) -> dict:
    return cluster_card(
        UIDS[cid],
        display_no=cid,
        size=size,
        labelled_count=0,
        dominant_class_name=None,
        dominant_count=0,
        label_agreement=None,
        labelled_share=None,
        is_unlabeled=True,
        representatives=[fill('ClusterRepresentative', crop_id=f'crop-{cid}', cluster_score=0.9)],
        detector_top_class='widget_x',
        detector_top_share=0.9,
        detector_conf_mean=0.8,
        review_counts={'all': review, 'outliers': 1},
        needs_review_count=review,
        flags=flags,
    )


CARDS = {
    1: card(1, 50, 2, ['unvalidated']),
    2: card(2, 40, 9, ['unvalidated', 'mixed_detector_classes']),
    3: card(3, 10, 5, []),
}
ORDERS = {
    'size_desc': [1, 2, 3],
    'needs_review_desc': [2, 3, 1],
    'size_asc': [3, 2, 1],
}


def clusters_handler(requests: list[dict], truncated: bool = False):
    def handler(request, _match):
        q = parse_qs(urlparse(request.url).query)
        requests.append(q)
        if 'cluster_uid' in q:
            one = next(c for c in CARDS.values() if c['cluster_uid'] == q['cluster_uid'][0])
            return (200, cluster_list(one, sort_options=SORT_OPTIONS, flag_vocabulary=FLAG_VOCAB))
        sort = q.get('sort', ['size_desc'])[0]
        ids = ORDERS[sort]
        for flag in q.get('flag', []):
            ids = [i for i in ids if flag in CARDS[i]['flags']]
        if q.get('needs_review') == ['true']:
            ids = [i for i in ids if CARDS[i]['needs_review_count'] > 0 and i != 3]
        return cluster_list(
            *[CARDS[i] for i in ids],
            representatives_limit=24,
            sort=sort,
            sort_options=SORT_OPTIONS,
            flag_vocabulary=FLAG_VOCAB,
            truncated=truncated,
        )

    return handler


def serve(stub, requests, truncated=False):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/methods(\?|$)', METHODS)
    stub.on('GET', r'/regions(\?|$)', {'items': [], 'total': 0})
    stub.on('GET', r'/clusters(\?|$)', clusters_handler(requests, truncated))


def card_ids(page) -> list[str]:
    # textContent, not innerText: cards below the fold are content-visibility:auto.
    return page.locator(
        '[data-testid=workspace-view] ul > li button [data-testid=cluster-uid]'
    ).evaluate_all('els => els.map(e => e.textContent.trim())')


def test_cards_render_in_the_served_order_with_served_quality_fields(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert card_ids(page) == ['#1', '#2', '#3']
    assert 'sort' not in requests[0]
    # One list request carries the first screen's thumbnails.
    assert len(requests) == 1
    assert requests[0]['offset'] == ['0']
    assert requests[0]['limit'] == ['24']
    second = page.locator('[data-testid=workspace-view] ul > li').nth(1)
    # One meta line and at most one status chip: the highest of review, mixed, unvalidated.
    expect(second.get_by_test_id('cluster-meta')).to_contain_text('detector: widget_x')
    status = second.get_by_test_id('cluster-status')
    expect(status).to_have_count(1)
    expect(status).to_contain_text('9 to review')
    options = page.get_by_test_id('cluster-sort').locator('option').all_inner_texts()
    assert options == ['Largest first', 'Most to review', 'Smallest first']


def test_a_chosen_sort_is_sent_and_the_served_order_is_shown(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    page.get_by_test_id('cluster-sort').select_option('needs_review_desc')
    expect(
        page.locator('[data-testid=workspace-view] ul > li button [data-testid=cluster-uid]').first
    ).to_have_text('#2', timeout=ACTION_TIMEOUT_MS)
    assert card_ids(page) == ['#2', '#3', '#1']
    assert requests[-1]['sort'] == ['needs_review_desc']
    assert 'sort=needs_review_desc' in page.url


def test_filters_reach_the_server_survive_a_reload_and_clear(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    page.get_by_test_id('cluster-refine').click()
    page.get_by_test_id('cluster-flag-filter-mixed_detector_classes').click()
    expect(page.locator('[data-testid=workspace-view] ul > li')).to_have_count(
        1, timeout=ACTION_TIMEOUT_MS
    )
    assert requests[-1]['flag'] == ['mixed_detector_classes']
    expect(page).to_have_url(re.compile('flag=mixed_detector_classes'), timeout=ACTION_TIMEOUT_MS)

    page.reload()
    expect(page.locator('[data-testid=workspace-view] ul > li')).to_have_count(
        1, timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('refinement-chips')).to_contain_text('Mixed detector classes')
    page.get_by_test_id('cluster-refine').click()
    expect(page.get_by_test_id('cluster-flag-filter-mixed_detector_classes')).to_have_attribute(
        'aria-pressed', 'true'
    )
    page.keyboard.press('Escape')

    page.get_by_test_id('cluster-filters-clear').click()
    expect(page.locator('[data-testid=workspace-view] ul > li')).to_have_count(
        3, timeout=ACTION_TIMEOUT_MS
    )
    expect(page).not_to_have_url(re.compile('flag='), timeout=ACTION_TIMEOUT_MS)


def test_a_filter_with_no_match_shows_the_empty_state_and_clears(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    page.get_by_test_id('cluster-refine').click()
    page.get_by_test_id('cluster-flag-filter-label_disagreement').click()
    empty = page.get_by_test_id('clusters-filtered-empty')
    expect(empty).to_contain_text('No clusters match these filters', timeout=ACTION_TIMEOUT_MS)
    page.keyboard.press('Escape')
    empty.get_by_role('button', name='Clear filters').click()
    expect(page.locator('[data-testid=workspace-view] ul > li')).to_have_count(
        3, timeout=ACTION_TIMEOUT_MS
    )


def test_a_truncated_list_says_so(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests, truncated=True)
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.get_by_test_id('clusters-truncated')).to_contain_text(
        '3 clusters', timeout=ACTION_TIMEOUT_MS
    )


def test_a_failed_list_shows_its_error_and_retries(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    stub.on('GET', r'/clusters(\?|$)', (500, {'detail': 'down'}))
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.get_by_test_id('clusters-error')).to_contain_text(
        'Could not load clusters', timeout=ACTION_TIMEOUT_MS
    )
    stub.on('GET', r'/clusters(\?|$)', clusters_handler(requests))
    page.get_by_role('button', name='Retry').click()
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )


def test_unrelated_item_filters_keep_the_cluster_filter_in_the_url(stub, page, app_url):
    requests: list[dict] = []
    serve(stub, requests)
    page.goto(f'{app_url}/p/default/workspace?view=clusters&flag=unvalidated&sort=size_asc')
    expect(page.locator('[data-testid=workspace-view] ul > li button').first).to_be_visible(
        timeout=ACTION_TIMEOUT_MS
    )
    assert card_ids(page) == ['#2', '#1']
    assert requests[0]['flag'] == ['unvalidated']
    assert requests[0]['sort'] == ['size_asc']
