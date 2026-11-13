"""The first page of a large cluster can take seconds (the backend orders every
member). Until it arrives the view says it is ordering the crops after 1.5 s and
its footer reads 'loading…', not '0 / 0 listed' and 'all loaded'."""

from __future__ import annotations

import json

from fixtures.contract import cluster_card, cluster_list, fill, job
from harness.constants import ACTION_TIMEOUT_MS
from playwright.sync_api import expect
from test_cluster_discard import CLASSES, CLUSTER_UID, CLUSTERS, crop


def test_first_page_of_a_slow_cluster_says_what_it_is_doing(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on('GET', r'/clusters(\?|$)', CLUSTERS)
    stub.on(
        'GET',
        r'/crops(\?|$)',
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [crop(i) for i in range(3)]},
    )
    held: list = []
    # Registered after the stub's catch-all, so it wins: hold the crops page.

    def hold(route) -> None:
        held.append(route)

    page.route('**/crops?*cluster_uid=*', hold)

    page.goto(f'{app_url}/p/default/workspace?view=clusters&scope=cluster:{CLUSTER_UID}')
    hint = page.get_by_test_id('cluster-slow-load-hint')
    expect(hint).to_contain_text(
        'A large cluster can take a few seconds.', timeout=ACTION_TIMEOUT_MS
    )
    expect(hint).to_contain_text('Ordering')
    footer = page.get_by_test_id('cluster-status-bar')
    expect(footer).to_contain_text('loading…')
    expect(footer).not_to_contain_text('all loaded')
    expect(footer).not_to_contain_text('0 / 0 listed')

    body = json.dumps(
        {'total': 3, 'next_cursor': None, 'limit': 60, 'crops': [crop(i) for i in range(3)]}
    )
    for route in held:
        route.fulfill(status=200, content_type='application/json', body=body)
    expect(footer).to_contain_text('3 / 3 listed', timeout=ACTION_TIMEOUT_MS)
    expect(footer).to_contain_text('all loaded')
    expect(hint).to_have_count(0)


def _pending_partition_view(serving: bool = False) -> dict:
    return fill(
        'ClusterPartitionView',
        id='part-pending-1',
        target='item',
        level='item',
        method='hdbscan',
        status='ready',
        serving=serving,
        n_clusters=9,
        n_items=120,
        n_review=0,
        n_unassigned=4,
        created_at='2026-11-11T10:00:00Z',
    )


def test_empty_clusters_page_says_a_built_partition_waits_and_activates_it(stub, page, app_url):
    """The grid shows the served reason and the activate banner, and reloads once it serves."""
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    served = {'now': False}

    def clusters(_request, _match):
        if served['now']:
            return cluster_list(cluster_card(CLUSTER_UID, size=3))
        return cluster_list(
            empty_reason='pending_activation',
            empty_message='A clustering run finished but is not serving yet. Activate it to see its clusters.',
            pending_partition={'partition_id': 'part-pending-1', 'clusters': 9, 'items': 120},
        )

    stub.on('GET', r'/clusters(\?|$)', clusters)

    def partitions(_request, _match):
        return fill(
            'ClusterPartitionList',
            partitions=[_pending_partition_view(serving=served['now'])],
        )

    stub.on('GET', r'/clusters/partitions(\?|$)', partitions)

    def activate(_request, _match):
        served['now'] = True
        return (202, {'job_id': 'job-act'})

    stub.on('PUT', r'/clusters/activation$', activate)
    stub.on('GET', r'/jobs/job-act$', job('job-act', 'completed'))

    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    message = page.get_by_test_id('clusters-empty')
    expect(message).to_contain_text('not serving yet', timeout=ACTION_TIMEOUT_MS)
    expect(message).not_to_contain_text('auto-label pipeline')
    pending = page.get_by_test_id('cluster-run-pending')
    expect(pending).to_contain_text('9 clusters', timeout=ACTION_TIMEOUT_MS)
    page.get_by_test_id('cluster-run-activate').click()
    expect(page.get_by_test_id('clusters-empty')).to_have_count(0, timeout=ACTION_TIMEOUT_MS)
    expect(page.get_by_test_id('cluster-meta')).to_have_count(1, timeout=ACTION_TIMEOUT_MS)


def test_empty_clusters_page_with_no_partition_points_at_clustering(stub, page, app_url):
    stub.on('GET', r'(?<!/stats)/classes(\?|$)', {'classes': CLASSES})
    stub.on(
        'GET',
        r'/clusters(\?|$)',
        cluster_list(
            empty_reason='no_partition',
            empty_message='No clustering run has been served yet. Run clustering from the Dashboard.',
        ),
    )
    page.goto(f'{app_url}/p/default/workspace?view=clusters')
    expect(page.get_by_test_id('clusters-empty')).to_contain_text(
        'No clustering run', timeout=ACTION_TIMEOUT_MS
    )
    expect(page.get_by_test_id('cluster-run-start')).to_be_visible()
    # One way forward, not the run panel and the served loop together.
    expect(page.get_by_test_id('empty-next-steps')).to_have_count(0)
