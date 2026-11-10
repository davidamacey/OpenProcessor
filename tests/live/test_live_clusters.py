"""Live cohort-(a) scenarios: cluster cards, region clustering,
false-positive centroids, and the three flag-gated overlay jobs.

These are the endpoints the audit expected to need a GPU and that in fact
need only stored embeddings — they run sklearn/UMAP over what OpenSearch
already holds and call no model at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import INDEXES, cohort_cluster_uid, refresh, search, wait_for_state, wait_job


pytestmark = pytest.mark.live

# A job view is final once it is none of these.
LIVE_STATES = ('queued', 'running', 'cancelling')


def _region_box_clusters(
    opensearch: Any, box_filter: dict[str, Any] | None = None
) -> dict[str, int]:
    """Boxes per ``cluster_uid`` of the serving region-box partition: a nested terms aggregation,
    so each BOX counts once (an item with two boxes in one cluster contributes two)."""
    inner: dict[str, Any] = {'terms': {'field': 'region_boxes.cluster_uid', 'size': 50}}
    agg: dict[str, Any] = {'boxes': {'nested': {'path': 'region_boxes'}, 'aggs': {'uid': inner}}}
    if box_filter is not None:
        agg = {
            'boxes': {
                'nested': {'path': 'region_boxes'},
                'aggs': {'scoped': {'filter': box_filter, 'aggs': {'uid': inner}}},
            }
        }
    resp = search(opensearch, INDEXES['items'], {'size': 0, 'aggs': agg})['aggregations']['boxes']
    buckets = (resp['scoped'] if box_filter is not None else resp)['uid']['buckets']
    return {b['key']: b['doc_count'] for b in buckets}


def _boxes_in_state(opensearch: Any, state: str) -> int:
    resp = search(
        opensearch,
        INDEXES['items'],
        {
            'size': 0,
            'aggs': {
                'boxes': {
                    'nested': {'path': 'region_boxes'},
                    'aggs': {'s': {'filter': {'term': {'region_boxes.state': state}}}},
                }
            },
        },
    )
    return int(resp['aggregations']['boxes']['s']['doc_count'])


def test_cluster_cards_report_agreement_and_unlabeled(api_client: Any, opensearch: Any) -> None:
    resp = api_client.get('/clusters', params={'per_cluster': 2, 'max_clusters': 100})
    assert resp.status_code == 200, resp.text
    by_uid = {item['cluster_uid']: item for item in resp.json()['items']}

    pure = by_uid[cohort_cluster_uid(opensearch, 'cls0')]
    assert pure['label_agreement'] == 1.0
    assert pure['dominant_class_name'] == 'box'
    assert pure['is_unlabeled'] is False
    assert len(pure['representatives']) == 2

    mixed = by_uid[cohort_cluster_uid(opensearch, 'cls1')]
    assert 0.5 < mixed['label_agreement'] < 0.85, mixed

    unlabeled = by_uid[cohort_cluster_uid(opensearch, 'cnd10000')]
    assert unlabeled['is_unlabeled'] is True
    assert unlabeled['size'] == 30


# ---------------------------------------------------------------------------
# Region clustering + false-positive centroids
# ---------------------------------------------------------------------------


def test_region_clustering_partitions_the_region_pool(api_client: Any, opensearch: Any) -> None:
    started = api_client.post('/clusters/runs', json={'target': 'region_box'})
    assert started.status_code == 202, started.text
    run = wait_job(api_client, started.json()['job_id'], timeout=300.0)
    assert run['status'] == 'completed', run
    result = run['result']
    assert result['status'] == 'success', result
    assert result['n_items'] >= 32, result
    assert result['n_clusters'] >= 2, result

    # A run writes a partition; the serving one keeps serving until it is activated.
    partitions = api_client.get('/clusters/partitions', params={'target': 'region_box'}).json()
    serving = next((p for p in partitions['partitions'] if p['serving']), None)
    activation = api_client.put(
        '/clusters/activation',
        json={
            'target': 'region_box',
            'partition_id': result['partition_id'],
            'expected_serving_partition_id': None if serving is None else serving['id'],
            'expected_revision': None if serving is None else serving['serving_revision'],
        },
    )
    assert activation.status_code == 202, activation.text
    projected = wait_job(api_client, activation.json()['job_id'], timeout=300.0)
    assert projected['status'] == 'completed', projected

    refresh(opensearch, INDEXES['items'])
    assigned = _region_box_clusters(opensearch)
    assert len(assigned) == result['n_clusters'], assigned
    assert sum(assigned.values()) == result['n_items'], assigned
    # Only proposed and accepted boxes are clustered: a false-positive box is never placed.
    false_positive = {'term': {'region_boxes.state': 'false_positive'}}
    assert _boxes_in_state(opensearch, 'false_positive') >= 1, 'the seed holds false-positive boxes'
    assert _region_box_clusters(opensearch, false_positive) == {}


def test_region_cluster_cards_list_the_serving_clusters(api_client: Any, opensearch: Any) -> None:
    resp = api_client.get('/regions/clusters', params={'max_clusters': 50, 'per_cluster': 2})
    assert resp.status_code == 200, resp.text
    clusters = resp.json()['clusters']
    assert clusters, resp.text
    # Cards are the clusters of the serving partition, largest first, identified by cluster_uid.
    assert {c['cluster_uid'] for c in clusters} == set(_region_box_clusters(opensearch))
    sizes = [c['box_count'] for c in clusters]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert all(c['representative_crop_ids'] for c in clusters)


def test_fp_centroid_build_then_suspected_false_positives(api_client: Any) -> None:
    resp = api_client.post('/regions/fp_centroids/build')
    assert resp.status_code == 202, resp.text
    job = wait_job(api_client, resp.json()['job_id'], timeout=300.0)
    assert job['status'] == 'completed', job
    result = job['result']
    assert result['status'] == 'success', result
    assert result['fp_centroids']['n_boxes'] >= 1, result

    suspected = api_client.get('/regions/suspected_false_positives', params={'threshold': 0.35})
    assert suspected.status_code == 200, suspected.text
    body = suspected.json()
    assert body['centroids_built'] is True, body
    # The seeded FP group is deliberately far from the good-region groups,
    # so a tight threshold must NOT drag good regions into the FP cohort.
    assert body['total'] == 0, body


# ---------------------------------------------------------------------------
# Flag-gated overlay jobs (scores / diverse select / viz projection)
# ---------------------------------------------------------------------------


def _run_scorer(api_client: Any, scorer: str) -> dict[str, Any]:
    resp = api_client.post('/scores/compute', json={'scorers': [scorer]})
    assert resp.status_code == 202, resp.text
    state = wait_for_state(
        lambda: api_client.get('/scores/status').json(),
        lambda s: s.get('status') not in LIVE_STATES,
        timeout=300.0,
    )
    assert state['status'] not in LIVE_STATES, 'the scoring job never finished'
    return state


def test_scores_compute_writes_the_overlay_fields(api_client: Any, opensearch: Any) -> None:
    state = _run_scorer(api_client, 'mistakenness')
    assert state['status'] == 'completed', state

    refresh(opensearch, INDEXES['items'])
    scored = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'exists': {'field': 'mistakenness_score'}}},
    ).json()['count']
    assert scored > 0, 'the mistakenness scorer wrote nothing'

    coverage = api_client.get('/scores/coverage')
    coverage.raise_for_status()
    mistakenness = coverage.json()['coverage']['mistakenness']
    assert mistakenness['n_scored'] == scored, mistakenness
    assert mistakenness['pct'] > 0


def test_embedding_scorers_report_their_missing_prerequisite(api_client: Any) -> None:
    """``uniqueness`` and ``near_dup`` need a serving item partition.

    The harness never runs the item-clustering pipeline, so this asserts
    the documented behaviour: the job fails loudly with an actionable
    message rather than silently writing nothing.
    """
    state = _run_scorer(api_client, 'uniqueness')
    assert state['status'] == 'failed', state
    assert 'serving item partition' in (state['error'] or ''), state
    assert state['processed'] == 0, state


def test_scores_cancel_on_an_idle_job_is_a_no_op(api_client: Any) -> None:
    resp = api_client.post('/scores/cancel')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['cancelled'] is False, body
    assert body['status'] in ('completed', 'idle', 'cancelled', 'failed'), body


def test_unknown_scorer_is_rejected(api_client: Any) -> None:
    resp = api_client.post('/scores/compute', json={'scorers': ['not_a_scorer']})
    assert resp.status_code == 400, resp.text
    assert 'unknown scorer' in resp.text


def test_diverse_select_returns_an_ordering_and_mutates_nothing(
    api_client: Any, opensearch: Any
) -> None:
    before = opensearch.get(f'/{INDEXES["items"]}/_stats/indexing').json()
    before_ops = before['_all']['primaries']['indexing']['index_total']

    resp = api_client.post(
        '/select/diverse',
        json={'k': 10, 'scope': {'cluster_uid': cohort_cluster_uid(opensearch, 'cnd10000')}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['method'] == 'kcenter_greedy'
    assert len(body['crop_ids']) == 10
    assert len(set(body['crop_ids'])) == 10
    assert body['n_pool'] >= 10

    after = opensearch.get(f'/{INDEXES["items"]}/_stats/indexing').json()
    assert after['_all']['primaries']['indexing']['index_total'] == before_ops


def test_viz_projection_rebuild_then_serve(api_client: Any, opensearch: Any) -> None:
    resp = api_client.post('/viz/projection/rebuild', params={'scope': 'residual'})
    assert resp.status_code == 202, resp.text

    state = wait_for_state(
        lambda: api_client.get('/viz/projection/status').json(),
        lambda s: s.get('status') not in LIVE_STATES,
        timeout=600.0,
        interval=2.0,
    )
    assert state['status'] == 'completed', state

    refresh(opensearch, INDEXES['items'])
    with_coords = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'exists': {'field': 'viz_x'}}},
    ).json()['count']
    assert with_coords > 0, 'the viz job wrote no coordinates'

    served = api_client.get('/viz/projection', params={'max_points': 1000})
    assert served.status_code == 200, served.text
    body = served.json()
    assert body.get('points'), body
    first = body['points'][0]
    assert {'crop_id', 'x', 'y'} <= set(first), first


def test_viz_state_index_honours_the_configured_index_prefix(opensearch: Any) -> None:
    """The UMAP run-metadata index names (viz-only and the retired
    clustering reducer's) are the bound project's, like every index:
    ``{OP_PROJECT_INDEX_PREFIX}default__umap_*``. The harness sets the
    prefix to ``verify_prj_``, so no ``op_*`` index should ever appear.
    """
    names = [i['index'] for i in opensearch.get('/_cat/indices?format=json').json()]
    unscoped = [n for n in names if n.startswith('op_') and not n.startswith('verify_')]
    assert unscoped == [], unscoped
