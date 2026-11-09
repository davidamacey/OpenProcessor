"""Live cohort-(a) scenarios: cluster cards, region clustering,
false-positive centroids, and the three flag-gated overlay jobs.

These are the endpoints the audit expected to need a GPU and that in fact
need only stored embeddings — they run sklearn/UMAP over what OpenSearch
already holds and call no model at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import (
    FP_REGION_CLUSTER_ID,
    INDEXES,
    cohort_cluster_uid,
    refresh,
    search,
    wait_for_state,
)


pytestmark = pytest.mark.live


def _region_box_buckets(
    opensearch: Any, box_filter: dict[str, Any] | None = None
) -> dict[int, int]:
    """Boxes per region cluster id: a nested terms aggregation, so each BOX
    counts once (an item with two boxes in one cluster contributes two)."""
    inner: dict[str, Any] = {
        'terms': {'field': 'region_boxes.cluster_id', 'size': 50},
    }
    agg: dict[str, Any] = {'boxes': {'nested': {'path': 'region_boxes'}, 'aggs': {'rc': inner}}}
    if box_filter is not None:
        agg = {
            'boxes': {
                'nested': {'path': 'region_boxes'},
                'aggs': {'scoped': {'filter': box_filter, 'aggs': {'rc': inner}}},
            }
        }
    resp = search(opensearch, INDEXES['items'], {'size': 0, 'aggs': agg})['aggregations']['boxes']
    buckets = (resp['scoped'] if box_filter is not None else resp)['rc']['buckets']
    return {b['key']: b['doc_count'] for b in buckets}


def _region_subid_terms(opensearch: Any, cluster_id: int) -> dict[str, int]:
    """Distinct box sub-cluster ids inside one region cluster."""
    agg = {
        'boxes': {
            'nested': {'path': 'region_boxes'},
            'aggs': {
                'in_cluster': {
                    'filter': {'term': {'region_boxes.cluster_id': cluster_id}},
                    'aggs': {
                        'subids': {'terms': {'field': 'region_boxes.cluster_subid', 'size': 50}}
                    },
                }
            },
        }
    }
    resp = search(opensearch, INDEXES['items'], {'size': 0, 'aggs': agg})['aggregations']['boxes']
    return {b['key']: b['doc_count'] for b in resp['in_cluster']['subids']['buckets']}


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


def _wait_for_job(get_state: Any, *, timeout: float = 300.0) -> dict[str, Any]:
    state = wait_for_state(get_state, lambda s: not s.get('running'), timeout=timeout)
    assert not state.get('running'), state
    return state


def test_region_clustering_partitions_the_region_pool(api_client: Any, opensearch: Any) -> None:
    resp = api_client.post('/regions/cluster', params={'auto_fp_threshold': 0.0})
    assert resp.status_code == 200, resp.text

    state = _wait_for_job(lambda: api_client.get('/regions/cluster/status').json())
    assert state.get('error') is None, state
    result = state['result']
    assert result['status'] == 'success', result
    assert result['n_boxes'] >= 32
    # Counts name their unit: a box is changed only if its cluster did, and
    # several boxes share an item.
    assert 0 < result['n_boxes_changed'] <= result['n_boxes'], result
    assert 0 < result['n_items_written'] <= result['n_boxes_changed'], result

    refresh(opensearch, INDEXES['items'])
    assigned = _region_box_buckets(opensearch)
    # The permanent false-positive bucket must survive a re-partition.
    assert assigned.get(FP_REGION_CLUSTER_ID, 0) == 12, assigned
    assert len([k for k in assigned if k >= 0]) >= 2, assigned


def test_region_cluster_cards_pin_the_false_positive_bucket_first(api_client: Any) -> None:
    resp = api_client.get('/regions/clusters')
    assert resp.status_code == 200, resp.text
    clusters = resp.json()['clusters']
    assert clusters, resp.text
    assert clusters[0]['id'] == FP_REGION_CLUSTER_ID
    assert clusters[0]['cluster_kind'] == 'false_positive'


def test_region_refine_writes_region_subids(api_client: Any, opensearch: Any) -> None:
    buckets = [
        {'key': key, 'doc_count': n}
        for key, n in _region_box_buckets(opensearch).items()
        if key >= 0
    ]
    biggest = max(buckets, key=lambda b: b['doc_count'])
    if biggest['doc_count'] < 4:
        pytest.skip(f'no region bucket has the 4-member AHC floor: {buckets}')

    resp = api_client.post(f'/regions/clusters/refine/{biggest["key"]}')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['action'] == 'refined', body
    assert body['n_boxes_updated'] == body['n_boxes']

    refresh(opensearch, INDEXES['items'])
    subids = _region_subid_terms(opensearch, biggest['key'])
    assert sum(subids.values()) == body['n_boxes'], subids


def test_fp_centroid_build_then_suspected_false_positives(api_client: Any) -> None:
    resp = api_client.post('/regions/fp_centroids/build')
    assert resp.status_code == 200, resp.text
    state = _wait_for_job(lambda: api_client.get('/regions/fp_centroids/status').json())
    assert state.get('error') is None, state
    result = state['result']
    assert result['status'] == 'success', result
    assert result['n_boxes'] >= 1, result

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
    assert resp.status_code == 200, resp.text
    state = wait_for_state(
        lambda: api_client.get('/scores/status').json(),
        lambda s: s.get('status') != 'running',
        timeout=300.0,
    )
    assert state['status'] != 'running', 'the scoring job never left the running state'
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
        lambda s: s.get('status') != 'running',
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
