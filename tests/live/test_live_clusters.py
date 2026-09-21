"""Live cohort-(a) scenarios: AHC refine, auto-promote, region clustering,
false-positive centroids, and the three flag-gated overlay jobs.

These are the endpoints the audit expected to need a GPU and that in fact
need only stored embeddings — they run sklearn/UMAP over what OpenSearch
already holds and call no model at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import (
    CANDIDATE_CLUSTER_ID,
    FP_REGION_CLUSTER_ID,
    INDEXES,
    MIXED_CLUSTER_ID,
    PURE_CLUSTER_ID,
    get_doc,
    refresh,
    search,
    wait_for_state,
)


pytestmark = pytest.mark.live

NONEXISTENT_CLUSTER_ID = 99999


def _subid_terms(opensearch: Any, cluster_id: int, field: str = 'cluster_subid') -> dict[str, int]:
    """Distinct sub-cluster ids in a cluster, via a terms aggregation."""
    id_field = 'cluster_id' if field == 'cluster_subid' else 'region_cluster_id'
    body = {
        'size': 0,
        'query': {'term': {id_field: cluster_id}},
        'aggs': {'subids': {'terms': {'field': field, 'size': 50}}},
    }
    buckets = search(opensearch, INDEXES['items'], body)['aggregations']['subids']['buckets']
    return {b['key']: b['doc_count'] for b in buckets}


def _v6_crop_ids(opensearch: Any, cluster_id: int, limit: int = 5) -> list[str]:
    body = {
        'size': limit,
        '_source': False,
        'sort': [{'crop_id': {'order': 'asc'}}],
        'query': {
            'bool': {
                'must': [
                    {'term': {'cluster_id': cluster_id}},
                    {'term': {'class_source': 'v6_model'}},
                ],
                'must_not': [{'term': {'class_validated': True}}],
            }
        },
    }
    return [h['_id'] for h in search(opensearch, INDEXES['items'], body)['hits']['hits']]


# ---------------------------------------------------------------------------
# AHC refine
# ---------------------------------------------------------------------------


def test_refine_splits_a_candidate_cluster_into_subclusters(kb: Any, opensearch: Any) -> None:
    resp = kb.post(f'/clusters/refine/{CANDIDATE_CLUSTER_ID}')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['action'] == 'refined', body
    assert body['n_members'] >= 4
    assert body['n_subclusters'] >= 2, body
    assert body['n_updated'] == body['n_members']

    refresh(opensearch, INDEXES['items'])
    subids = _subid_terms(opensearch, CANDIDATE_CLUSTER_ID)
    assert len(subids) >= 2, subids
    assert sum(subids.values()) == body['n_members']
    # Sub-ids are cluster-local and formatted "<cluster_id><letter>".
    assert all(key.startswith(str(CANDIDATE_CLUSTER_ID)) for key in subids), subids


def test_refine_is_idempotent(kb: Any, opensearch: Any) -> None:
    before = _subid_terms(opensearch, CANDIDATE_CLUSTER_ID)
    resp = kb.post(f'/clusters/refine/{CANDIDATE_CLUSTER_ID}')
    assert resp.status_code == 200, resp.text
    assert resp.json()['action'] == 'refined'
    refresh(opensearch, INDEXES['items'])
    assert _subid_terms(opensearch, CANDIDATE_CLUSTER_ID) == before


def test_refine_honours_the_member_floor(kb: Any) -> None:
    resp = kb.post(f'/clusters/refine/{NONEXISTENT_CLUSTER_ID}')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['action'] == 'skipped_too_small', body
    assert body['n_subclusters'] == 0


def test_cluster_cards_report_kind_purity_and_subclusters(kb: Any) -> None:
    resp = kb.get('/clusters', params={'per_cluster': 2, 'max_clusters': 100})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_id = {item['cluster_id']: item for item in body['items']}

    pure = by_id[PURE_CLUSTER_ID]
    assert pure['cluster_kind'] == 'class'
    assert pure['purity'] == 1.0
    assert pure['dominant_class_name'] == 'box'
    assert len(pure['representatives']) == 2

    mixed = by_id[MIXED_CLUSTER_ID]
    assert mixed['cluster_kind'] == 'class'
    assert 0.5 < mixed['purity'] < 0.85, mixed

    candidate = by_id[CANDIDATE_CLUSTER_ID]
    assert candidate['cluster_kind'] == 'candidate'
    assert candidate['is_unlabeled'] is True
    assert candidate['n_subclusters'] >= 2

    assert by_id[-1]['cluster_kind'] == 'unassigned'


# ---------------------------------------------------------------------------
# Auto-promote
# ---------------------------------------------------------------------------


def test_auto_promote_dry_run_writes_nothing(kb: Any, opensearch: Any) -> None:
    sample_ids = _v6_crop_ids(opensearch, PURE_CLUSTER_ID)
    assert sample_ids, 'precondition: the pure cluster must hold unvalidated v6 rows'
    before = {
        crop_id: get_doc(opensearch, INDEXES['items'], crop_id)['_seq_no'] for crop_id in sample_ids
    }

    resp = kb.post('/clusters/auto_promote', params={'dry_run': True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    promotable = {c['cluster_id']: c for c in body['clusters']}
    assert promotable[PURE_CLUSTER_ID]['promote'] is True
    assert promotable[MIXED_CLUSTER_ID]['promote'] is False

    for crop_id, seq_no in before.items():
        assert get_doc(opensearch, INDEXES['items'], crop_id)['_seq_no'] == seq_no


def test_auto_promote_apply_validates_only_the_high_purity_cluster(
    kb: Any, opensearch: Any
) -> None:
    promotable = _v6_crop_ids(opensearch, PURE_CLUSTER_ID, limit=10)
    untouched = _v6_crop_ids(opensearch, MIXED_CLUSTER_ID, limit=5)
    assert promotable, 'precondition: the pure cluster must hold promotable rows'
    assert untouched, 'precondition: the mixed cluster must hold unvalidated rows'

    resp = kb.post('/clusters/auto_promote')
    assert resp.status_code == 200, resp.text
    assert resp.json()['promoted'] >= len(promotable)

    refresh(opensearch, INDEXES['items'])
    for crop_id in promotable:
        src = get_doc(opensearch, INDEXES['items'], crop_id)['_source']
        assert src['class_validated'] is True, crop_id
        assert src['class_source'] == 'cluster_v6_majority_agreement'
        assert src['class_id_history'][-1]['writer'] == 'auto_promote'
    for crop_id in untouched:
        src = get_doc(opensearch, INDEXES['items'], crop_id)['_source']
        assert src['class_validated'] is False, crop_id
        assert src['class_source'] == 'v6_model'


# ---------------------------------------------------------------------------
# Region clustering + false-positive centroids
# ---------------------------------------------------------------------------


def _wait_for_job(get_state: Any, *, timeout: float = 300.0) -> dict[str, Any]:
    state = wait_for_state(get_state, lambda s: not s.get('running'), timeout=timeout)
    assert not state.get('running'), state
    return state


def test_region_clustering_partitions_the_region_pool(kb: Any, opensearch: Any) -> None:
    resp = kb.post('/regions/cluster', params={'auto_fp_threshold': 0.0})
    assert resp.status_code == 200, resp.text

    state = _wait_for_job(lambda: kb.get('/regions/cluster/status').json())
    assert state.get('error') is None, state
    result = state['result']
    assert result['status'] == 'success', result
    assert result['n_regions'] >= 32
    assert result['assigned'] == result['n_regions']

    refresh(opensearch, INDEXES['items'])
    buckets = search(
        opensearch,
        INDEXES['items'],
        {
            'size': 0,
            'query': {'exists': {'field': 'region_embedding'}},
            'aggs': {'rc': {'terms': {'field': 'region_cluster_id', 'size': 50}}},
        },
    )['aggregations']['rc']['buckets']
    assigned = {b['key']: b['doc_count'] for b in buckets}
    # The permanent false-positive bucket must survive a re-partition.
    assert assigned.get(FP_REGION_CLUSTER_ID, 0) == 12, assigned
    assert len([k for k in assigned if k >= 0]) >= 2, assigned


def test_region_cluster_cards_pin_the_false_positive_bucket_first(kb: Any) -> None:
    resp = kb.get('/regions/clusters')
    assert resp.status_code == 200, resp.text
    clusters = resp.json()['clusters']
    assert clusters, resp.text
    assert clusters[0]['id'] == FP_REGION_CLUSTER_ID
    assert clusters[0]['cluster_kind'] == 'false_positive'


def test_region_refine_writes_region_subids(kb: Any, opensearch: Any) -> None:
    buckets = search(
        opensearch,
        INDEXES['items'],
        {
            'size': 0,
            'query': {'range': {'region_cluster_id': {'gte': 0}}},
            'aggs': {'rc': {'terms': {'field': 'region_cluster_id', 'size': 50}}},
        },
    )['aggregations']['rc']['buckets']
    biggest = max(buckets, key=lambda b: b['doc_count'])
    if biggest['doc_count'] < 4:
        pytest.skip(f'no region bucket has the 4-member AHC floor: {buckets}')

    resp = kb.post(f'/regions/clusters/refine/{biggest["key"]}')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['action'] == 'refined', body
    assert body['n_updated'] == body['n_members']

    refresh(opensearch, INDEXES['items'])
    subids = _subid_terms(opensearch, biggest['key'], field='region_cluster_subid')
    assert sum(subids.values()) == body['n_members'], subids


def test_fp_centroid_build_then_suspected_false_positives(kb: Any) -> None:
    resp = kb.post('/regions/fp_centroids/build')
    assert resp.status_code == 200, resp.text
    state = _wait_for_job(lambda: kb.get('/regions/fp_centroids/status').json())
    assert state.get('error') is None, state
    result = state['result']
    assert result['status'] == 'success', result
    assert result['n_members'] >= 1, result

    suspected = kb.get('/regions/suspected_false_positives', params={'threshold': 0.35})
    assert suspected.status_code == 200, suspected.text
    body = suspected.json()
    assert body['centroids_built'] is True, body
    # The seeded FP group is deliberately far from the good-region groups,
    # so a tight threshold must NOT drag good regions into the FP cohort.
    assert body['total'] == 0, body


# ---------------------------------------------------------------------------
# Flag-gated overlay jobs (scores / diverse select / viz projection)
# ---------------------------------------------------------------------------


def _run_scorer(kb: Any, scorer: str) -> dict[str, Any]:
    resp = kb.post('/scores/compute', json={'scorers': [scorer]})
    assert resp.status_code == 200, resp.text
    state = wait_for_state(
        lambda: kb.get('/scores/status').json(),
        lambda s: s.get('status') != 'running',
        timeout=300.0,
    )
    assert state['status'] != 'running', 'the scoring job never left the running state'
    return state


def test_scores_compute_writes_the_overlay_fields(kb: Any, opensearch: Any) -> None:
    state = _run_scorer(kb, 'mistakenness')
    assert state['status'] == 'completed', state

    refresh(opensearch, INDEXES['items'])
    scored = opensearch.post(
        f'/{INDEXES["items"]}/_count',
        json={'query': {'exists': {'field': 'mistakenness_score'}}},
    ).json()['count']
    assert scored > 0, 'the mistakenness scorer wrote nothing'

    coverage = kb.get('/scores/coverage')
    coverage.raise_for_status()
    mistakenness = coverage.json()['coverage']['mistakenness']
    assert mistakenness['n_scored'] == scored, mistakenness
    assert mistakenness['pct'] > 0


def test_embedding_scorers_report_their_missing_prerequisite(kb: Any) -> None:
    """``uniqueness`` and ``near_dup`` need a trained IVF centroid store.

    The harness never runs the item-clustering pipeline, so this asserts
    the documented behaviour: the job fails loudly with an actionable
    message rather than silently writing nothing.
    """
    state = _run_scorer(kb, 'uniqueness')
    assert state['status'] == 'failed', state
    assert 'IVF centroid store' in (state['error'] or ''), state
    assert state['processed'] == 0, state


def test_scores_cancel_on_an_idle_job_is_a_no_op(kb: Any) -> None:
    resp = kb.post('/scores/cancel')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['cancelled'] is False, body
    assert body['status'] in ('completed', 'idle', 'cancelled', 'failed'), body


def test_unknown_scorer_is_rejected(kb: Any) -> None:
    resp = kb.post('/scores/compute', json={'scorers': ['not_a_scorer']})
    assert resp.status_code == 400, resp.text
    assert 'unknown scorer' in resp.text


def test_diverse_select_returns_an_ordering_and_mutates_nothing(kb: Any, opensearch: Any) -> None:
    before = opensearch.get(f'/{INDEXES["items"]}/_stats/indexing').json()
    before_ops = before['_all']['primaries']['indexing']['index_total']

    resp = kb.post(
        '/select/diverse',
        json={'k': 10, 'scope': {'cluster_id': CANDIDATE_CLUSTER_ID}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['method'] == 'kcenter_greedy'
    assert len(body['crop_ids']) == 10
    assert len(set(body['crop_ids'])) == 10
    assert body['n_pool'] >= 10

    after = opensearch.get(f'/{INDEXES["items"]}/_stats/indexing').json()
    assert after['_all']['primaries']['indexing']['index_total'] == before_ops


def test_viz_projection_rebuild_then_serve(kb: Any, opensearch: Any) -> None:
    resp = kb.post('/viz/projection/rebuild', params={'scope': 'residual'})
    assert resp.status_code == 202, resp.text

    state = wait_for_state(
        lambda: kb.get('/viz/projection/status').json(),
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

    served = kb.get('/viz/projection', params={'max_points': 1000})
    assert served.status_code == 200, served.text
    body = served.json()
    assert body.get('points'), body
    first = body['points'][0]
    assert {'crop_id', 'x', 'y'} <= set(first), first


def test_viz_state_index_honours_the_configured_index_prefix(opensearch: Any) -> None:
    """The UMAP run-metadata index names (viz-only and the retired
    clustering reducer's) are resolved through
    ``CurationConfig.umap_viz_state_index`` / ``umap_state_index``,
    each overridable via its own ``OP_*_INDEX`` env var — the harness sets
    both to ``verify_``-prefixed names, so no unscoped ``op_*`` index
    should ever appear.
    """
    names = [i['index'] for i in opensearch.get('/_cat/indices?format=json').json()]
    unscoped = [n for n in names if n.startswith('op_') and not n.startswith('verify_')]
    assert unscoped == [], unscoped
