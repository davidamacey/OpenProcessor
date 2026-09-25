"""
Unit tests for `src.services.curation.clustering.auto_promote` /
`orchestrator` — focused on ``auto_promote_clusters`` and the small
helpers around it.

No real OpenSearch / FAISS instance is required. We mock the
``AsyncOpenSearch`` surface used by ``auto_promote_clusters``.

The reference line's audit-remediation pass converted the promotion
write from a bare ``update_by_query`` painless script to a per-doc OCC
bulk pass (scroll for doc ids -> ``occ_skip_on_conflict_bulk``) so
``class_id_history`` gets appended per promoted crop. The mock surface
below grew ``scroll``/``clear_scroll``/``mget``/``bulk`` to match;
``update_by_query`` is no longer called by this function at all.

A later pass rewrote ``occ_skip_on_conflict_bulk`` to page doc_ids
through batched ``_mget`` + ``_bulk`` instead of per-doc ``get``/
``update``, so the fake below implements ``mget``/``bulk`` rather than
``get``/``update``.

This is the "distinct file, same basename as the clustering package's
test_legacy_clustering.py" file from the reference tree's top-level
``tests/test_legacy_clustering.py`` — see
``tests/curation/test_clustering_orchestrator.py`` for the other one.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from src.services.curation.clustering import orchestrator as _orchestrator


# Import order matters here: orchestrator.py's bottom-of-file
# import of auto_promote.py only resolves cleanly if orchestrator is
# the FIRST of the two modules loaded in this process — a consequence
# of orchestrator.py being one of the ratchet-exempt oversize files
# documented in docs/design/curation_design_rationale.md §5. The assignment
# below (rather than a plain `from orchestrator import ITEMS_INDEX`)
# is deliberate: it's a real statement that breaks ruff/isort's import
# block so it can't silently re-alphabetize auto_promote's import back
# above orchestrator's, which reintroduces the ImportError this file
# exists to avoid.
ITEMS_INDEX = _orchestrator.ITEMS_INDEX

from src.services.curation.clustering.auto_promote import auto_promote_clusters  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================


def _bucket(
    cluster_id: int,
    *,
    members: int,
    classes: list[tuple[str, int]],
) -> dict[str, Any]:
    """Build an OpenSearch bucket like the one ``auto_promote_clusters`` consumes.

    F-29: cluster buckets now come from a ``composite`` agg (paged by
    cluster_id) rather than a single ``terms: size=10000`` agg — the
    composite bucket key is a dict of source-name -> value.
    """
    return {
        'key': {'cluster_id': cluster_id},
        'doc_count': members,
        'top_class': {
            'buckets': [{'key': name, 'doc_count': count} for name, count in classes],
        },
    }


def _search_response(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    return {'aggregations': {'clusters': {'buckets': buckets}}}


def _make_client(search_response: dict[str, Any], *, count: int = 0) -> MagicMock:
    """Bare aggregation-only client for paths that never reach the write
    branch (dry_run, zero-label buckets, below min_members)."""
    client = MagicMock()
    client.search = AsyncMock(return_value=search_response)
    client.update_by_query = AsyncMock(return_value={'updated': 0})
    # CM-2: dry-run now calls client.count(...) against the real
    # promote_query instead of computing members - top_count locally.
    client.count = AsyncMock(return_value={'count': count})
    return client


_PRE_WRITE_SOURCE: dict[str, Any] = {
    'class_id': 1,
    'class_name': 'cruiserbike',
    'class_source': 'classifier_model',
    'class_validated': False,
    'test_holdout': False,
}


class _FakeAutoPromoteClient:
    """Fake covering the full scroll + OCC-bulk write path.

    ``crop_ids_by_cluster`` maps ``cluster_id -> [doc_id, ...]`` — the ids
    :func:`auto_promote_clusters` should discover via its scroll-for-ids
    helper for that cluster's promote query. Every discovered id reads
    back as a plausible pre-write classifier_model source on ``get`` and its
    write lands in ``update_calls``.
    """

    def __init__(
        self,
        agg_response: dict[str, Any],
        crop_ids_by_cluster: dict[int, list[str]],
    ) -> None:
        self._agg_response = agg_response
        self._crop_ids_by_cluster = crop_ids_by_cluster
        self.search_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.update_by_query = AsyncMock(
            side_effect=AssertionError(
                'update_by_query should no longer be called by auto_promote_clusters'
            )
        )

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.search_calls.append(body)
        if 'aggs' in body:
            return self._agg_response
        filt = body['query']['bool']['filter']
        cluster_id = next(
            int(m['term']['cluster_id']) for m in filt if 'cluster_id' in m.get('term', {})
        )
        ids = self._crop_ids_by_cluster.get(cluster_id, [])
        # The promote scroll reads the class state the write re-checks.
        hits = [{'_id': i, '_source': dict(_PRE_WRITE_SOURCE)} for i in ids]
        return {'_scroll_id': f'scroll-{cluster_id}', 'hits': {'hits': hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        found = {d['_id']: dict(_PRE_WRITE_SOURCE) for d in body['docs']}
        return make_mget_response(found)

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            self.update_calls.append({'id': doc_id, 'doc': doc['doc']})
            items.append(make_bulk_update_item(doc_id, status=200))
        return make_bulk_response(items)


# =============================================================================
# auto_promote_clusters
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.usefixtures('reference_ingest_profiles')
async def test_auto_promote_clusters_promotes_only_high_purity() -> None:
    """3 buckets: pure (promote) / mixed (skip) / unlabelled (skip).

    The only auto-promotion path is the ``cluster_majority_agreement``
    write — crops where v6 already chose the cluster's dominant class
    get ``class_validated=True`` set. Unlabeled crops are left for the
    VLM + the review queue. This avoids the prototype-era pattern of
    silently labelling crops by cluster proximity, which was shown to
    poison the dataset on the reference line.
    """
    buckets = [
        # Cluster 1: purity = 10/10 = 1.0, members 10 → promote.
        _bucket(1, members=10, classes=[('cruiserbike', 10)]),
        # Cluster 2: purity = 5/10 = 0.5, members 10 → skip.
        _bucket(2, members=10, classes=[('sportycar', 5), ('pickup', 5)]),
        # Cluster 3: 8 members, zero labelled → skip.
        _bucket(3, members=8, classes=[]),
    ]
    crop_ids = [f'crop-{i}' for i in range(10)]
    client = _FakeAutoPromoteClient(_search_response(buckets), {1: crop_ids})

    out = await auto_promote_clusters(client, min_purity=0.85, min_members=4)

    assert out['status'] == 'success'
    assert out['min_purity'] == 0.85
    assert out['min_members'] == 4
    assert out['dry_run'] is False
    # All 10 discovered crops in the promoted cluster get written.
    assert out['promoted'] == 10
    # Cluster 2 (10) + cluster 3 (8) skipped.
    assert out['skipped'] == 18

    # update_by_query is no longer used for the write.
    client.update_by_query.assert_not_called()
    assert len(client.update_calls) == 10
    written = client.update_calls[0]['doc']
    assert written['class_validated'] is True
    assert written['class_source'] == 'cluster_majority_agreement'
    assert written['label_source'] == 'cluster_majority_agreement'
    # Every promoted crop now snapshots its pre-write classifier_model state into
    # class_id_history.
    history = written['class_id_history']
    assert len(history) == 1
    assert history[0]['class_id'] == 1
    assert history[0]['class_source'] == 'classifier_model'
    assert history[0]['writer'] == 'auto_promote'

    # The scroll-for-ids query targets the right cluster/class/holdout shape.
    scroll_init_call = next(c for c in client.search_calls if 'aggs' not in c)
    filt = scroll_init_call['query']['bool']['filter']
    must_not = scroll_init_call['query']['bool']['must_not']
    assert {'term': {'cluster_id': 1}} in filt
    assert {'terms': {'class_source': ['classifier_model']}} in filt
    assert {'term': {'class_name': 'cruiserbike'}} in filt
    assert {'term': {'class_validated': True}} in must_not
    assert {'term': {'test_holdout': True}} in must_not

    # Per-cluster summary fields are present.
    summaries = {s['cluster_id']: s for s in out['clusters']}
    assert set(summaries.keys()) == {1, 2, 3}
    assert summaries[1]['promote'] is True
    assert summaries[1]['top_class'] == 'cruiserbike'
    assert summaries[1]['top_count'] == 10
    assert summaries[1]['purity'] == 1.0
    assert summaries[2]['promote'] is False
    assert summaries[2]['purity'] == 0.5
    assert summaries[3]['promote'] is False


@pytest.mark.asyncio
async def test_auto_promote_clusters_dry_run_does_not_call_update() -> None:
    buckets = [
        _bucket(1, members=10, classes=[('cruiserbike', 10)]),
        _bucket(2, members=10, classes=[('sportycar', 5), ('pickup', 5)]),
    ]
    # CM-2: dry-run's count must come from an actual client.count(...) call
    # against the real promote_query, not a locally-computed guess.
    client = _make_client(_search_response(buckets), count=7)

    out = await auto_promote_clusters(client, min_purity=0.85, min_members=4, dry_run=True)

    # No writes in dry-run mode.
    assert client.update_by_query.await_count == 0
    assert out['dry_run'] is True
    # Cluster 1 (pure, promoted) contributes whatever client.count(...)
    # reports for its promote_query — no longer members - top_count.
    client.count.assert_awaited_once()
    assert client.count.await_args.kwargs['index'] == ITEMS_INDEX
    assert out['promoted'] == 7
    # Cluster 2 is the only skip.
    assert out['skipped'] == 10


@pytest.mark.asyncio
async def test_auto_promote_clusters_skips_zero_label_buckets() -> None:
    """A cluster with no class buckets at all should be counted as skipped."""
    buckets = [
        _bucket(99, members=8, classes=[]),
    ]
    client = _make_client(_search_response(buckets))

    out = await auto_promote_clusters(client)

    assert out['promoted'] == 0
    assert out['skipped'] == 8
    # Zero-label clusters are recorded for operator visibility, but not
    # promoted (promote=False, top_class=None, members preserved).
    assert len(out['clusters']) == 1
    only = out['clusters'][0]
    assert only['cluster_id'] == 99
    assert only['promote'] is False
    assert only['top_class'] is None
    assert only['members'] == 8
    assert client.update_by_query.await_count == 0


@pytest.mark.asyncio
async def test_auto_promote_clusters_respects_min_members() -> None:
    """A pure cluster with only 3 members fails the ``min_members=4`` gate."""
    buckets = [
        _bucket(7, members=3, classes=[('cruiserbike', 3)]),
    ]
    client = _make_client(_search_response(buckets))
    out = await auto_promote_clusters(client, min_purity=0.85, min_members=4)
    assert out['promoted'] == 0
    assert out['skipped'] == 3
    assert client.update_by_query.await_count == 0
    assert out['clusters'][0]['promote'] is False


@pytest.mark.asyncio
async def test_auto_promote_clusters_search_targets_correct_index() -> None:
    client = _make_client(_search_response([]))
    await auto_promote_clusters(client)
    client.search.assert_awaited_once()
    assert client.search.await_args.kwargs['index'] == ITEMS_INDEX
    body = client.search.await_args.kwargs['body']
    # CM-1: the outer query restricts the aggregation to candidate clusters
    # (cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET). Class clusters have
    # cluster_id == class_id by construction, so their purity is always
    # 1.0 and every member trivially "agrees" -- a self-referential signal,
    # not an independent one. Purity is still computed across ALL labelled
    # crops in a candidate cluster (validated and unvalidated); the
    # validated/v6 distinction is enforced at write time, not at agg time.
    from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET

    assert body['query']['bool']['filter'] == [
        {'range': {'cluster_id': {'gte': RESIDUAL_CLUSTER_ID_OFFSET}}}
    ]
    # CM-2: excluded items never contribute to a cluster's purity call.
    assert {'term': {'class_excluded': True}} in body['query']['bool']['must_not']
    # Aggregation shape matches what the helper expects to consume (F-29:
    # composite agg paged by cluster_id, not a single terms:size=10000).
    sources = body['aggs']['clusters']['composite']['sources']
    assert sources == [{'cluster_id': {'terms': {'field': 'cluster_id'}}}]
    # ``class_name`` is mapped keyword directly on the live index — no
    # ``.keyword`` subfield.
    assert body['aggs']['clusters']['aggs']['top_class']['terms']['field'] == 'class_name'


# =============================================================================
# CM-1: class clusters must never be auto-promote targets.
# =============================================================================


class _FilteringAutoPromoteClient(_FakeAutoPromoteClient):
    """Like :class:`_FakeAutoPromoteClient`, but its `search` honors the
    outer `query.bool.filter` range on `cluster_id` for the aggregation
    call -- close enough to real OpenSearch behavior to prove the
    CM-1 range filter actually excludes class-range buckets, not just
    that the query body contains the right clause (already covered by
    ``test_auto_promote_clusters_search_targets_correct_index``).
    """

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.search_calls.append(body)
        if 'aggs' in body:
            min_cluster_id = body['query']['bool']['filter'][0]['range']['cluster_id']['gte']
            buckets = self._agg_response['aggregations']['clusters']['buckets']
            kept = [b for b in buckets if int(b['key']['cluster_id']) >= min_cluster_id]
            return {'aggregations': {'clusters': {'buckets': kept}}}
        # F-19: pure predicates live in bool.filter now (bool.must before).
        clauses = body['query']['bool'].get('filter', []) + body['query']['bool'].get('must', [])
        cluster_id = next(
            int(m['term']['cluster_id']) for m in clauses if 'cluster_id' in m.get('term', {})
        )
        ids = self._crop_ids_by_cluster.get(cluster_id, [])
        # The promote scroll reads the class state the write re-checks.
        hits = [{'_id': i, '_source': dict(_PRE_WRITE_SOURCE)} for i in ids]
        return {'_scroll_id': f'scroll-{cluster_id}', 'hits': {'hits': hits}}


@pytest.mark.asyncio
@pytest.mark.usefixtures('reference_ingest_profiles')
async def test_auto_promote_clusters_never_promotes_a_class_cluster() -> None:
    """A class cluster (cluster_id == class_id, always purity 1.0 by
    construction) with 10 classifier-labeled members must produce 0
    promotions -- the exact circularity CM-1 fixes. A candidate cluster
    (cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET) with the same shape still
    promotes normally.
    """
    from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET

    class_cluster_id = 7  # cluster_id == class_id, well under the offset
    candidate_cluster_id = RESIDUAL_CLUSTER_ID_OFFSET + 3
    buckets = [
        _bucket(class_cluster_id, members=10, classes=[('cruiserbike', 10)]),
        _bucket(candidate_cluster_id, members=10, classes=[('cruiserbike', 10)]),
    ]
    crop_ids = [f'crop-{i}' for i in range(10)]
    client = _FilteringAutoPromoteClient(
        _search_response(buckets),
        {class_cluster_id: crop_ids, candidate_cluster_id: crop_ids},
    )

    out = await auto_promote_clusters(client, min_purity=0.85, min_members=4)

    seen_cluster_ids = {s['cluster_id'] for s in out['clusters']}
    assert class_cluster_id not in seen_cluster_ids
    assert candidate_cluster_id in seen_cluster_ids
    # Only the candidate cluster's 10 crops were written.
    assert out['promoted'] == 10
    written_ids = {c['id'] for c in client.update_calls}
    assert written_ids == set(crop_ids)
