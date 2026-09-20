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
    """Build an OpenSearch bucket like the one ``auto_promote_clusters`` consumes."""
    return {
        'key': cluster_id,
        'doc_count': members,
        'top_class': {
            'buckets': [{'key': name, 'doc_count': count} for name, count in classes],
        },
    }


def _search_response(buckets: list[dict[str, Any]]) -> dict[str, Any]:
    return {'aggregations': {'clusters': {'buckets': buckets}}}


def _make_client(search_response: dict[str, Any]) -> MagicMock:
    """Bare aggregation-only client for paths that never reach the write
    branch (dry_run, zero-label buckets, below min_members)."""
    client = MagicMock()
    client.search = AsyncMock(return_value=search_response)
    client.update_by_query = AsyncMock(return_value={'updated': 0})
    return client


class _FakeAutoPromoteClient:
    """Fake covering the full scroll + OCC-bulk write path.

    ``crop_ids_by_cluster`` maps ``cluster_id -> [doc_id, ...]`` — the ids
    :func:`auto_promote_clusters` should discover via its scroll-for-ids
    helper for that cluster's promote query. Every discovered id reads
    back as a plausible pre-write v6_model source on ``get`` and its
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
        must = body['query']['bool']['must']
        cluster_id = next(
            int(m['term']['cluster_id']) for m in must if 'cluster_id' in m.get('term', {})
        )
        ids = self._crop_ids_by_cluster.get(cluster_id, [])
        return {'_scroll_id': f'scroll-{cluster_id}', 'hits': {'hits': [{'_id': i} for i in ids]}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        source = {
            'class_id': 1,
            'class_name': 'cruiserbike',
            'class_source': 'v6_model',
            'class_validated': False,
            'test_holdout': False,
        }
        found = {d['_id']: source for d in body['docs']}
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
async def test_auto_promote_clusters_promotes_only_high_purity() -> None:
    """3 buckets: pure (promote) / mixed (skip) / unlabelled (skip).

    The only auto-promotion path is the ``cluster_v6_majority_agreement``
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
    assert written['class_source'] == 'cluster_v6_majority_agreement'
    assert written['label_source'] == 'cluster_v6_majority_agreement'
    # Every promoted crop now snapshots its pre-write v6_model state into
    # class_id_history.
    history = written['class_id_history']
    assert len(history) == 1
    assert history[0]['class_id'] == 1
    assert history[0]['class_source'] == 'v6_model'
    assert history[0]['writer'] == 'auto_promote'

    # The scroll-for-ids query targets the right cluster/class/holdout shape.
    scroll_init_call = next(c for c in client.search_calls if 'aggs' not in c)
    must = scroll_init_call['query']['bool']['must']
    must_not = scroll_init_call['query']['bool']['must_not']
    assert {'term': {'cluster_id': 1}} in must
    assert {'term': {'class_source': 'v6_model'}} in must
    assert {'term': {'class_name': 'cruiserbike'}} in must
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
    client = _make_client(_search_response(buckets))

    out = await auto_promote_clusters(client, min_purity=0.85, min_members=4, dry_run=True)

    # No writes in dry-run mode.
    assert client.update_by_query.await_count == 0
    assert out['dry_run'] is True
    # In dry-run, ``promoted`` is the *count of crops that would be relabelled*
    # (members - top_count) for promoted clusters. Cluster 1 is pure → 0.
    assert out['promoted'] == 0
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
    # Outer query is unconstrained — purity is computed across ALL labelled
    # crops (validated and unvalidated). The validated/v6 distinction is
    # enforced at update_by_query time, not at agg time, so a 99-validated
    # cluster doesn't get its purity computed off the lone unvalidated crop.
    assert 'query' not in body
    # Aggregation shape matches what the helper expects to consume.
    assert body['aggs']['clusters']['terms']['field'] == 'cluster_id'
    # ``class_name`` is mapped keyword directly on the live index — no
    # ``.keyword`` subfield.
    assert body['aggs']['clusters']['aggs']['top_class']['terms']['field'] == 'class_name'
