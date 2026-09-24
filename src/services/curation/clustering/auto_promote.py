"""Auto-promote stage for the curation auto-label pipeline.

Extracted from the orchestrator module alongside the disable-by-default
change in the ingest pipeline. Disabled because the v6-confidence-
floor-less rule contaminated class clusters with visually wrong crops
via ``class_source='cluster_majority_agreement'``. Kept here as an
opt-in path so the eventual confidence-gated rewrite has a home.

A crop is promoted only when v6's class call already matches the
cluster's dominant class (supervised classifier + sibling-embedding
majority agree). Earlier behaviour propagated the dominant label onto
every member of a high-purity cluster (including crops v6 never saw),
which silently poisoned a large historical cohort of rows. A
backfill demoted that cohort; this function no longer creates that
pollution shape going forward, but the unsolved problem — promoting
low-confidence v6 predictions — is why the pipeline now defaults to
skipping this stage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.occ import occ_skip_on_conflict_bulk
from src.core.logging import get_logger
from src.services.curation.class_write_guard import CLASS_GUARD_SOURCE_FIELDS, ClassWriteGuard
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET
from src.services.curation.cluster_purity import (
    PROMOTE_MIN_MEMBERS,
    PROMOTE_MIN_PURITY,
    is_promotable,
)

# Import order matters here — see orchestrator.py's bottom-of-file import
# and plan §7 R11. orchestrator.py imports auto_promote_clusters from this
# module at the bottom of its file, forming an intentional, preserved
# circular import: importing this module first (in isolation) fails, but
# the app always imports orchestrator first, so this resolves fine in
# practice. Do not "fix" this cycle.
from src.services.curation.clustering.orchestrator import ITEMS_INDEX
from src.services.curation.history import record_class_history
from src.services.curation.ingest_class_sources import (
    CLUSTER_MAJORITY_CLASS_SOURCE,
    classifier_class_sources,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)

_SCROLL_PAGE = 500


async def _scroll_hits(
    client: AsyncOpenSearch,
    *,
    index: str,
    query: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Every doc matching ``query`` -> its class-state ``_source``
    (:data:`CLASS_GUARD_SOURCE_FIELDS`), scrolled in pages. The merger
    promotes a doc only if that state is unchanged at write time.

    Phase 3 (b): replaces the direct-target scroll a painless
    ``update_by_query`` script would otherwise need — we need doc ids so
    each write can go through the OCC bulk merger below (which is how
    ``class_id_history`` gets appended per-doc; a painless script would
    have to reimplement the dedupe + cap logic in-cluster, which is the
    riskier of the two options the plan calls out).
    """
    found: dict[str, dict[str, Any]] = {}
    body = {'size': _SCROLL_PAGE, 'query': query, '_source': list(CLASS_GUARD_SOURCE_FIELDS)}
    resp = await client.search(index=index, body=body, scroll='2m')
    scroll_id = resp.get('_scroll_id')
    hits = resp['hits']['hits']
    while hits:
        found.update((h['_id'], h.get('_source') or {}) for h in hits)
        resp = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp.get('_scroll_id')
        hits = resp['hits']['hits']
    if scroll_id:
        try:
            await client.clear_scroll(scroll_id=scroll_id)
        except Exception as exc:  # nosec B110 — advisory cleanup only
            logger.info('legacy_auto_promote_clear_scroll_failed', error=str(exc))
    return found


async def auto_promote_clusters(
    client: AsyncOpenSearch,
    *,
    min_purity: float = PROMOTE_MIN_PURITY,
    min_members: int = PROMOTE_MIN_MEMBERS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Auto-validate crops in high-purity clusters where v6 agrees.

    Returns a summary keyed by ``promoted``, ``skipped``, ``clusters``.
    """
    # Aggregation: per-cluster top class. Purity is computed across ALL
    # labelled members (validated + unvalidated) so a cluster with 99
    # v6 honda + 1 unvalidated cruiserbike isn't deemed 100% cruiserbike.
    #
    # CM-1: restrict to candidate clusters (cluster_id >= the residual
    # offset). Class clusters (0..RESIDUAL_CLUSTER_ID_OFFSET-1) have
    # cluster_id == class_id by construction, so their purity is always
    # 1.0 -- every member "agrees" with the cluster because the cluster
    # IS the class. Without this filter, any classifier label with at
    # least min_members siblings gets stamped class_validated=true from
    # nothing but the classifier's own earlier output: a circular
    # self-validation, not an independent signal.
    #
    # CM-2: exclude class_excluded items from the aggregation too, so an
    # excluded item's class can't skew a cluster's purity/top-class call
    # for the *other* members that do get promoted.
    body = {
        'size': 0,
        'query': {
            'bool': {
                'filter': [{'range': {'cluster_id': {'gte': RESIDUAL_CLUSTER_ID_OFFSET}}}],
                'must_not': [{'term': {'class_excluded': True}}],
            },
        },
        'aggs': {
            'clusters': {
                'terms': {'field': 'cluster_id', 'size': 10000},
                'aggs': {
                    'top_class': {
                        # class_name is mapped keyword directly on the live
                        # index — no .keyword subfield exists. See
                        # legacy_clusters.py's top_class agg for the full story.
                        'terms': {
                            'field': 'class_name',
                            'size': 5,
                            'order': {'_count': 'desc'},
                        },
                    },
                },
            },
        },
    }
    resp = await client.search(index=ITEMS_INDEX, body=body)

    summaries: list[dict[str, Any]] = []
    total_promoted = 0
    total_skipped = 0
    now = datetime.now(UTC).isoformat()

    for bucket in resp.get('aggregations', {}).get('clusters', {}).get('buckets', []):
        cluster_id = int(bucket['key'])
        members = int(bucket['doc_count'])
        cls_buckets = bucket.get('top_class', {}).get('buckets', [])
        if not cls_buckets:
            total_skipped += members
            summaries.append(
                {
                    'cluster_id': cluster_id,
                    'members': members,
                    'labelled_total': 0,
                    'top_class': None,
                    'top_count': 0,
                    'purity': 0.0,
                    'promote': False,
                }
            )
            continue
        top_name = cls_buckets[0]['key']
        top_count = int(cls_buckets[0]['doc_count'])
        # Denominator: every labelled member (``terms`` skips nulls). The
        # agg only returns the top buckets; ``sum_other_doc_count`` holds
        # the members of every other class.
        labelled_total = sum(int(b['doc_count']) for b in cls_buckets) + int(
            bucket.get('top_class', {}).get('sum_other_doc_count') or 0
        )
        purity = top_count / labelled_total if labelled_total else 0.0

        promote = is_promotable(
            members=members,
            labelled=labelled_total,
            purity=purity,
            min_purity=min_purity,
            min_members=min_members,
        )

        summaries.append(
            {
                'cluster_id': cluster_id,
                'members': members,
                'labelled_total': labelled_total,
                'top_class': top_name,
                'top_count': top_count,
                'purity': round(purity, 4),
                'promote': promote,
            }
        )
        if not promote:
            total_skipped += members
            continue

        # WARNING: this rule has no v6-confidence floor; even v6 @ 61%
        # passes if its prediction matches the cluster majority. That's
        # why the pipeline defaults to skipping this stage. A
        # confidence-gated rewrite is the prerequisite to enabling
        # ``run_auto_promote=true`` in production.
        promote_query = {
            'bool': {
                'must': [
                    {'term': {'cluster_id': cluster_id}},
                    {'terms': {'class_source': sorted(classifier_class_sources())}},
                    {'term': {'class_name': top_name}},
                ],
                'must_not': [
                    {'term': {'class_validated': True}},
                    # P0-3: never auto-promote a frozen test_holdout
                    # crop's class fields — this writer is class-only,
                    # so an unconditional exclusion is correct here.
                    {'term': {'test_holdout': True}},
                    # CM-2: never auto-promote an excluded item's class.
                    {'term': {'class_excluded': True}},
                ],
            },
        }

        if dry_run:
            # CM-2: `members - top_count` counted every non-majority
            # member of the cluster, not the set this query actually
            # touches (which is also gated on class_source and
            # class_validated=false). Count the real query instead.
            count_resp = await client.count(index=ITEMS_INDEX, body={'query': promote_query})
            total_promoted += int(count_resp.get('count', 0))
            continue

        try:
            read = await _scroll_hits(client, index=ITEMS_INDEX, query=promote_query)
        except Exception as exc:
            logger.warning('legacy_auto_promote_cluster_failed', cluster_id=cluster_id, error=str(exc))
            total_skipped += members
            continue
        if not read:
            continue
        guard = ClassWriteGuard('auto_promote')
        for doc_id, source in read.items():
            guard.remember(doc_id, source)

        def _merge_promote(
            doc_id: str, current: dict[str, Any], _guard: ClassWriteGuard = guard
        ) -> dict[str, Any]:
            # Phase 3 (b): this used to be a bare update_by_query painless
            # script with no class_id_history append. Converting to a
            # per-doc OCC bulk pass (same shape as legacy_gemma.py's
            # gemma_label_batch) both lets us reuse record_class_history
            # (so dedupe (c) and the class_validated cap exemption (d)
            # apply uniformly) and re-checks the human/holdout guards
            # against the freshest doc state at write time, not just at
            # scroll time.
            if current.get('test_holdout'):
                return {}
            # Promote only the class state the cluster vote was taken on: a
            # human write, validation or exclusion since the scroll read
            # (CM-2) — even an undo back to a classifier label — wins.
            if not _guard.allows(doc_id, current):
                return {}
            update: dict[str, Any] = {
                'class_validated': True,
                'class_source': CLUSTER_MAJORITY_CLASS_SOURCE,
                'label_source': CLUSTER_MAJORITY_CLASS_SOURCE,
                'updated_at': now,
            }
            update['class_id_history'] = record_class_history(current, writer='auto_promote')
            return update

        try:
            result = await occ_skip_on_conflict_bulk(
                client,
                doc_ids=list(read),
                merger=_merge_promote,
                index=ITEMS_INDEX,
                refresh=True,
                writer_id='auto_promote',
            )
        except Exception as exc:
            logger.warning('legacy_auto_promote_cluster_failed', cluster_id=cluster_id, error=str(exc))
            total_skipped += members
            continue
        total_promoted += int(result.get('updated', 0))
        total_skipped += int(result.get('skipped_due_to_conflict', 0))
        if result.get('errors'):
            logger.warning(
                'legacy_auto_promote_bulk_partial_errors',
                cluster_id=cluster_id,
                errors=len(result['errors']),
            )

    return {
        'status': 'success',
        'min_purity': min_purity,
        'min_members': min_members,
        'dry_run': dry_run,
        'promoted': total_promoted,
        'skipped': total_skipped,
        'clusters': summaries,
    }


__all__ = ['auto_promote_clusters']
