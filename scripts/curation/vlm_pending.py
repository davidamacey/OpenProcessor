"""Which crops still need the VLM: the pending query, the id fetch the producer
runs per project, and the freshness filter over what it fetched."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.domain.embedding_state import embedded_clause
from src.services.curation.vlm_policy import VlmPolicy
from src.services.curation.vlm_scope import ScopeCache, daily_budget_remaining, vlm_scope_clauses


if TYPE_CHECKING:
    from collections.abc import Collection


_classifier_sources_empty_warned = False


def _warn_classifier_sources_empty_once() -> None:
    """Log once (not every poll) that the 'classifier already confident'
    must_not guard is inactive because classifier_class_sources() is
    empty in this environment."""
    global _classifier_sources_empty_warned  # noqa: PLW0603 - warn-once flag
    if not _classifier_sources_empty_warned:
        _classifier_sources_empty_warned = True
        print(
            '[vlm-worker] classifier_class_sources() is empty; the '
            "'classifier already confident' skip guard is inactive -- "
            'every crop is a VLM candidate regardless of classifier confidence.'
        )


def _build_pending_query(
    classifier_skip_conf: float,
    exclude_ids: list[str] | None = None,
    *,
    policy: VlmPolicy | None = None,
    representative_ids: Collection[str] | None = None,
) -> dict:
    """Crops that need the VLM right now.

    Mirrors the ``must_not`` clauses in pipeline_auto_label so the same
    crops the on-demand pipeline would process are picked up by the worker.
    ``policy`` (default: scope ``all``) narrows them through the same scope
    function the sweep uses.

    ``exclude_ids`` pushes the producer's in-flight set into the
    query server-side (``must_not: {ids: ...}``) instead of over-fetching
    ``batch_size + len(in_flight)`` docs and filtering in-flight ids out
    in Python.
    """
    # Lazy: keeps the module import light; src.config is all this pulls in.
    from src.config.vlm_class_attempt import VLM_CLASS_ATTEMPTED_AT_FIELD
    from src.domain.ingest_class_sources import (
        OPEN_VOCAB_TARGET_CLASS_SOURCE,
        classifier_class_sources,
    )
    from src.services.curation.vlm_class_attempt import recent_empty_answer_clause

    must_not: list[dict] = [
        {'term': {'class_validated': True}},
        # Asked recently and the answer had no class: not again until the
        # retry window passes (the class fields are untouched, so nothing
        # else keeps the item out of this query).
        recent_empty_answer_clause(),
    ]
    classifier_sources = sorted(classifier_class_sources())
    if classifier_sources:
        # classifier already confident
        must_not.append(
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': classifier_sources}},
                        {'range': {'confidence': {'gte': classifier_skip_conf}}},
                    ],
                },
            },
        )
    else:
        # An empty terms clause matches nothing (correct as a
        # must_not exclusion) but is dead weight in the query shape --
        # only emit it when there's something to exclude, and log once
        # so operators know this guard rail is inactive in this env.
        _warn_classifier_sources_empty_once()
    # One `terms` clause instead of 5 separate `term` clauses on the
    # same field — same match semantics, one less clause for OS to eval.
    must_not.append(
        {
            'terms': {
                'class_source': [
                    # prototype + ensemble_proto_rescue +
                    # ensemble_consensus class_source values are gone.
                    # Surviving auto-validation path is
                    # 'classifier_vlm_agreement' (A-PR2 ensemble writer;
                    # this query excludes already-labeled rows).
                    # VLM already labeled successfully:
                    'vlm',
                    'classifier_vlm_agreement',
                    'cluster_majority_agreement',
                    # VLM already failed once — won't help to retry:
                    'vlm_unmatched',
                    'vlm_new_class_pending',
                ],
            },
        },
    )
    # A user-named open-vocabulary class is never relabeled; the VLM is asked
    # once so its answer can ride along as a suggestion.
    must_not.append(
        {
            'bool': {
                'filter': [
                    {'terms': {'class_source': [OPEN_VOCAB_TARGET_CLASS_SOURCE]}},
                    {'exists': {'field': VLM_CLASS_ATTEMPTED_AT_FIELD}},
                ],
            },
        },
    )
    if exclude_ids:
        must_not.append({'ids': {'values': exclude_ids}})
    scope = vlm_scope_clauses(policy or VlmPolicy(), representative_ids=representative_ids)
    must_not.extend(scope.must_not)
    return {
        'bool': {
            'filter': [embedded_clause(), *scope.filter],
            'must_not': must_not,
        },
    }


def _filter_fresh_ids[K](
    ids: list[K],
    *,
    in_flight: set[K],
    released_at: dict[K, float],
    fetch_started: float,
) -> list[K]:
    """Ids a producer may safely dispatch: not currently in flight, and not
    released at or after ``fetch_started``.

    A fetch that started before (or at the same moment as) a consumer's
    release may still observe pre-write state, since the write uses
    ``refresh=False``. Only an id released strictly *before* this fetch
    began is guaranteed fresh.
    """
    return [
        i for i in ids if i not in in_flight and released_at.get(i, float('-inf')) < fetch_started
    ]


async def fetch_pending_ids(
    opensearch: Any,
    *,
    slug: str,
    scope_cache: ScopeCache,
    batch_size: int,
    classifier_skip_conf: float,
    exclude_ids: list[str] | None = None,
) -> list[str]:
    """Pull up to ``batch_size`` crop IDs of the BOUND project (``slug``) that need
    the VLM, under its VLM scope policy: none when the scope is ``off`` or today's
    budget is spent, and at most what is left of the budget.

    ``_source: False`` returns ids only. (Not ``stored_fields: '_none_'``:
    OpenSearch drops the ``_id`` metadata field with it too.)
    ``track_total_hits: False`` skips the exact-count pass this producer
    never reads. ``exclude_ids`` (the caller's in-flight set) is pushed
    into the query itself instead of being filtered out in Python after
    over-fetching ``batch_size + len(in_flight)`` docs.
    """
    from src.config.curation import items_index

    policy, reps = await scope_cache.get(opensearch, slug)
    remaining = await daily_budget_remaining(opensearch, policy)
    limit = batch_size if remaining is None else min(batch_size, remaining)
    if policy.scope == 'off' or limit <= 0:
        return []
    body = {
        'size': limit,
        '_source': False,
        'track_total_hits': False,
        'query': _build_pending_query(
            classifier_skip_conf,
            exclude_ids=exclude_ids,
            policy=policy,
            representative_ids=reps,
        ),
        # Oldest pending first — fairness across crops added across the
        # run; also avoids head-of-line starvation when new crops keep
        # arriving from ingest. crop_id tiebreaker keeps paging stable
        # for same-timestamp crops.
        'sort': [{'created_at': {'order': 'asc', 'unmapped_type': 'date'}}, {'crop_id': 'asc'}],
    }
    resp = await opensearch.search(index=items_index(), body=body)
    return [h['_id'] for h in resp.get('hits', {}).get('hits', [])]
