"""The cluster list: one aggregation to cards, then server-side sort and filter.

``GET {P}/clusters`` serves every number a reviewer ranks clusters by, so the
client renders and never recomputes. The aggregation is a single request (no
per-card query); the sort and the card-level filters run over the aggregated
cards, and the response says when the aggregation was cut at ``max_clusters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from src.config.curation_accessors import items_index
from src.core.exceptions import RequestRejectedError
from src.domain.cluster_flags import FLAG_VOCABULARY, ValidationState, card_flags, validation_state
from src.services.curation import review_queries
from src.services.curation.cluster_ids import (
    CORE_SIMILARITY_MIN,
    RESIDUAL_CLUSTER_ID_OFFSET,
    cluster_kind,
)
from src.services.curation.cluster_purity import is_promotable, purity_thresholds, purity_tier
from src.services.curation.cluster_representatives import representatives_msearch_body
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.wire import current_cluster_distance
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.schemas.curation.clusters import (
        ClusterCard,
        ClusterListResponse,
        ClusterRepresentative,
        PurityThresholds,
    )


CANDIDATE_DOMINANT_MIN_COUNT = 3
"""A candidate cluster names a dominant class only when at least this many
members carry it."""

CANDIDATE_DOMINANT_MIN_SHARE = 0.5
"""...and that class holds at least this share of the labelled members."""

MAX_SUBCLUSTER_IDS = 200
"""The most sub-cluster ids one card lists (``n_subclusters`` stays exact)."""

PURITY_BASIS = 'nearest_centroid'
"""What a card's ``purity`` measures (served per card): see ``FITS``."""

CLUSTER_REVIEW_TABS: tuple[str, ...] = (
    'all',
    'detector_disagreements',
    'outliers',
    'vlm_low_conf',
    'mismatches',
)
"""Review tabs whose per-cluster member count a card serves. Each is a
``review_tab_clauses.SIMPLE_TABS`` id, so a count is the same query ``/review`` runs."""

MEASURED_SCRIPT = "doc['cluster_distance_cluster_id'].value == doc['cluster_id'].value"
FITS_SCRIPT = "doc['cluster_nearest_id'].value == doc['cluster_id'].value"

ClusterSortId = Literal[
    'needs_review_desc',
    'size_desc',
    'size_asc',
    'label_agreement_asc',
    'purity_asc',
    'purity_desc',
    'detector_conf_asc',
    'unvalidated_desc',
    'updated_desc',
    'dominant_class',
]
ClusterKindFilter = Literal['class', 'candidate', 'all']
ValidationFilter = Literal['none', 'partial', 'complete']


def _script(source: str) -> dict[str, Any]:
    return {'script': {'script': {'lang': 'painless', 'source': source}}}


# Measured by the geometry pass against the item's *current* cluster
# (an item that moved since carries a reference to its old cluster).
_MEASURED: dict[str, Any] = {
    'bool': {
        'filter': [
            {'exists': {'field': 'cluster_nearest_id'}},
            {'exists': {'field': 'cluster_distance_cluster_id'}},
            _script(MEASURED_SCRIPT),
        ]
    }
}
# ...and its nearest cluster centroid is its own cluster's.
_FITS: dict[str, Any] = _script(FITS_SCRIPT)


@dataclass(frozen=True)
class ClusterListParams:
    item_filter: ItemFilter = field(default_factory=ItemFilter)
    per_cluster: int = 4
    max_clusters: int = 1000
    kind: ClusterKindFilter = 'all'
    cluster_id: int | None = None
    offset: int = 0
    limit: int = 50
    sort: ClusterSortId = 'size_desc'
    dominant_class: tuple[str, ...] = ()
    detector_class: tuple[str, ...] = ()
    min_size: int | None = None
    max_size: int | None = None
    validation_state: ValidationFilter | None = None
    flags: tuple[str, ...] = ()
    needs_review: bool | None = None


def _candidate_dominant_name(cls_buckets: list[dict[str, Any]], labelled: int) -> str | None:
    """Dominant class name for a candidate cluster card, or ``None``.

    A candidate cluster is an unnamed residual grouping; naming it after
    the plurality class of a handful of labelled members (e.g. a 1-1-1
    tie in a 116-member cluster) overclaims. Require a unique winner that
    clears both floors.
    """
    if not cls_buckets or labelled <= 0:
        return None
    top = int(cls_buckets[0]['doc_count'])
    runner_up = int(cls_buckets[1]['doc_count']) if len(cls_buckets) > 1 else 0
    if (
        top >= CANDIDATE_DOMINANT_MIN_COUNT
        and top / labelled >= CANDIDATE_DOMINANT_MIN_SHARE
        and top > runner_up
    ):
        return str(cls_buckets[0]['key'])
    return None


def review_aggs() -> dict[str, Any]:
    """One ``filters`` agg holding a filter per tab in :data:`CLUSTER_REVIEW_TABS`,
    built by the same clause builder ``GET /review/{tab}`` uses (holdout hidden,
    as there)."""
    named: dict[str, Any] = {}
    for tab in CLUSTER_REVIEW_TABS:
        must, must_not, _reason = review_queries.build_tab_query(
            tab, include_test=False, text=None, max_rank=None
        )
        named[tab] = {'bool': {'must': must, 'must_not': must_not}}
    return {'review': {'filters': {'filters': named}}}


def aggregation_body(params: ClusterListParams) -> dict[str, Any]:
    """The one search body: items grouped by ``cluster_id`` with every card input."""
    base_match: dict[str, Any] = (
        {'term': {'cluster_id': params.cluster_id}}
        if params.cluster_id is not None
        else {'match_all': {}}
    )
    gate_filter = item_filter_clauses(params.item_filter)
    # `kind` goes into the query as a bounded cluster_id range *before*
    # aggregating: candidate ids (often far more numerous than the ~80 class
    # ids) would otherwise outrank class buckets by count and crowd them out
    # of the truncated aggregation.
    if params.kind == 'class':
        gate_filter.append({'range': {'cluster_id': {'gte': 0, 'lt': RESIDUAL_CLUSTER_ID_OFFSET}}})
    elif params.kind == 'candidate':
        gate_filter.append({'range': {'cluster_id': {'gte': RESIDUAL_CLUSTER_ID_OFFSET}}})
    query = {
        'bool': {
            'filter': [base_match, *gate_filter],
            # Human-ignored items left their bucket for good; the frozen test
            # holdout is never part of a cluster the operator works.
            'must_not': [{'term': {'class_excluded': True}}, {'term': {'test_holdout': True}}],
        }
    }
    sub_aggs: dict[str, Any] = {
        'top_class': {'terms': {'field': 'class_name', 'size': 3, 'order': {'_count': 'desc'}}},
        # True labelled count: top_class only returns its top buckets, so
        # summing them undercounts clusters with many classes.
        'labelled': {'filter': {'exists': {'field': 'class_name'}}},
        'validated': {'filter': {'term': {'class_validated': True}}},
        'subclusters': {'cardinality': {'field': 'cluster_subid'}},
        'subcluster_ids': {
            'terms': {
                'field': 'cluster_subid',
                'size': MAX_SUBCLUSTER_IDS,
                'order': {'_key': 'asc'},
            }
        },
        # Purity: members measured against their current cluster by the
        # geometry pass, and those whose nearest centroid is it.
        'geometry_measured': {'filter': _MEASURED},
        'geometry_fits': {'filter': {'bool': {'filter': [_MEASURED, _FITS]}}},
        'latest_update': {'max': {'field': 'updated_at'}},
        'detector_top': {'terms': {'field': 'detector_class_name', 'size': 1}},
        'detector_conf': {'avg': {'field': 'detector_confidence'}},
        **review_aggs(),
    }
    terms = {
        'terms': {'field': 'cluster_id', 'size': params.max_clusters, 'order': {'_count': 'desc'}},
        'aggs': sub_aggs,
    }
    return {'size': 0, 'query': query, 'aggs': {'clusters': terms}}


def bucket_to_card(bucket: dict[str, Any]) -> ClusterCard:
    """One ``cluster_id`` bucket as a card (representatives are filled later)."""
    cid = int(bucket['key'])
    ck = cluster_kind(cid) or 'unassigned'
    size = int(bucket['doc_count'])
    cls_buckets = bucket.get('top_class', {}).get('buckets', [])
    labelled_total = sum(int(b['doc_count']) for b in cls_buckets)
    if 'labelled' in bucket:
        labelled_total = max(labelled_total, int(bucket['labelled'].get('doc_count') or 0))
    top_name: str | None = None
    top_count = 0
    if cls_buckets:
        top_name = cls_buckets[0]['key']
        top_count = int(cls_buckets[0]['doc_count'])
    label_agreement = (top_count / labelled_total) if labelled_total else None
    purity_n = int(bucket.get('geometry_measured', {}).get('doc_count') or 0)
    fits = int(bucket.get('geometry_fits', {}).get('doc_count') or 0)
    purity = (fits / purity_n) if purity_n else None
    if ck == 'candidate':
        top_name = _candidate_dominant_name(cls_buckets, labelled_total)
    validated_count = int(bucket.get('validated', {}).get('doc_count') or 0)
    promotable = ck == 'candidate' and is_promotable(
        members=size, labelled=labelled_total, purity=label_agreement
    )
    det_buckets = bucket.get('detector_top', {}).get('buckets', [])
    detector = bucket.get('detector_top', {})
    # Members with a detector class: the top bucket plus everything the size-1 terms agg left out.
    det_total = (
        int(det_buckets[0]['doc_count']) + int(detector.get('sum_other_doc_count') or 0)
        if det_buckets
        else 0
    )
    det_share = (
        (int(det_buckets[0]['doc_count']) / det_total) if det_buckets and det_total else None
    )
    review = bucket.get('review', {}).get('buckets', {})
    review_counts = {
        tab: int(review.get(tab, {}).get('doc_count') or 0) for tab in CLUSTER_REVIEW_TABS
    }
    state: ValidationState = validation_state(size, validated_count)
    tier = purity_tier(purity)
    return {
        'cluster_id': cid,
        'cluster_kind': ck,
        'size': size,
        'validated_count': validated_count,
        'labelled_count': labelled_total,
        # The classifier registry id when the cluster is a class cluster
        # (cluster_id == class_id by invariant); a candidate has none yet.
        'dominant_class_id': cid if ck == 'class' and top_name else None,
        'dominant_class_name': top_name,
        'dominant_count': top_count,
        'purity': purity,
        'purity_n': purity_n,
        'purity_basis': PURITY_BASIS,
        'purity_tier': tier,
        'label_agreement': label_agreement,
        'labelled_share': (labelled_total / size) if size else None,
        'promotable': promotable,
        'is_unlabeled': labelled_total == 0,
        'n_subclusters': int(bucket.get('subclusters', {}).get('value') or 0),
        # Sorted so '7a', '7aa', '7b' read in the order the tabs show; capped
        # at MAX_SUBCLUSTER_IDS (n_subclusters stays the true count).
        'subcluster_ids': sorted(
            str(b['key']) for b in bucket.get('subcluster_ids', {}).get('buckets', [])
        ),
        'updated_at': bucket.get('latest_update', {}).get('value_as_string'),
        'representatives': [],
        'detector_top_class': str(det_buckets[0]['key']) if det_buckets else None,
        'detector_top_share': det_share,
        'detector_conf_mean': bucket.get('detector_conf', {}).get('value'),
        'validation_state': state,
        'review_counts': review_counts,
        'needs_review_count': review_counts['all'],
        'flags': card_flags(
            size=size,
            detector_top_share=det_share,
            purity_tier=tier,
            label_agreement=label_agreement,
            labelled_count=labelled_total,
            promotable=promotable,
            needs_review_count=review_counts['all'],
            validation_state=state,
        ),
    }


# id -> (label, value of a card, descending). A card whose value is ``None``
# sorts after every card that has one; ties break on cluster_id ascending.
_SORTS: dict[str, tuple[str, Callable[[ClusterCard], Any], bool]] = {
    'needs_review_desc': ('Needs review', lambda c: c['needs_review_count'], True),
    'size_desc': ('Largest first', lambda c: c['size'], True),
    'size_asc': ('Smallest first', lambda c: c['size'], False),
    'label_agreement_asc': ('Least label agreement', lambda c: c['label_agreement'], False),
    'purity_asc': ('Lowest purity', lambda c: c['purity'], False),
    'purity_desc': ('Highest purity', lambda c: c['purity'], True),
    'detector_conf_asc': ('Lowest detector confidence', lambda c: c['detector_conf_mean'], False),
    'unvalidated_desc': ('Most unvalidated', lambda c: c['size'] - c['validated_count'], True),
    'updated_desc': ('Recently updated', lambda c: c['updated_at'], True),
    'dominant_class': (
        'Dominant class',
        lambda c: (c['dominant_class_name'] or '').casefold() or None,
        False,
    ),
}

SORT_OPTIONS: tuple[tuple[str, str], ...] = tuple((k, v[0]) for k, v in _SORTS.items())


def sort_cards(cards: list[ClusterCard], sort: str) -> list[ClusterCard]:
    _label, value, descending = _SORTS[sort]
    ordered = sorted(cards, key=lambda c: c['cluster_id'])
    present = [c for c in ordered if value(c) is not None]
    absent = [c for c in ordered if value(c) is None]
    # reverse=True keeps equal elements in their input (cluster_id) order.
    present.sort(key=value, reverse=descending)
    return [*present, *absent]


def _names(names: tuple[str, ...]) -> set[str]:
    return {normalize_class_name(n) for n in names}


def filter_cards(cards: list[ClusterCard], params: ClusterListParams) -> list[ClusterCard]:
    dominant = _names(params.dominant_class)
    detector = _names(params.detector_class)
    wanted_flags = set(params.flags)

    def keep(card: ClusterCard) -> bool:
        return (
            (not dominant or normalize_class_name(card['dominant_class_name'] or '') in dominant)
            and (not detector or normalize_class_name(card['detector_top_class'] or '') in detector)
            and (params.min_size is None or card['size'] >= params.min_size)
            and (params.max_size is None or card['size'] <= params.max_size)
            and (
                params.validation_state is None
                or card['validation_state'] == params.validation_state
            )
            and (not wanted_flags or bool(wanted_flags & set(card['flags'])))
            and (
                params.needs_review is None
                or (card['needs_review_count'] > 0) is params.needs_review
            )
        )

    return [c for c in cards if keep(c)]


def _representatives(hits: list[dict[str, Any]]) -> list[ClusterRepresentative]:
    reps: list[ClusterRepresentative] = []
    for h in hits:
        src = h.get('_source') or {}
        crop_id = src.get('crop_id') or h.get('_id')
        if crop_id:
            reps.append(
                {
                    'crop_id': crop_id,
                    'cluster_distance': current_cluster_distance(src),
                    'class_name': src.get('class_name'),
                    'cluster_subid': src.get('cluster_subid'),
                }
            )
    return reps


async def fetch_representatives(
    opensearch: Any, cluster_ids: list[int], per_cluster: int
) -> list[list[ClusterRepresentative]]:
    """The top ``per_cluster`` crops of each cluster, in ``cluster_ids`` order, by one msearch."""
    if not cluster_ids:
        return []
    lines: list[dict[str, Any]] = []
    for cid in cluster_ids:
        lines.append({'index': items_index()})
        lines.append(representatives_msearch_body(cid, per_cluster))
    try:
        resp = await opensearch.msearch(body=lines)
    except Exception as exc:
        raise RequestRejectedError(500, f'representatives msearch failed: {exc}') from exc
    responses = resp.get('responses', [])
    return [
        _representatives(
            (((responses[i] if i < len(responses) else None) or {}).get('hits') or {}).get('hits')
            or []
        )
        for i in range(len(cluster_ids))
    ]


async def list_cluster_cards(opensearch: Any, params: ClusterListParams) -> ClusterListResponse:
    try:
        resp = await opensearch.search(index=items_index(), body=aggregation_body(params))
    except Exception as exc:
        raise RequestRejectedError(500, f'clusters query failed: {exc}') from exc
    buckets = resp.get('aggregations', {}).get('clusters', {}).get('buckets', [])
    # An aggregation that came back full may have dropped smaller clusters.
    truncated = len(buckets) >= params.max_clusters
    cards = [bucket_to_card(b) for b in buckets]
    cards = [c for c in cards if params.kind in ('all', c['cluster_kind'])]
    cards = sort_cards(filter_cards(cards, params), params.sort)
    if params.per_cluster > 0:
        page = cards[params.offset : params.offset + params.limit]
        reps = await fetch_representatives(
            opensearch, [c['cluster_id'] for c in page], params.per_cluster
        )
        for card, card_reps in zip(page, reps, strict=True):
            card['representatives'] = card_reps
    return {
        'items': cards,
        'total': len(cards),
        'total_class_clusters': sum(c['cluster_kind'] == 'class' for c in cards),
        'total_candidate_clusters': sum(c['cluster_kind'] == 'candidate' for c in cards),
        'cluster_id_offset': RESIDUAL_CLUSTER_ID_OFFSET,
        'purity_thresholds': cast('PurityThresholds', purity_thresholds()),
        'core_similarity_min': CORE_SIMILARITY_MIN,
        'representatives_offset': params.offset,
        'representatives_limit': params.limit,
        'sort': params.sort,
        'sort_options': [{'id': i, 'label': label} for i, label in SORT_OPTIONS],
        'flag_vocabulary': list(FLAG_VOCABULARY),
        'truncated': truncated,
    }
