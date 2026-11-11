"""``GET /stats/dataset``: the dashboard payload built from one ``size: 0`` search.

``build_dataset_stats`` runs the aggregation body of :mod:`stats_dataset_query` and rolls the
buckets up into the fixed wire fields Cropwright reads plus the pipeline dashboard fields. The
Triton repository listing (for the region stall reason) is injected by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.config.region_fields import get_region_fields
from src.config.region_state import RegionStatus
from src.domain.ingest_class_sources import (
    CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE,
    CLUSTER_MAJORITY_CLASS_SOURCE,
    DEFAULT_PROPOSAL_CLASS_SOURCE,
    LABEL_IMPORT_CLASS_SOURCE,
    VLM_CLASS_SOURCE,
    classifier_class_sources,
    unlabeled_proposal_class_sources,
)
from src.services.curation import stats_imports as imp
from src.services.curation.region_dependency_health import check_region_dependencies, stall_reason
from src.services.curation.stats_dataset_query import (
    StatsQueryError,
    build_dataset_query_body,
    filter_query,
)
from src.services.curation.stats_embedding import embedding_summary
from src.services.detection.profile_registry import region_profile_or_neutral
from src.services.embedding_spaces.readers import item_write_fields
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.curation.item_filter import ItemFilter


# Provenance keys that mean "human-applied or human-validated", used to
# avoid double-counting class_source buckets against the ``by_human``
# rollup which is already computed from class_validated / the region
# validated flag.
_HUMAN_SOURCE_PREFIXES = ('human',)


def sum_prefixed(buckets: dict[str, int], prefix: str) -> int:
    """Sum bucket counts whose key starts with ``prefix``. An empty prefix
    (e.g. no detector configured) matches nothing rather than everything."""
    if not prefix:
        return 0
    return sum(cnt for k, cnt in buckets.items() if k.startswith(prefix))


def rollup_class_sources(buckets: list[dict[str, Any]]) -> dict[str, int]:
    """Roll up ``class_source`` buckets into dashboard taxonomy.

    Returns a dict of class label provenance counts. ``by_human``
    counts ONLY rows whose ``class_source`` is a human-prefixed value
    (e.g. 'human', 'human_move'). A majority-agreement auto-validator
    also sets ``class_validated=True`` but that's auto-validation, not a
    human label — using the validated flag here would inflate by_human.
    Region-detector breakdown (detector / segmenter / human) is computed
    separately by the caller via region-detector aggregations.

    Callers pass ONLY ``class_sources_with_class`` buckets (docs that
    carry a class_id — see ``build_dataset_query_body``). This
    function used to also bucket a ``by_proposal`` count here, but the
    unlabeled-proposal class_source values (``unlabeled_proposal_class_sources``)
    are, by construction, the detector's "no class assigned yet" marker
    -- such a doc never carries a class_id, so it can never appear in
    ``class_sources_with_class``. ``by_proposal`` was therefore always 0
    on every real deployment. See ``count_by_proposal`` for the fixed
    accounting, now sourced from the class-less buckets and reported
    under ``unlabeled`` instead.
    """
    classifier_sources = classifier_class_sources() | {
        CLUSTER_MAJORITY_CLASS_SOURCE,
        CLASSIFIER_VLM_AGREEMENT_CLASS_SOURCE,
    }
    by_human = 0
    by_vlm = 0
    by_classifier = 0
    by_other = 0
    by_import = 0
    for b in buckets:
        key = str(b.get('key', ''))
        cnt = int(b.get('doc_count', 0))
        if key == LABEL_IMPORT_CLASS_SOURCE:
            by_import += cnt
        elif key.startswith(_HUMAN_SOURCE_PREFIXES):
            by_human += cnt
        elif key.startswith(VLM_CLASS_SOURCE):
            by_vlm += cnt
        elif key in classifier_sources:
            by_classifier += cnt
        else:
            # Truly unknown / future provenance — surface in 'other'
            # rather than dropping.
            by_other += cnt
    return {
        'by_human': by_human,
        'by_vlm': by_vlm,
        'by_classifier': by_classifier,
        'by_import': by_import,
        'other': by_other,
    }


def count_by_proposal(buckets: list[dict[str, Any]]) -> int:
    """Sum ``class_source`` buckets that are a detector's "proposed but not
    yet classified" marker (``unlabeled_proposal_class_sources()`` plus the
    ingest-model default). Takes ``class_sources_no_class`` buckets (docs
    with NO class_id) -- these sources never carry a class_id, so counting
    them here (not in ``rollup_class_sources``) is what makes the number
    non-zero. A subset of ``unlabeled.no_label_source``, surfaced
    explicitly the same way ``vlm_no_class`` is."""
    proposal_sources = unlabeled_proposal_class_sources() | {DEFAULT_PROPOSAL_CLASS_SOURCE}
    return sum(
        int(b.get('doc_count', 0)) for b in buckets if str(b.get('key', '')) in proposal_sources
    )


async def read_last_clustering_meta(
    session: AsyncSession,
    fallback_residual: int,
    fallback_noise: int,
) -> dict[str, Any]:
    """The last finished item clustering run, from the newest ``ready`` or ``discarded``
    partition (whichever path started the run). ``cluster_count`` is the run's own count (the
    caller replaces it with the index total and moves this to ``last_run_cluster_count``). With
    no partition the time and method are ``None`` and the counts fall back to the index
    aggregates."""
    from src.config.project_context import current_project_id
    from src.db.session import project_tx
    from src.services.clustering.partitions import list_partitions

    async with project_tx(session, current_project_id()):
        rows = await list_partitions(session, target='item', limit=20)
    finished = next((r for r in rows if r.status in ('ready', 'discarded')), None)
    if finished is None:
        return {
            'last_run_at': None,
            'cluster_count': 0,
            'residual_count': fallback_residual,
            'noise_count': fallback_noise,
            'method': None,
        }
    return {
        'last_run_at': finished.created_at,
        'cluster_count': int(finished.n_clusters or 0),
        'residual_count': int(finished.n_items or fallback_residual),
        'noise_count': int(finished.n_unassigned or fallback_noise),
        'method': finished.method,
    }


async def build_dataset_stats(
    opensearch: Any,
    item_filter: ItemFilter,
    session: AsyncSession,
    *,
    repository_index: Callable[[], Awaitable[list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Overall dataset balance + label-source + pipeline breakdown.

    Single ``_search?size=0`` with composite aggregations. ``total_crops``,
    ``validated``, ``test_holdout`` and ``by_source`` are fixed wire fields
    that Cropwright reads.

    The dashboard fields:

    - ``as_of`` — ISO8601 timestamp of the query.
    - ``labeled.{by_human, by_vlm, by_classifier, other}`` — rolled-up counts
      derived from ``class_source`` plus the ``class_validated`` /
      region-validated flags.
    - ``unlabeled.{pending_detection, pending_verification, no_label_source,
      vlm_no_class, by_proposal}`` — the region-status field plus crops with
      no ``class_id``. ``by_proposal`` is the fixed accounting for
      detector-proposed-but-not-yet-classified crops (moved here from the
      always-0 ``labeled.by_proposal``, since those class_source values
      never carry a class_id).
    - ``in_progress.region_drain_total_unfinished`` — matches the value
      returned by ``/curation/ingest/region_drain``.
    - ``clusters.{last_run_at, cluster_count, residual_count, noise_count, method}`` —
      sourced from the record each clustering run writes (``clustering.last_run``);
      ``cluster_count`` falls back to the live ``cluster_uid`` cardinality.

    ``repository_index`` lists the Triton model repository for the stall reason. Raises
    :class:`InvalidItemFilterError` for a filter that cannot be built and
    :class:`StatsQueryError` when the search fails.
    """
    fields = get_region_fields()
    embedding = await item_write_fields()
    body = build_dataset_query_body(fields, embedding)
    if not item_filter.is_empty():
        body['query'] = filter_query(item_filter, embedding)
    try:
        # This is also the query the SSE stats stream re-runs
        # every ~10-15s (see pipeline_events.py's TTL cache). Shard
        # request-cache eligible (size:0, no `now`/random scoring) —
        # OpenSearch invalidates it on every index refresh anyway, so
        # this only helps (bursts of refresh-free polls hit cache) and
        # can't make results staler than they already are.
        resp = await opensearch.search(index=items_index(), body=body, request_cache=True)
    except Exception as exc:
        raise StatsQueryError from exc

    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    aggs = resp.get('aggregations') or {}

    # Rollup is built from ``class_sources_with_class`` (docs that
    # actually carry a class_id), not the flat ``class_sources`` agg --
    # otherwise a class-less vlm_unmatched/vlm_new_class_pending crop
    # counts as VLM-labeled. See the agg comment in
    # ``build_dataset_query_body``.
    with_class_buckets = ((aggs.get('class_sources_with_class') or {}).get('by_source') or {}).get(
        'buckets'
    ) or []
    rollup = rollup_class_sources(with_class_buckets)

    no_class_buckets = ((aggs.get('class_sources_no_class') or {}).get('by_source') or {}).get(
        'buckets'
    ) or []
    vlm_no_class = sum_prefixed(
        {str(b.get('key', '')): int(b.get('doc_count', 0)) for b in no_class_buckets},
        VLM_CLASS_SOURCE,
    )
    # was miscounted as (always-zero) labeled.by_proposal; these
    # sources never carry a class_id, so the real count only exists among
    # the class-less buckets.
    by_proposal = count_by_proposal(no_class_buckets)

    # Region-detector rollup — separate from class label rollup.
    # by_detector counts crops where the profile's primary region detector
    # found the region; by_segmenter where its secondary segmenter did.
    # region_total is the denominator for "% of crops with a region detection".
    profile = region_profile_or_neutral()
    region_detector_buckets: dict[str, int] = {}
    for b in ((aggs.get('region_detectors') or {}).get('by_detector') or {}).get('buckets') or []:
        # `crops.doc_count` (M1 fix), not the bucket's own `doc_count` --
        # the bucket counts BOXES under the nested agg; `reverse_nested`
        # un-nests back to the parent crop, so a crop with >1 matching box
        # is counted once.
        crop_count = (b.get('crops') or {}).get('doc_count', b.get('doc_count', 0))
        region_detector_buckets[str(b.get('key', ''))] = int(crop_count)
    regions_by_detector = sum_prefixed(region_detector_buckets, profile.detector_model)
    regions_by_segmenter = sum_prefixed(region_detector_buckets, profile.segmenter_name)
    regions_by_human_drew = sum_prefixed(region_detector_buckets, profile.human_detector_name)
    region_total_detected = sum(region_detector_buckets.values())

    # region-verifier rollup — distinct attribution of who confirmed the
    # region, regardless of who detected the bbox. A region is verified
    # either by a human or by the VLM (whose verifier value is the VLM's
    # model id, so "not human" is the only deployment-neutral test).
    region_verifier_buckets: dict[str, int] = {}
    for b in (aggs.get('region_verifiers') or {}).get('buckets') or []:
        region_verifier_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    regions_verified_by_human = sum_prefixed(region_verifier_buckets, profile.human_detector_name)
    regions_verified_by_vlm = imp.verified_by_vlm(
        region_verifier_buckets, regions_verified_by_human
    )

    # Validated-by-human union (drew the bbox OR confirmed an AI bbox).
    # This is the honest "you reviewed N regions" count for the dashboard.
    regions_validated_by_human = int(
        (aggs.get('regions_validated_by_human') or {}).get('doc_count', 0)
    )

    region_status_buckets: dict[str, int] = {}
    for b in (aggs.get('region_status') or {}).get('buckets') or []:
        region_status_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))

    pending_buckets: dict[str, int] = {}
    in_scope = aggs.get('region_pending_in_scope') or {}
    for b in (in_scope.get('by_status') or {}).get('buckets') or []:
        pending_buckets[str(b.get('key', ''))] = int(b.get('doc_count', 0))
    pending_detection = pending_buckets.get(RegionStatus.PENDING_DETECTION, 0)
    pending_verification = pending_buckets.get(RegionStatus.PENDING_VERIFICATION, 0)
    region_drain_total_unfinished = pending_detection + pending_verification

    # V-1: same stall-reason computation GET /ingest/region_drain serves,
    # mirrored here so the dashboard's "In-flight pipeline" panel (which
    # reads this endpoint, not region_drain) can render *why* the queue
    # isn't shrinking instead of a bare "0 (stalled)".
    region_deps = await check_region_dependencies(session, repository_index, profile)
    region_stall_reason = stall_reason(region_deps, pending_detection=pending_detection)

    no_label_source = int((aggs.get('no_label_source') or {}).get('doc_count', 0))

    cluster_meta = await read_last_clustering_meta(
        session,
        fallback_residual=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
        fallback_noise=int((aggs.get('noise_clusters') or {}).get('doc_count', 0)),
    )
    # The last run's count (often a residual pass) is not the index total: serve both.
    cluster_meta['last_run_cluster_count'] = cluster_meta['cluster_count'] or None
    cluster_meta['cluster_count'] = int(
        ((aggs.get('distinct_clusters') or {}).get('n') or {}).get('value', 0)
    )

    return {
        'as_of': now_iso(),
        # --- wire fields Cropwright's getStats() reads (do not remove) ----
        'total_crops': int(total),
        'validated': (aggs.get('validated') or {}).get('doc_count', 0),
        'validated_by_import': imp.count_in(aggs, 'validated_by_import'),
        'test_holdout': (aggs.get('test_holdout') or {}).get('doc_count', 0),
        'by_source': (aggs.get('by_source') or {}).get('buckets', []),
        # --- dashboard fields ---------------------------------------------
        'labeled': {
            # Class-label provenance (denominator = total_crops).
            # by_human counts class_source startswith 'human' — the actual
            # "human labeled the class" signal. Auto-validation (majority
            # agreement) lives in by_classifier, not by_human.
            **rollup,
        },
        # Region-detection provenance (denominator = total_crops).
        #
        # - ``by_detector`` / ``by_segmenter`` / ``by_human_drew`` are the
        #   *detector* counts (who created the bbox). by_human_drew is
        #   the strict 'operator drew a new bbox from scratch' count.
        # - ``verified_by_human`` / ``verified_by_vlm`` are the
        #   *verifier* counts (who said 'yes that's a region').
        # - ``validated_by_human`` is the union — every region the
        #   operator touched, whether they drew the bbox or confirmed an
        #   AI-proposed one. This is the honest 'I reviewed N regions'
        #   number the dashboard surfaces to the operator.
        'regions': {
            # boxed = crops with a region bbox right now (the honest
            # "crops with a region" count). confirmed = the pipeline said
            # it's a real region (region status == 'detected').
            # total_detected sums detector CREDIT and includes
            # rejected/failed attempts, so it overstates real regions —
            # kept for back-compat but no longer the headline number.
            'boxed': int((aggs.get('region_boxed') or {}).get('doc_count', 0)),
            'confirmed': region_status_buckets.get(RegionStatus.DETECTED, 0),
            'total_detected': region_total_detected,
            'by_detector': regions_by_detector,
            'by_segmenter': regions_by_segmenter,
            'by_human_drew': regions_by_human_drew,
            'verified_by_human': regions_verified_by_human,
            'verified_by_vlm': regions_verified_by_vlm,
            'validated_by_human': regions_validated_by_human,
            **imp.import_region_counts(aggs, region_verifier_buckets, region_detector_buckets),
        },
        'unlabeled': {
            'pending_detection': pending_detection,
            'pending_verification': pending_verification,
            'no_label_source': no_label_source,
            # Crops a VLM answered or proposed but that never got a
            # class_id (sources "vlm_unmatched" / "vlm_new_class_pending")
            # -- these used to be double counted as VLM labeled in the
            # "labeled" section above even though they carry no class.
            # They are a subset of no_label_source, surfaced explicitly so
            # a dashboard can tell "no class_source at all" apart from
            # "the VLM tried but didn't land on a registry class".
            'vlm_no_class': vlm_no_class,
            # the detector proposed this crop as an object of
            # interest but nothing has classified it yet -- moved here
            # from labeled.by_proposal, which was structurally always 0
            # (these class_source values never carry a class_id, so they
            # could never appear in the class_id-scoped rollup that fed
            # it). A subset of no_label_source, same as vlm_no_class.
            'by_proposal': by_proposal,
        },
        'in_progress': {
            'region_drain_total_unfinished': region_drain_total_unfinished,
            # V-1: null when nothing is pending or every region-profile
            # Triton dependency is READY; otherwise a human-readable line
            # naming which dependency is down and since when.
            'region_stall_reason': region_stall_reason,
        },
        'clusters': cluster_meta,
        'embedding': embedding_summary(aggs, int(total)),
    }
