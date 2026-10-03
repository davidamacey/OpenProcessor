"""The ``GET /stats/dataset`` aggregation body: one ``size: 0`` search whose
aggregations the route turns into the dashboard payload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.region_state import RegionStatus
from src.services.curation import stats_imports as imp
from src.services.curation.region_boxes import box_query
from src.services.curation.stats_embedding import embedding_aggregations


if TYPE_CHECKING:
    from src.config.region_fields import RegionFields


def build_dataset_query_body(fields: RegionFields) -> dict[str, Any]:
    return {
        'size': 0,
        # Exact total_crops, not the default 10000-hit cap (it would look like a stuck pipeline).
        'track_total_hits': True,
        'aggs': {
            # --- legacy fields (preserved for back-compat) ----------------
            # source/class_source/region-detector/region-verifier are
            # all mapped keyword directly on the live index — no .keyword
            # subfield exists (only the region-status field is text+.keyword).
            'by_source': {'terms': {'field': 'source', 'size': 32}},
            'validated': {'filter': {'term': {'class_validated': True}}},
            'test_holdout': {'filter': {'term': {'test_holdout': True}}},
            # --- new: label provenance breakdown --------------------------
            'class_sources': {
                # missing: a terms agg silently drops docs with no
                # class_source.keyword value (e.g. an unlabel_crop'd
                # crop) instead of bucketing them — contradicts this
                # rollup's own "surface in 'other' rather than
                # dropping" intent and broke the labeled.* buckets'
                # sum-to-total_crops invariant on a real crop. Give
                # missing values an explicit bucket key that
                # _rollup_class_sources routes to 'other'.
                'terms': {
                    'field': 'class_source',
                    'size': 64,
                    'missing': '__none__',
                },
            },
            # The flat 'class_sources' agg above buckets EVERY crop by
            # class_source regardless of whether a class_id was ever
            # assigned -- 'vlm_unmatched' / 'vlm_new_class_pending' both
            # start with 'vlm' and class_id is null on both, so the old
            # labeled.by_vlm rollup (built straight off 'class_sources')
            # counted class-less crops as VLM-labeled. This sibling agg
            # scopes the same terms breakdown to docs that actually carry
            # a class_id, so ``labeled.*`` only counts real labels.
            'class_sources_with_class': {
                'filter': {'exists': {'field': 'class_id'}},
                'aggs': {
                    'by_source': {
                        'terms': {
                            'field': 'class_source',
                            'size': 64,
                            'missing': '__none__',
                        },
                    },
                },
            },
            # The class-less half of the same breakdown, so a VLM-touched
            # but never-classed crop (vlm_unmatched / vlm_new_class_pending)
            # can be surfaced explicitly under unlabeled.* instead of
            # silently vanishing from every bucket.
            'class_sources_no_class': {
                'filter': {'bool': {'must_not': [{'exists': {'field': 'class_id'}}]}},
                'aggs': {
                    'by_source': {
                        'terms': {
                            'field': 'class_source',
                            'size': 64,
                            'missing': '__none__',
                        },
                    },
                },
            },
            # region-detector breakdown — distinct from class_source. primary detector /
            # segmenter / human region detections show up here. The dashboard
            # surfaces "detector found N regions" from this, NOT from
            # class_source (which never carries a region-detector value).
            #
            # W8-cleanup: the detector lives on each box in region_boxes now
            # (the retired item-level region_detector scalar), so this is a
            # nested agg over the box list, not a plain terms agg.
            'region_detectors': {
                'nested': {'path': fields.boxes},
                'aggs': {
                    'by_detector': {
                        'terms': {'field': f'{fields.boxes}.detector', 'size': 16},
                        # M1 fix: without `reverse_nested`, this counts
                        # BOXES, not crops -- an item with 2 accepted boxes
                        # from the same detector (or an accepted + a
                        # rejected/FP box) counted twice, inflating
                        # `total_detected` / `by_detector` below, which
                        # this dashboard number is documented (see
                        # `by_detector counts crops...` below) to count as
                        # one crop per detector.
                        'aggs': {'crops': {'reverse_nested': {}}},
                    }
                },
            },
            # region-verifier breakdown — VLM (AI) vs human. Item-level
            # (unaffected by W8): a region's per-item ``verified``/
            # ``verifier`` fields describe the item's own verification pass,
            # not any one box.
            'region_verifiers': {
                'terms': {'field': fields.verifier, 'size': 16},
            },
            # Operator-touched regions: anything where the validated flag
            # is True AND a human was involved (either drew a box OR
            # confirmed an AI-proposed one). The dashboard surfaces this as
            # the honest "you confirmed N regions today" number. "drew a
            # box" is now a nested check (any box with detector='human'),
            # not the retired item-level region_detector scalar.
            'regions_validated_by_human': {
                'filter': {
                    'bool': {
                        'filter': [{'term': {fields.validated: True}}],
                        'should': [
                            box_query({'term': {f'{fields.boxes}.detector': 'human'}}, fields),
                            {'term': {fields.verifier: 'human'}},
                        ],
                        'minimum_should_match': 1,
                    }
                }
            },
            **imp.import_aggregations(fields),
            **embedding_aggregations(),
            'region_status': {
                'terms': {'field': fields.status, 'size': 32},
            },
            # Crops that actually carry a region box right now. This — not
            # total_detected (which sums detector CREDIT, including
            # rejected/failed attempts) — is the honest "crops with a
            # region" number and matches the region cluster view. W8-cleanup:
            # a box is "carried" when it's accepted, or false_positive (kept
            # for FP analysis/training) -- the retired region_bbox_norm
            # scalar's existence used to mean the same thing.
            'region_boxed': {
                'filter': box_query(
                    {
                        'terms': {
                            f'{fields.boxes}.{fields.boxes_state}': [
                                'accepted',
                                RegionStatus.FALSE_POSITIVE.value,
                            ]
                        }
                    },
                    fields,
                )
            },
            'no_label_source': {
                'filter': {
                    'bool': {
                        'must_not': [{'exists': {'field': 'class_id'}}],
                    }
                }
            },
            # How many clusters the index holds now (noise ids < 0 are
            # not clusters).
            'distinct_clusters': {
                'filter': {'range': {'cluster_id': {'gte': 0}}},
                'aggs': {
                    'n': {'cardinality': {'field': 'cluster_id', 'precision_threshold': 4000}}
                },
            },
            # Negative cluster_id is reserved for noise (legacy HDBSCAN
            # convention; AHC doesn't emit -1 today but the agg stays so
            # any future hybrid algorithm still surfaces noise here).
            'noise_clusters': {
                'filter': {'range': {'cluster_id': {'lt': 0}}},
            },
        },
    }


__all__ = ['build_dataset_query_body']
