"""Wire model for ``GET /stats/dataset`` (also the live stats stream's payload).

Documentation/OpenAPI model: the route declares it through ``responses=``, and
a test pins that every key :func:`dataset_stats` emits is a field here.
Leaf module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.curation.views import SortOption
from src.services.curation.stats_embedding import EmbeddingBreakdown


class SourceBucket(BaseModel):
    key: str
    doc_count: int


class LabeledRow(BaseModel):
    key: str
    label: str
    count: int
    share: float = Field(description='Fraction (0 to 1) of `labeled.total`.')


class LabeledStats(BaseModel):
    """Class-label provenance of items that carry a class."""

    by_human: int
    by_vlm: int
    by_classifier: int
    by_detector: int = Field(
        description='Classes the detector named and nobody has checked (`class_source` detector).'
    )
    by_import: int
    other: int
    total: int = Field(description='Every provenance summed; the rows add up to it.')
    rows: list[LabeledRow]


class RegionStats(BaseModel):
    boxed: int = Field(description='Items with a region box right now.')
    coverage_share: float = Field(description='Fraction (0 to 1) of all items that have a box.')
    confirmed: int = Field(description="Items whose region status is 'detected'.")
    total_detected: int = Field(
        description='Detector credit, rejected and failed attempts included; overstates real regions.'
    )
    by_detector: int
    by_segmenter: int
    by_human_drew: int
    verified_by_human: int
    verified_by_vlm: int
    validated_by_human: int = Field(description='Drew the box or confirmed a proposed one.')
    by_import: int
    verified_by_import: int
    validated_by_import: int


class UnlabeledStats(BaseModel):
    pending_detection: int
    pending_verification: int
    no_label_source: int
    vlm_no_class: int
    by_proposal: int


class InProgressStats(BaseModel):
    region_drain_total_unfinished: int
    region_stall_reason: str | None = Field(
        description='Which region dependency is down, and since when; null when none is.'
    )


class ClusterStats(BaseModel):
    last_run_at: str | None
    cluster_count: int = Field(description='Distinct clusters in the index now.')
    last_run_cluster_count: int | None = Field(description="The last run's own count.")
    residual_count: int
    noise_count: int
    method: str | None


class DatasetStatsResponse(BaseModel):
    as_of: str
    join_truncated: bool = Field(
        description='The area, location or group predicates matched more images than the join '
        'could use, so the numbers may be incomplete.'
    )
    total_crops: int
    validated: int
    validated_by_import: int
    test_holdout: int
    by_source: list[SourceBucket]
    labeled: LabeledStats
    regions: RegionStats
    unlabeled: UnlabeledStats
    in_progress: InProgressStats
    clusters: ClusterStats
    embedding: EmbeddingBreakdown


__all__ = ['DatasetStatsResponse']


class ClassStatsRow(BaseModel):
    class_id: int
    class_name: str
    count: int
    validated_count: int
    trainable: int = Field(description='Validated minus holdout minus human-excluded crops.')
    trainable_gap: int = Field(description='Shortfall against the per-class minimum, floored at 0.')
    adequacy: Literal['block', 'warn', 'ok']
    aug_target: int
    aug_gap: int
    test_count: int = Field(description='Frozen test-holdout crops of the class.')
    test_deficient: bool = Field(description='test_count is below the per-class test minimum.')


class ClassStatsThresholds(BaseModel):
    block_below: int
    warn_below: int
    min_test_per_class: int
    min_train_per_class: int
    min_val_per_class: int
    aug_target_min: int
    aug_target_max: int


class ClassStatsResponse(BaseModel):
    """``GET /stats/classes``: the registry-joined ``classes`` and ``thresholds`` beside the
    raw OpenSearch ``aggregations`` (``by_class`` and the like), which are carried as extras
    so none is dropped."""

    model_config = ConfigDict(extra='allow')

    classes: list[ClassStatsRow]
    thresholds: ClassStatsThresholds
    sort_applied: str
    sort_options: list[SortOption]
