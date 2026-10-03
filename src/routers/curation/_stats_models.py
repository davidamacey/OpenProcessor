"""Wire model for ``GET /stats/dataset`` (also the live stats stream's payload).

Documentation/OpenAPI model: the route declares it through ``responses=``, and
a test pins that every key :func:`dataset_stats` emits is a field here.
Leaf module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.services.curation.stats_embedding import EmbeddingBreakdown  # noqa: TC001 - pydantic


class SourceBucket(BaseModel):
    key: str
    doc_count: int


class LabeledStats(BaseModel):
    """Class-label provenance of items that carry a class."""

    by_human: int
    by_vlm: int
    by_classifier: int
    by_import: int
    other: int


class RegionStats(BaseModel):
    boxed: int = Field(description='Items with a region box right now.')
    confirmed: int = Field(description="Items whose region status is 'detected'.")
    total_detected: int = Field(
        description='Detector credit, rejected and failed attempts included; overstates real regions.'
    )
    by_detector: int
    by_segmenter: int
    by_human: int = Field(description='Legacy name of by_human_drew.')
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
