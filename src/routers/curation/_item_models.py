"""Wire models for items: the shared item document and the browse page.

Documentation/OpenAPI models only — handlers return
``src.services.curation.wire.serialize_item`` output directly. Leaf module
(re-exported by ``_common``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ItemDoc(BaseModel):
    """The wire item every item-returning endpoint emits.

    Documentation/OpenAPI model only: handlers return
    ``src.services.curation.wire.serialize_item`` output directly (a test
    pins this model's fields to that serializer's keys), so a stored value
    of an unexpected type never 500s a browse page. Region attributes use
    the fixed ``region_<attr>`` wire names regardless of any
    ``OP_REGION_FIELD_*`` storage override.
    """

    id: str
    crop_id: str
    image_id: str = ''
    image_path: str = ''
    source_image_path: str = ''
    bbox_norm: list[float] = Field(default_factory=list)
    class_id: int | None = None
    class_name: str | None = ''
    class_source: str | None = ''
    confidence: float = 0.0
    classifier_raw_confidence: float | None = None
    # label_source is nullable: VLM writers set it to None when
    # overwriting a prior validation tag.
    label_source: str | None = ''
    # Derived: class_validated OR region_validated.
    label_validated: bool = False
    class_validated: bool = False
    class_detector: str | None = None
    class_detector_version: str | None = None
    class_labeled_at: str | None = None
    class_labeler: str | None = None
    vlm_confidence: str | None = None
    # VLM class suggestion: the registry class the VLM chose while the
    # label is unvalidated (class_source vlm / vlm_reclassified), or, for
    # vlm_new_class_pending, the proposed new class name with a null id.
    vlm_proposed_class_id: int | None = None
    vlm_proposed_class_name: str | None = None
    # What a one-key confirm applies: the VLM suggestion, else the current
    # class (name falls back to the raw unmatched VLM answer).
    proposed_class_id: int | None = None
    proposed_class_name: str = ''
    needs_new_class: bool = False
    needs_new_class_note: str | None = None
    cluster_id: int | None = None
    cluster_kind: Literal['class', 'candidate', 'unassigned'] | None = None
    cluster_distance: float | None = None
    # Similarity to the centroid (one minus the cosine distance, clamped);
    # core when at or above GET /clusters core_similarity_min.
    cluster_similarity: float | None = None
    cluster_is_core: bool | None = None
    # AHC sub-cluster id (e.g. "47a"); cleared whenever cluster_id changes.
    cluster_subid: str | None = None
    class_excluded: bool = False
    excluded_reason: str | None = None
    excluded_at: str | None = None
    # Ingest source tag.
    source: str = ''
    test_holdout: bool = False
    crop_rank_in_image: int | None = None
    crop_area_norm: float | None = None
    blur_lap_ratio: float | None = None
    proposal_name: str | None = None
    probe_pred_class: Any = None
    probe_pred_class_id: int | None = None
    probe_pred_entropy: float | None = None
    mistakenness_score: float | None = None
    mistakenness_method: str | None = None
    mistakenness_version: str | None = None
    mistakenness_scored_at: str | None = None
    uniqueness_score: float | None = None
    dup_group_id: Any = None
    dup_group_size: int | None = None
    dup_is_representative: bool | None = None
    updated_at: str = ''
    thumbnail_url: str = ''
    region_thumbnail_url: str = ''
    region_bbox_norm: list[float] | None = None
    # Derived: the region box in the item-crop frame (xyxy, [0, 1]); null
    # when there is no region or no usable item box.
    region_bbox_in_parent: list[float] | None = None
    region_bbox_frame: str | None = None
    region_bbox_correct: bool | None = None
    region_status: str | None = None
    region_score: float | None = None
    region_confidence: Any = None
    region_reason: str | None = None
    region_rejection_reason: str | None = None
    region_text: str | None = None
    region_text_raw: str | None = None
    region_text_confidence: float | None = None
    region_text_source: str | None = None
    region_text_engine_version: str | None = None
    region_validated: bool | None = None
    region_verified: bool | None = None
    region_verified_at: str | None = None
    region_verifier: str | None = None
    region_verifier_version: str | None = None
    region_visible: bool | None = None
    region_detector: str | None = None
    region_detector_version: str | None = None
    region_detector_chain: list[str] | None = None
    region_detected_at: str | None = None
    region_cluster_id: int | None = None
    region_cluster_subid: str | None = None
    region_cluster_distance: float | None = None
    region_class_id: int | None = None
    region_label_source: str | None = None
    region_source: str | None = None
    region_pairing: Any = None
    region_skip_verify: bool | None = None


class CropsPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    crops: list[ItemDoc]
    # Only set for a pool-scale overlay ordering (order='outliers' /
    # 'diverse') — the operator-facing "from N crops in scope" caption
    # needs to know it's looking at a ranked overlay rather than the
    # default newest-first sort. Absent (None) for every other ordering.
    method: str | None = None
    version: str | None = None
    n_pool: int | None = None


__all__ = ['CropsPageResponse', 'ItemDoc']
