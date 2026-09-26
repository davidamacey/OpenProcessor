"""Wire models for items: the shared item document and the browse page.

Documentation/OpenAPI models only — handlers return
``src.services.curation.wire.serialize_item`` output directly. Leaf module
(re-exported by ``_common``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ItemTextLine(BaseModel):
    """One OCR text line on the item crop (``box_norm`` in the item-crop
    frame; ``rel_height`` = line height / crop height)."""

    text: str | None = None
    box_norm: list[float] | None = None
    confidence: float | None = None
    rel_height: float | None = None


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
    # The detector/classifier score, whatever wrote the label.
    confidence: float = 0.0
    # Confidence of the writer that set the label: the VLM category mapped
    # through high 0.92 / medium 0.70 / low 0.40 ('vlm'), or the classifier
    # score ('model'); null for human / merge / import / proposal labels.
    class_confidence: float | None = None
    class_confidence_source: Literal['vlm', 'model'] | None = None
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
    # When a VLM was last asked for this item's class, and why that attempt
    # gave no class (no_answer / no_match / invalid_index / unparseable);
    # null reason = it answered. An empty answer leaves the class untouched.
    vlm_class_attempted_at: str | None = None
    vlm_class_empty_reason: str | None = None
    # The VLM's class answer verbatim (for vlm_unmatched: the label it
    # named that is not in the registry).
    vlm_raw_class: str | None = None
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
    # Cluster whose centroid is nearest this item (cluster-geometry pass).
    cluster_nearest_id: int | None = None
    # AHC sub-cluster id (e.g. "47a"); cleared whenever cluster_id changes.
    cluster_subid: str | None = None
    class_excluded: bool = False
    excluded_reason: str | None = None
    excluded_at: str | None = None
    review_dismissed_at: str | None = None
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
    # D1: null = the probe has no opinion (not scored yet, or the item's
    # class is outside the probe's class set — see probe_in_scope). The UI
    # must not offer an "accept model's class" action when this is null.
    probe_disagreement: bool | None = None
    # True once the probe has scored this item AND its class is one the
    # probe was trained on (probe_disagreement is then a real bool); False
    # when scored but out of the probe's class set (probe_disagreement is
    # then null, not agreement); null before the probe has scored it.
    probe_in_scope: bool | None = None
    # Probe checkpoint version tag -- the closest thing to a "probe run
    # id" today (see src.services.curation.probe_predictions).
    probe_model_version: str | None = None
    # Backend's own accept/no-accept decision (D1 follow-up): null mirrors
    # probe_in_scope/probe_disagreement (not scored yet); true only when
    # in-scope + disagreeing + the probe is confident enough
    # (CurationConfig.probe_actionable_min_confidence); false otherwise,
    # including disagreeing-but-unsure. Only true offers "accept model's
    # class"; disagreeing-but-not-actionable shows "model unsure: <class>"
    # with no accept action. See src.services.curation.wire.probe_actionable.
    probe_actionable: bool | None = None
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
    region_text_vlm: str | None = None
    region_text_ocr: str | None = None
    region_text_disagreement: bool | None = None
    # Why the chosen region_text won, and why the VLM's reading (kept in
    # region_text_vlm) was rejected as not text; see GET /regions/vocabulary.
    region_text_choice: str | None = None
    region_text_vlm_invalid: str | None = None
    # Human validation only (a human confirmed / drew / rejected it).
    region_validated: bool | None = None
    # The worker's auto-confirm policy accepted the box: an accepted but
    # not human-reviewed region (it stays in the region review queue).
    region_auto_confirmed: bool | None = None
    region_verified: bool | None = None
    region_verified_at: str | None = None
    region_verifier: str | None = None
    region_verifier_version: str | None = None
    region_visible: bool | None = None
    region_detector: str | None = None
    region_detector_version: str | None = None
    region_detector_chain: list[str] | None = None
    region_detected_at: str | None = None
    # A detector box the verifier rejected (region_status verify_rejected),
    # kept for review: never an accepted region. A human confirm (PATCH
    # region_meta region_status=detected, or PUT region with this box)
    # promotes it to region_bbox_norm with this provenance.
    region_candidate_bbox_norm: list[float] | None = None
    region_candidate_score: float | None = None
    region_candidate_detector: str | None = None
    region_candidate_detector_version: str | None = None
    region_candidate_source: str | None = None
    # Derived: the candidate box in the item-crop frame (xyxy, [0, 1]).
    region_candidate_bbox_in_parent: list[float] | None = None
    region_cluster_id: int | None = None
    region_cluster_subid: str | None = None
    region_cluster_distance: float | None = None
    region_class_id: int | None = None
    region_label_source: str | None = None
    region_source: str | None = None
    region_pairing: Any = None
    region_skip_verify: bool | None = None
    # Every OCR line read on the item crop ([] when none / not yet read).
    item_text_lines: list[ItemTextLine] = Field(default_factory=list)


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


__all__ = ['CropsPageResponse', 'ItemDoc', 'ItemTextLine']
