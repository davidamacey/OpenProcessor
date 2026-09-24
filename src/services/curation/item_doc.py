"""OpenSearch document builders for the curation ingest service.

Split out of :mod:`src.services.curation.ingest` (which would otherwise
exceed the repo's 700-LOC ratchet) so the "how do we shape an ingest
result into an images/items document" logic has its own file, separate
from pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config.curation import BACKBONE_EMBEDDING_FIELD
from src.services.detection.cascade_detect import class_provenance


# ``class_labeler`` recorded on every ingest-written items doc — the same
# id ``occ_upsert_bulk`` logs ingest writes under.
INGEST_CLASS_LABELER = 'ingest'


@dataclass
class DetectedItem:
    """One detected object on a source image, in the ingest pipeline's
    internal representation — full-image pixel-space bbox plus whatever
    the detector cascade attached before doc-building.
    """

    bbox_pixel: tuple[float, float, float, float]
    score: float
    class_id: int | None = None
    class_name: str | None = None
    class_source: str = 'unlabeled_proposal'
    proposal_name: str | None = None
    pe_embedding: Any | None = None  # np.ndarray | None, kept loose to avoid a numpy import here
    backbone_embedding: Any | None = None  # np.ndarray | None — BACKBONE_EMBEDDING_FIELD
    cluster_id: int | None = None
    cluster_distance: float | None = None
    # Model name + version of whichever detector last set this item's
    # class/proposal (primary, or a secondary that overrode it).
    class_detector: str | None = None
    class_detector_version: str | None = None


def build_image_doc(
    *,
    image_id: str,
    image_path: str,
    source: str,
    width: int,
    height: int,
    imohash: str,
    now: str,
    whole_frame_embedding: Any | None = None,
) -> dict[str, Any]:
    """Build the single images-index document for one ingested photo."""
    doc: dict[str, Any] = {
        'image_id': image_id,
        'image_path': image_path,
        'hdd_source': source,
        'width': width,
        'height': height,
        'imohash': imohash,
        'indexed_at': now,
        'original_resolution': f'{width}x{height}',
    }
    if whole_frame_embedding is not None:
        doc['pe_embedding'] = list(whole_frame_embedding)
    return doc


def build_item_doc(
    *,
    crop_id: str,
    image_id: str,
    image_path: str,
    source: str,
    request_id: str,
    bbox_norm: list[float],
    item: DetectedItem,
    now: str,
    crop_area_norm: float,
    crop_rank_in_image: int,
    blur_full_var: float | None,
    blur_lap_var: float | None,
    blur_lap_ratio: float | None,
) -> dict[str, Any]:
    """Build one items-index document for a single detected object.

    Field names match the mapping in ``src/clients/curation_opensearch.py``
    and the router queries in ``src/routers/curation/{crops,clusters,regions}.py``
    exactly — this is the first production writer of ``crop_area_norm``,
    ``crop_rank_in_image``, ``blur_lap_var``, ``blur_lap_ratio`` and
    ``pe_embedding`` on the items index.

    When the item carries a ``class_detector``, the class-provenance
    fields (``class_detector``, ``class_detector_version``,
    ``class_labeler='ingest'``, ``class_labeled_at=now``) are stamped
    with the same shape every other class writer uses.
    """
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': image_id,
        'image_path': image_path,
        'hdd_source': source,
        'request_id': request_id,
        'bbox_norm': bbox_norm,
        'class_source': item.class_source,
        'confidence': float(item.score),
        'class_validated': False,
        'label_source': item.class_source,
        'test_holdout': False,
        'created_at': now,
        'updated_at': now,
        'crop_area_norm': round(crop_area_norm, 6),
        'crop_rank_in_image': crop_rank_in_image,
    }
    if blur_full_var is not None and blur_full_var > 0:
        doc['blur_full_var'] = blur_full_var
    if blur_lap_var is not None:
        doc['blur_lap_var'] = blur_lap_var
    if blur_lap_ratio is not None:
        doc['blur_lap_ratio'] = blur_lap_ratio
    if item.class_id is not None:
        doc['class_id'] = item.class_id
    if item.class_name:
        doc['class_name'] = item.class_name
    if item.proposal_name:
        # Diagnostic only — the primary detector's raw proposal name,
        # kept even when a secondary/ensemble detector or human relabel
        # later overrides class_id/class_name, so mismatches stay
        # auditable.
        doc['coco_proposal_name'] = item.proposal_name
    if item.cluster_id is not None:
        doc['cluster_id'] = item.cluster_id
        doc['cluster_distance'] = item.cluster_distance
    if item.pe_embedding is not None:
        doc['pe_embedding'] = list(item.pe_embedding)
    if item.backbone_embedding is not None:
        doc[BACKBONE_EMBEDDING_FIELD] = [float(x) for x in item.backbone_embedding]
    if item.class_detector:
        doc.update(
            class_provenance(
                item.class_detector,
                item.class_detector_version or '1',
                labeler=INGEST_CLASS_LABELER,
                labeled_at=now,
            )
        )
    return doc


__all__ = ['INGEST_CLASS_LABELER', 'DetectedItem', 'build_image_doc', 'build_item_doc']
