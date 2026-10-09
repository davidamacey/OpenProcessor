"""Index names, detection categories and class-to-category helpers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.services.clustering import ClusterIndex


# =============================================================================
# Index Constants and Category Mappings
# =============================================================================


class IndexName(str, Enum):
    """OpenSearch index names for visual search."""

    GLOBAL = 'visual_search_global'
    VEHICLES = 'visual_search_vehicles'
    PEOPLE = 'visual_search_people'
    FACES = 'visual_search_faces'
    OCR = 'visual_search_ocr'  # Text detection and recognition


class IndexMappingError(RuntimeError):
    """An index exists but its embedding field isn't mapped as knn_vector.

    F-25 (fresh-start E2E findings 2026-09-25): this happens when
    something writes to an index (e.g. the first POST /ingest) before
    ``create_all_indexes`` ever runs against it -- OpenSearch then
    dynamically maps the embedding field as a plain ``float`` array
    instead of ``knn_vector``, and every k-NN search against it fails
    with a message like "Field 'global_embedding' is not knn_vector
    type". That failure must not be swallowed into an empty result list:
    an outage/misconfiguration is not the same answer as "no similar
    images", and a caller has no way to tell them apart otherwise.
    """


class DetectionCategory(str, Enum):
    """Detection categories for routing."""

    VEHICLE = 'vehicle'
    PERSON = 'person'
    FACE = 'face'
    OTHER = 'other'


# COCO class ID to category mapping
VEHICLE_CLASSES = {2, 3, 5, 7, 8}  # car, motorcycle, bus, truck, boat
PERSON_CLASS = 0

# Class ID to human-readable name (COCO subset)
CLASS_NAMES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'boat',
}


def get_category(class_id: int) -> DetectionCategory:
    """Map COCO class ID to detection category."""
    if class_id == PERSON_CLASS:
        return DetectionCategory.PERSON
    if class_id in VEHICLE_CLASSES:
        return DetectionCategory.VEHICLE
    return DetectionCategory.OTHER


def get_class_name(class_id: int) -> str:
    """Get human-readable class name from COCO class ID."""
    return CLASS_NAMES.get(class_id, f'class_{class_id}')


def get_cluster_index_name(index_name: IndexName) -> ClusterIndex:
    """Map OpenSearch IndexName to FAISS ClusterIndex."""
    # Import here to avoid circular imports
    from src.services.clustering import ClusterIndex

    mapping = {
        IndexName.GLOBAL: ClusterIndex.GLOBAL,
        IndexName.VEHICLES: ClusterIndex.VEHICLES,
        IndexName.PEOPLE: ClusterIndex.PEOPLE,
        IndexName.FACES: ClusterIndex.FACES,
    }
    return mapping[index_name]
