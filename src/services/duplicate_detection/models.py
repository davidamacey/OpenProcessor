"""Result records and the default threshold for near-duplicate detection."""

from dataclasses import dataclass
from typing import Any


# Default similarity threshold for near-duplicates (matches Immich's maxDistance=0.01)
# 0.90 = similar content (variations in angle, lighting)
# 0.95 = very similar (same scene, slight variations)
# 0.99 = nearly identical (crops, resizes, compression) <- Immich default
DEFAULT_SIMILARITY_THRESHOLD = 0.99


@dataclass
class DuplicateMatch:
    """A near-duplicate match result."""

    image_id: str
    image_path: str
    similarity: float
    duplicate_group_id: str | None = None


@dataclass
class DuplicateGroup:
    """A group of near-duplicate images."""

    group_id: str
    primary_image_id: str
    primary_image_path: str
    member_count: int
    members: list[dict[str, Any]]


@dataclass
class ScanStats:
    """Statistics from a duplicate scan operation."""

    total_images: int
    images_scanned: int
    groups_created: int
    duplicates_found: int
    already_grouped: int
    scan_time_seconds: float
