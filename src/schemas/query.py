"""Response bodies of the data retrieval routes (index stats, stored images, duplicates)."""

from pydantic import BaseModel, Field


class IndexStats(BaseModel):
    """Statistics for a single index."""

    name: str = Field(..., description='Index name')
    doc_count: int = Field(default=0, description='Number of documents')
    size_bytes: int = Field(default=0, description='Index size in bytes')
    size_human: str = Field(default='0 B', description='Human-readable size')
    exists: bool = Field(default=False, description='Whether index exists')


class AllIndexStatsResponse(BaseModel):
    """Response for all index statistics."""

    status: str = Field(..., description="'success' or 'error'")
    indexes: list[IndexStats] = Field(default_factory=list, description='Stats per index')
    total_documents: int = Field(default=0, description='Total documents across all indexes')
    total_size_bytes: int = Field(default=0, description='Total size in bytes')
    total_size_human: str = Field(default='0 B', description='Human-readable total size')
    error: str | None = Field(default=None, description='Error message if failed')


class ImageMetadata(BaseModel):
    """Stored image metadata."""

    image_id: str = Field(..., description='Image identifier')
    image_path: str | None = Field(default=None, description='Original file path')
    width: int | None = Field(default=None, description='Image width in pixels')
    height: int | None = Field(default=None, description='Image height in pixels')
    indexed_at: str | None = Field(default=None, description='Indexing timestamp')
    imohash: str | None = Field(default=None, description='Content hash for deduplication')
    file_size_bytes: int | None = Field(default=None, description='File size in bytes')

    # Duplicate info
    duplicate_group_id: str | None = Field(default=None, description='Duplicate group if assigned')
    is_duplicate_primary: bool | None = Field(
        default=None, description='Whether this is the primary'
    )
    duplicate_score: float | None = Field(default=None, description='Similarity to primary')

    # Cluster info
    cluster_id: int | None = Field(default=None, description='Assigned cluster ID')
    cluster_distance: float | None = Field(default=None, description='Distance to cluster centroid')


class ImageQueryResponse(BaseModel):
    """Response for single image query."""

    status: str = Field(..., description="'success', 'not_found', or 'error'")
    image_id: str = Field(..., description='Queried image ID')

    # Metadata
    metadata: ImageMetadata | None = Field(default=None, description='Image metadata')

    # What indexes contain this image
    indexed_in: list[str] = Field(default_factory=list, description='Indexes containing this image')

    # Detection counts
    num_detections: int = Field(default=0, description='Number of YOLO detections')
    num_faces: int = Field(default=0, description='Number of faces indexed')
    num_vehicle_detections: int = Field(default=0, description='Vehicle detections in index')
    num_person_detections: int = Field(default=0, description='Person detections in index')
    has_ocr: bool = Field(default=False, description='Whether OCR text was indexed')

    # Optional: Include embedding
    global_embedding: list[float] | None = Field(
        default=None, description='512-dim CLIP embedding (if include_embedding=True)'
    )

    error: str | None = Field(default=None, description='Error message if failed')


class DeleteImageResponse(BaseModel):
    """Response for image deletion."""

    status: str = Field(..., description="'success' or 'error'")
    image_id: str = Field(..., description='Deleted image ID')
    deleted_from: list[str] = Field(default_factory=list, description='Indexes deleted from')
    total_deleted: int = Field(default=0, description='Total documents deleted')
    error: str | None = Field(default=None, description='Error message if failed')


class DuplicateGroupSummary(BaseModel):
    """Summary of a duplicate group."""

    group_id: str = Field(..., description='Duplicate group ID')
    primary_image_id: str = Field(..., description='Primary image in group')
    primary_image_path: str | None = Field(default=None, description='Primary image path')
    member_count: int = Field(..., description='Total images in group')


class DuplicateGroupsResponse(BaseModel):
    """Response for listing duplicate groups."""

    status: str = Field(..., description="'success' or 'error'")
    total_groups: int = Field(default=0, description='Total number of duplicate groups')
    total_duplicates: int = Field(default=0, description='Total images in groups')
    page: int = Field(default=0, description='Current page')
    size: int = Field(default=50, description='Page size')
    groups: list[DuplicateGroupSummary] = Field(
        default_factory=list, description='Duplicate groups'
    )
    error: str | None = Field(default=None, description='Error message if failed')


class DuplicateGroupMember(BaseModel):
    """Member of a duplicate group."""

    image_id: str = Field(..., description='Image identifier')
    image_path: str | None = Field(default=None, description='Image file path')
    is_primary: bool = Field(default=False, description='Whether this is the primary')
    duplicate_score: float | None = Field(default=None, description='Similarity to primary')
    width: int | None = Field(default=None, description='Image width')
    height: int | None = Field(default=None, description='Image height')
    indexed_at: str | None = Field(default=None, description='Indexing timestamp')


class DuplicateGroupDetailResponse(BaseModel):
    """Response for duplicate group detail."""

    status: str = Field(..., description="'success' or 'error'")
    group_id: str = Field(..., description='Duplicate group ID')
    member_count: int = Field(default=0, description='Total members')
    members: list[DuplicateGroupMember] = Field(default_factory=list, description='Group members')
    error: str | None = Field(default=None, description='Error message if failed')


class DuplicateStatsResponse(BaseModel):
    """Response for duplicate detection statistics."""

    status: str = Field(..., description="'success' or 'error'")
    total_images: int = Field(default=0, description='Total images in global index')
    grouped_images: int = Field(default=0, description='Images in duplicate groups')
    ungrouped_images: int = Field(default=0, description='Images not in any group')
    duplicate_groups: int = Field(default=0, description='Number of duplicate groups')
    average_group_size: float = Field(default=0, description='Average images per group')
    error: str | None = Field(default=None, description='Error message if failed')
