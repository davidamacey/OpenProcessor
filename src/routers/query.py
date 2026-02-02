"""
Data Retrieval Router - Query stored image data and metadata.

Provides endpoints for retrieving indexed data, statistics, and managing
stored images.

Endpoints:
- GET /query/image/{image_id} - Get stored image data/metadata
- GET /query/stats - Get index statistics for all indexes
- DELETE /query/image/{image_id} - Remove image from all indexes
- GET /query/duplicates - List duplicate groups
- GET /query/duplicates/{group_id} - Get duplicate group members
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.clients.opensearch import IndexName
from src.core.dependencies import VisualSearchDep


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/query',
    tags=['Data Retrieval'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


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


# =============================================================================
# Helper Functions
# =============================================================================


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    size: float = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f'{size:.1f} {unit}'
        size /= 1024.0
    return f'{size:.1f} PB'


# =============================================================================
# Endpoints
# =============================================================================


@router.get('/stats', response_model=AllIndexStatsResponse)
async def get_all_stats(
    search_service: VisualSearchDep,
):
    """
    Get statistics for all visual search indexes.

    Returns document counts and sizes for:
    - visual_search_global (whole image embeddings)
    - visual_search_vehicles (vehicle detections)
    - visual_search_people (person detections)
    - visual_search_faces (face embeddings)
    - visual_search_ocr (text content)

    Returns:
        Statistics for all indexes.
    """
    try:
        result = await search_service.get_index_stats()

        if result.get('status') == 'error':
            return AllIndexStatsResponse(
                status='error',
                error=result.get('error'),
            )

        indexes_data = result.get('indexes', {})
        indexes = []
        total_docs = 0
        total_size = 0

        for name, stats in indexes_data.items():
            doc_count = stats.get('doc_count', 0)
            size_bytes = stats.get('size_bytes', 0)
            total_docs += doc_count
            total_size += size_bytes

            indexes.append(
                IndexStats(
                    name=name,
                    doc_count=doc_count,
                    size_bytes=size_bytes,
                    size_human=format_bytes(size_bytes),
                    exists=stats.get('exists', True),
                )
            )

        return AllIndexStatsResponse(
            status='success',
            indexes=indexes,
            total_documents=total_docs,
            total_size_bytes=total_size,
            total_size_human=format_bytes(total_size),
        )

    except Exception as e:
        logger.error(f'Get stats failed: {e}')
        raise HTTPException(status_code=500, detail=f'Stats retrieval failed: {e!s}') from e


@router.get('/image/{image_id}', response_model=ImageQueryResponse)
async def get_image_data(
    search_service: VisualSearchDep,
    image_id: Annotated[str, Path(description='Image identifier')],
    include_embedding: Annotated[bool, Query(description='Include 512-dim CLIP embedding')] = False,
):
    """
    Get stored data and metadata for an image.

    Retrieves all indexed data for an image across all indexes:
    - Global metadata (dimensions, hash, timestamps)
    - Detection counts (objects, faces, vehicles, people)
    - Clustering info (cluster_id, distance)
    - Duplicate group info

    Args:
        image_id: Image identifier
        include_embedding: Include the 512-dim CLIP embedding

    Returns:
        Image metadata and indexed data summary.
    """
    try:
        client = search_service.opensearch.client

        # Query global index for primary metadata
        try:
            global_doc = await client.get(
                index=IndexName.GLOBAL.value,
                id=image_id,
                _source_excludes=['global_embedding'] if not include_embedding else [],
            )
            source = global_doc['_source']
        except Exception:
            # Image not found in global index
            return ImageQueryResponse(
                status='not_found',
                image_id=image_id,
                error=f'Image {image_id} not found in global index',
            )

        # Build metadata
        metadata = ImageMetadata(
            image_id=image_id,
            image_path=source.get('image_path'),
            width=source.get('width'),
            height=source.get('height'),
            indexed_at=source.get('indexed_at'),
            imohash=source.get('imohash'),
            file_size_bytes=source.get('file_size_bytes'),
            duplicate_group_id=source.get('duplicate_group_id'),
            is_duplicate_primary=source.get('is_duplicate_primary'),
            duplicate_score=source.get('duplicate_score'),
            cluster_id=source.get('cluster_id'),
            cluster_distance=source.get('cluster_distance'),
        )

        indexed_in = ['global']

        # Count vehicles for this image
        vehicles_count = 0
        try:
            vehicles_resp = await client.count(
                index=IndexName.VEHICLES.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            vehicles_count = vehicles_resp.get('count', 0)
            if vehicles_count > 0:
                indexed_in.append('vehicles')
        except Exception as e:
            # Index may not exist - continue checking other indexes
            logger.debug('Could not check vehicles index: %s', e)

        # Count people for this image
        people_count = 0
        try:
            people_resp = await client.count(
                index=IndexName.PEOPLE.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            people_count = people_resp.get('count', 0)
            if people_count > 0:
                indexed_in.append('people')
        except Exception as e:
            # Index may not exist - continue checking other indexes
            logger.debug('Could not check people index: %s', e)

        # Count faces for this image
        faces_count = 0
        try:
            faces_resp = await client.count(
                index=IndexName.FACES.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            faces_count = faces_resp.get('count', 0)
            if faces_count > 0:
                indexed_in.append('faces')
        except Exception as e:
            # Index may not exist - continue checking other indexes
            logger.debug('Could not check faces index: %s', e)

        # Check OCR
        has_ocr = False
        try:
            ocr_resp = await client.count(
                index=IndexName.OCR.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            has_ocr = ocr_resp.get('count', 0) > 0
            if has_ocr:
                indexed_in.append('ocr')
        except Exception as e:
            # Index may not exist - continue checking other indexes
            logger.debug('Could not check OCR index: %s', e)

        # Get embedding if requested
        global_embedding = None
        if include_embedding and 'global_embedding' in source:
            global_embedding = source['global_embedding']

        return ImageQueryResponse(
            status='success',
            image_id=image_id,
            metadata=metadata,
            indexed_in=indexed_in,
            num_detections=vehicles_count + people_count,
            num_faces=faces_count,
            num_vehicle_detections=vehicles_count,
            num_person_detections=people_count,
            has_ocr=has_ocr,
            global_embedding=global_embedding,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Get image data failed for {image_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e


@router.delete('/image/{image_id}', response_model=DeleteImageResponse)
async def delete_image(
    search_service: VisualSearchDep,
    image_id: Annotated[str, Path(description='Image identifier to delete')],
):
    """
    Remove image from all indexes.

    Deletes the image and all associated data from:
    - visual_search_global
    - visual_search_vehicles (all detections)
    - visual_search_people (all detections)
    - visual_search_faces (all faces)
    - visual_search_ocr

    This is a permanent operation.

    Args:
        image_id: Image identifier to delete

    Returns:
        Deletion summary.
    """
    try:
        client = search_service.opensearch.client
        deleted_from = []
        total_deleted = 0

        # Delete from global
        try:
            await client.delete(index=IndexName.GLOBAL.value, id=image_id)
            deleted_from.append('global')
            total_deleted += 1
        except Exception as e:
            # Not found is OK - document may not exist in this index
            logger.debug('Could not delete from global index: %s', e)

        # Delete from vehicles (by query - multiple docs per image)
        try:
            result = await client.delete_by_query(
                index=IndexName.VEHICLES.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            count = result.get('deleted', 0)
            if count > 0:
                deleted_from.append('vehicles')
                total_deleted += count
        except Exception as e:
            # Index may not exist or document may not be in this index - continue with other indexes
            logger.debug('Could not delete from vehicles index: %s', e)

        # Delete from people
        try:
            result = await client.delete_by_query(
                index=IndexName.PEOPLE.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            count = result.get('deleted', 0)
            if count > 0:
                deleted_from.append('people')
                total_deleted += count
        except Exception as e:
            # Index may not exist or document may not be in this index - continue with other indexes
            logger.debug('Could not delete from people index: %s', e)

        # Delete from faces
        try:
            result = await client.delete_by_query(
                index=IndexName.FACES.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            count = result.get('deleted', 0)
            if count > 0:
                deleted_from.append('faces')
                total_deleted += count
        except Exception as e:
            # Index may not exist or document may not be in this index - continue with other indexes
            logger.debug('Could not delete from faces index: %s', e)

        # Delete from OCR
        try:
            await client.delete(index=IndexName.OCR.value, id=image_id)
            deleted_from.append('ocr')
            total_deleted += 1
        except Exception as e:
            # Index may not exist or document may not be in OCR index - continue
            logger.debug('Could not delete from OCR index: %s', e)

        if total_deleted == 0:
            return DeleteImageResponse(
                status='not_found',
                image_id=image_id,
                deleted_from=[],
                total_deleted=0,
                error=f'Image {image_id} not found in any index',
            )

        return DeleteImageResponse(
            status='success',
            image_id=image_id,
            deleted_from=deleted_from,
            total_deleted=total_deleted,
        )

    except Exception as e:
        logger.error(f'Delete image failed for {image_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Deletion failed: {e!s}') from e


@router.get('/duplicates', response_model=DuplicateGroupsResponse)
async def list_duplicate_groups(
    search_service: VisualSearchDep,
    page: Annotated[int, Query(ge=0, description='Page number (0-indexed)')] = 0,
    size: Annotated[int, Query(ge=1, le=100, description='Page size')] = 50,
    min_size: Annotated[int, Query(ge=2, description='Minimum group size')] = 2,
):
    """
    List duplicate groups.

    Returns groups of near-duplicate images detected during ingestion.
    Each group has a primary image (best quality) and duplicates.

    Args:
        page: Page number (0-indexed)
        size: Page size (max 100)
        min_size: Minimum group size to include (default: 2)

    Returns:
        Paginated list of duplicate groups.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)

        # Get groups
        groups = await dup_service.get_duplicate_groups(
            min_size=min_size,
            page=page,
            size=size,
        )

        # Get stats
        stats = await dup_service.get_stats()

        group_summaries = [
            DuplicateGroupSummary(
                group_id=g.group_id,
                primary_image_id=g.primary_image_id,
                primary_image_path=g.primary_image_path,
                member_count=g.member_count,
            )
            for g in groups
        ]

        return DuplicateGroupsResponse(
            status='success',
            total_groups=stats.get('duplicate_groups', 0),
            total_duplicates=stats.get('grouped_images', 0),
            page=page,
            size=size,
            groups=group_summaries,
        )

    except Exception as e:
        logger.error(f'List duplicates failed: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e


@router.get('/duplicates/stats', response_model=DuplicateStatsResponse)
async def get_duplicate_stats(
    search_service: VisualSearchDep,
):
    """
    Get duplicate detection statistics.

    Returns overall statistics about duplicate groups:
    - Total images and how many are grouped
    - Number of duplicate groups
    - Average group size

    Returns:
        Duplicate detection statistics.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)
        stats = await dup_service.get_stats()

        return DuplicateStatsResponse(
            status='success',
            total_images=stats.get('total_images', 0),
            grouped_images=stats.get('grouped_images', 0),
            ungrouped_images=stats.get('ungrouped_images', 0),
            duplicate_groups=stats.get('duplicate_groups', 0),
            average_group_size=stats.get('average_group_size', 0),
        )

    except Exception as e:
        logger.error(f'Get duplicate stats failed: {e}')
        raise HTTPException(status_code=500, detail=f'Stats retrieval failed: {e!s}') from e


@router.get('/duplicates/{group_id}', response_model=DuplicateGroupDetailResponse)
async def get_duplicate_group(
    search_service: VisualSearchDep,
    group_id: Annotated[str, Path(description='Duplicate group ID')],
):
    """
    Get members of a duplicate group.

    Returns all images in a duplicate group, sorted by:
    1. Primary first
    2. Then by similarity score (highest first)

    Args:
        group_id: Duplicate group ID

    Returns:
        Group members with metadata.
    """
    try:
        from src.services.duplicate_detection import DuplicateDetectionService

        dup_service = DuplicateDetectionService(search_service.opensearch)
        members = await dup_service.get_group_members(group_id, include_primary=True)

        if not members:
            return DuplicateGroupDetailResponse(
                status='not_found',
                group_id=group_id,
                error=f'Duplicate group {group_id} not found',
            )

        member_list = [
            DuplicateGroupMember(
                image_id=m.get('image_id', ''),
                image_path=m.get('image_path'),
                is_primary=m.get('is_duplicate_primary', False),
                duplicate_score=m.get('duplicate_score'),
                width=m.get('width'),
                height=m.get('height'),
                indexed_at=m.get('indexed_at'),
            )
            for m in members
        ]

        return DuplicateGroupDetailResponse(
            status='success',
            group_id=group_id,
            member_count=len(member_list),
            members=member_list,
        )

    except Exception as e:
        logger.error(f'Get duplicate group failed for {group_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Query failed: {e!s}') from e
