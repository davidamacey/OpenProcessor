"""
Person Management Router - Face Clustering and Identity Management.

Provides endpoints for organizing detected faces into person groups:
- Auto-clustering faces into persons based on embedding similarity
- Person lifecycle management (rename, merge, delete)
- Listing persons with face counts

Endpoints:
- POST /persons/cluster - Auto-cluster faces into persons
- GET /persons - List all persons with face counts
- GET /persons/{person_id} - Get a single person with their faces
- PUT /persons/{person_id}/name - Rename a person
- POST /persons/merge - Merge two persons into one
- DELETE /persons/{person_id} - Delete a person
"""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.services.face_identity import FaceIdentityService, get_face_identity_service


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/persons',
    tags=['Person Management'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Dependency
# =============================================================================


def get_face_identity() -> FaceIdentityService:
    """Dependency for FaceIdentityService."""
    return get_face_identity_service()


FaceIdentityDep = Annotated[FaceIdentityService, lambda: get_face_identity()]


# =============================================================================
# Response Models
# =============================================================================


class ClusterResponse(BaseModel):
    """Response for auto-clustering endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    total_faces: int = Field(..., description='Total faces processed')
    persons_created: int = Field(..., description='Number of persons created')
    faces_assigned: int = Field(..., description='Faces assigned to persons')
    singletons: int = Field(..., description='Faces without matches (individual persons)')
    similarity_threshold: float | None = Field(
        default=None, description='Similarity threshold used'
    )
    error: str | None = Field(default=None, description='Error message if failed')


class PersonFace(BaseModel):
    """Face belonging to a person."""

    face_id: str = Field(..., description='Unique face identifier')
    image_id: str | None = Field(default=None, description='Source image ID')
    image_path: str | None = Field(default=None, description='Source image path')
    box: list[float] | None = Field(
        default=None, description='Face box in source image [x1, y1, x2, y2]'
    )
    confidence: float | None = Field(default=None, description='Detection confidence')
    quality_score: float | None = Field(default=None, description='Face quality score')
    is_reference: bool | None = Field(
        default=None, description='Whether this is the reference face for the person'
    )
    thumbnail_b64: str | None = Field(default=None, description='Base64-encoded face thumbnail')
    indexed_at: str | None = Field(default=None, description='When face was indexed')


class PersonSummary(BaseModel):
    """Summary of a person (for listing)."""

    person_id: str = Field(..., description='Unique person identifier')
    person_name: str | None = Field(default=None, description='Friendly name if assigned')
    face_count: int = Field(..., description='Number of faces for this person')
    reference_face_id: str | None = Field(default=None, description='ID of the reference face')
    reference_thumbnail: str | None = Field(
        default=None, description='Base64 thumbnail of reference face'
    )


class ListPersonsResponse(BaseModel):
    """Response for list persons endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    total_persons: int = Field(..., description='Total number of persons')
    persons: list[PersonSummary] = Field(default_factory=list, description='List of persons')
    error: str | None = Field(default=None, description='Error message if failed')


class PersonDetailResponse(BaseModel):
    """Response for get person endpoint with faces."""

    status: str = Field(..., description="'success' or 'error'")
    person_id: str = Field(..., description='Person identifier')
    person_name: str | None = Field(default=None, description='Friendly name if assigned')
    face_count: int = Field(..., description='Number of faces')
    faces: list[PersonFace] = Field(default_factory=list, description='All faces for person')
    error: str | None = Field(default=None, description='Error message if failed')


class RenameRequest(BaseModel):
    """Request body for renaming a person."""

    name: str = Field(..., min_length=1, max_length=255, description='New name for the person')


class RenameResponse(BaseModel):
    """Response for rename person endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    person_id: str = Field(..., description='Person identifier')
    person_name: str = Field(..., description='New name assigned')
    faces_updated: int = Field(..., description='Number of faces updated')
    error: str | None = Field(default=None, description='Error message if failed')


class MergeRequest(BaseModel):
    """Request body for merging persons."""

    source_person_id: str = Field(..., description='Person ID to merge from (will be dissolved)')
    target_person_id: str = Field(
        ..., description='Person ID to merge into (will receive all faces)'
    )


class MergeResponse(BaseModel):
    """Response for merge persons endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    source_person_id: str = Field(..., description='Person ID merged from')
    target_person_id: str = Field(..., description='Person ID merged into')
    faces_merged: int = Field(..., description='Number of faces moved')
    error: str | None = Field(default=None, description='Error message if failed')


class DeleteResponse(BaseModel):
    """Response for delete person endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    person_id: str = Field(..., description='Person identifier')
    faces_deleted: int | None = Field(
        default=None, description='Number of faces deleted (if delete_faces=true)'
    )
    faces_unassigned: int | None = Field(
        default=None, description='Number of faces unassigned (if delete_faces=false)'
    )
    error: str | None = Field(default=None, description='Error message if failed')


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/cluster', response_model=ClusterResponse)
async def cluster_faces(
    similarity_threshold: Annotated[
        float,
        Query(
            ge=0.5,
            le=0.9,
            description='ArcFace cosine similarity threshold (0.7 = same person verification)',
        ),
    ] = 0.7,
    min_faces_per_person: Annotated[
        int,
        Query(ge=1, le=100, description='Minimum faces to form a person group'),
    ] = 2,
    max_persons: Annotated[
        int,
        Query(ge=1, le=10000, description='Maximum number of persons to create'),
    ] = 1000,
):
    """
    Auto-cluster all indexed faces into person groups.

    Uses ArcFace embedding similarity to group faces belonging to the same person.
    This is useful for organizing a photo library into people albums.

    Algorithm: Connected Components via Union-Find
    - Computes pairwise cosine similarity between all ArcFace embeddings
    - Groups faces where similarity >= threshold into connected components
    - Each connected component becomes a person group with unique person_id

    Note: For large-scale datasets (>100k faces), use POST /clusters/train/faces
    which uses GPU-accelerated FAISS IVF clustering.

    Args:
        similarity_threshold: ArcFace cosine similarity threshold.
            - 0.5: Loose matching (may include similar-looking different people)
            - 0.6: Moderate matching (good for casual photo albums)
            - 0.7: Strict matching (same person verification, recommended)
            - 0.8: Very strict (may miss some matches due to pose/lighting)
        min_faces_per_person: Minimum faces required to form a person group.
            Faces below this threshold become singletons with unique person_id.
        max_persons: Maximum number of persons to create (safety limit).

    Returns:
        Clustering statistics including persons created and faces assigned.
    """
    service = get_face_identity_service()
    try:
        result = await service.auto_cluster_faces(
            similarity_threshold=similarity_threshold,
            min_faces_per_person=min_faces_per_person,
            max_persons=max_persons,
        )

        if result.get('status') == 'error':
            return ClusterResponse(
                status='error',
                total_faces=result.get('total_faces', 0),
                persons_created=0,
                faces_assigned=0,
                singletons=0,
                error=result.get('error'),
            )

        return ClusterResponse(
            status='success',
            total_faces=result.get('total_faces', 0),
            persons_created=result.get('persons_created', 0),
            faces_assigned=result.get('faces_assigned', 0),
            singletons=result.get('singletons', 0),
            similarity_threshold=result.get('similarity_threshold'),
        )

    except Exception as e:
        logger.error(f'Face clustering failed: {e}')
        raise HTTPException(status_code=500, detail=f'Clustering failed: {e!s}') from e


@router.get('', response_model=ListPersonsResponse)
async def list_persons(
    limit: Annotated[
        int,
        Query(ge=1, le=1000, description='Maximum number of persons to return'),
    ] = 100,
):
    """
    List all persons with face counts.

    Returns a summary of each person including their ID, name (if assigned),
    face count, and reference face thumbnail for UI display.

    Persons are aggregated from the faces index by person_id.

    Args:
        limit: Maximum number of persons to return (default: 100)

    Returns:
        List of person summaries sorted by face count (descending).
    """
    service = get_face_identity_service()
    try:
        persons = await service.get_all_persons(limit=limit)

        person_summaries = [
            PersonSummary(
                person_id=p.get('person_id', ''),
                person_name=p.get('person_name'),
                face_count=p.get('face_count', 0),
                reference_face_id=p.get('reference_face_id'),
                reference_thumbnail=p.get('reference_thumbnail'),
            )
            for p in persons
        ]

        return ListPersonsResponse(
            status='success',
            total_persons=len(person_summaries),
            persons=person_summaries,
        )

    except Exception as e:
        logger.error(f'List persons failed: {e}')
        raise HTTPException(status_code=500, detail=f'List persons failed: {e!s}') from e


@router.get('/{person_id}', response_model=PersonDetailResponse)
async def get_person(
    person_id: Annotated[str, Path(description='Person identifier')],
):
    """
    Get a single person with all their faces.

    Retrieves detailed information about a person including all associated
    faces with their metadata, boxes, and thumbnails.

    Args:
        person_id: The unique person identifier

    Returns:
        Person details with all associated faces.
    """
    service = get_face_identity_service()
    try:
        faces = await service.get_person_faces(person_id)

        if not faces:
            raise HTTPException(
                status_code=404,
                detail=f'Person {person_id} not found or has no faces',
            )

        # Extract person_name from first face (all faces in a person share the name)
        person_name = faces[0].get('person_name') if faces else None

        face_models = [
            PersonFace(
                face_id=f.get('face_id', f.get('_id', '')),
                image_id=f.get('image_id'),
                image_path=f.get('image_path'),
                box=f.get('box'),
                confidence=f.get('confidence'),
                quality_score=f.get('quality_score'),
                is_reference=f.get('is_reference'),
                thumbnail_b64=f.get('thumbnail_b64'),
                indexed_at=f.get('indexed_at'),
            )
            for f in faces
        ]

        return PersonDetailResponse(
            status='success',
            person_id=person_id,
            person_name=person_name,
            face_count=len(face_models),
            faces=face_models,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Get person failed for {person_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Get person failed: {e!s}') from e


@router.put('/{person_id}/name', response_model=RenameResponse)
async def rename_person(
    person_id: Annotated[str, Path(description='Person identifier')],
    request: RenameRequest,
):
    """
    Assign or update a friendly name for a person.

    Sets the person_name field on all faces belonging to this person.
    This is useful for labeling people in a photo library (e.g., "Mom", "John").

    Args:
        person_id: The unique person identifier
        request: Request body containing the new name

    Returns:
        Rename result with number of faces updated.
    """
    service = get_face_identity_service()
    try:
        result = await service.rename_person(person_id, request.name)

        if result.get('status') == 'error':
            return RenameResponse(
                status='error',
                person_id=person_id,
                person_name=request.name,
                faces_updated=0,
                error=result.get('error'),
            )

        return RenameResponse(
            status='success',
            person_id=result.get('person_id', person_id),
            person_name=result.get('person_name', request.name),
            faces_updated=result.get('faces_updated', 0),
        )

    except Exception as e:
        logger.error(f'Rename person failed for {person_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Rename failed: {e!s}') from e


@router.post('/merge', response_model=MergeResponse)
async def merge_persons(
    request: MergeRequest,
):
    """
    Merge two persons into one.

    Moves all faces from the source person to the target person.
    The source person is effectively dissolved (no longer exists).

    Use cases:
    - Correct auto-clustering mistakes (same person split into two)
    - Manually consolidate duplicate persons

    Args:
        request: Request body with source and target person IDs

    Returns:
        Merge result with number of faces moved.
    """
    service = get_face_identity_service()
    try:
        if request.source_person_id == request.target_person_id:
            raise HTTPException(
                status_code=400,
                detail='Source and target person IDs must be different',
            )

        result = await service.merge_persons(
            source_person_id=request.source_person_id,
            target_person_id=request.target_person_id,
        )

        if result.get('status') == 'error':
            return MergeResponse(
                status='error',
                source_person_id=request.source_person_id,
                target_person_id=request.target_person_id,
                faces_merged=0,
                error=result.get('error'),
            )

        return MergeResponse(
            status='success',
            source_person_id=result.get('source_person_id', request.source_person_id),
            target_person_id=result.get('target_person_id', request.target_person_id),
            faces_merged=result.get('faces_merged', 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Merge persons failed: {e}')
        raise HTTPException(status_code=500, detail=f'Merge failed: {e!s}') from e


@router.delete('/{person_id}', response_model=DeleteResponse)
async def delete_person(
    person_id: Annotated[str, Path(description='Person identifier')],
    delete_faces: Annotated[
        bool,
        Query(
            description='If true, delete all faces. If false, unassign faces (each becomes own person).'
        ),
    ] = False,
):
    """
    Delete a person.

    Two modes:
    1. delete_faces=false (default): Dissolve the person group. Each face becomes
       its own individual person with a unique person_id.
    2. delete_faces=true: Permanently delete all faces belonging to this person
       from the index.

    Use cases:
    - Undo incorrect clustering (dissolve and re-cluster)
    - Remove unwanted person from library (delete faces)

    Args:
        person_id: The unique person identifier
        delete_faces: Whether to delete faces or just unassign them

    Returns:
        Deletion result with counts.
    """
    service = get_face_identity_service()
    try:
        result = await service.delete_person(person_id, delete_faces=delete_faces)

        if result.get('status') == 'error':
            return DeleteResponse(
                status='error',
                person_id=person_id,
                error=result.get('error'),
            )

        return DeleteResponse(
            status='success',
            person_id=result.get('person_id', person_id),
            faces_deleted=result.get('faces_deleted'),
            faces_unassigned=result.get('faces_unassigned'),
        )

    except Exception as e:
        logger.error(f'Delete person failed for {person_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Delete failed: {e!s}') from e
