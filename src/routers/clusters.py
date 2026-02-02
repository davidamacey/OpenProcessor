"""
Clustering & Albums Router - FAISS IVF Clustering for Visual Search.

Provides clustering operations for organizing images into visual albums
using industry-standard FAISS IVF indexes.

Endpoints:
- POST /clusters/train/{index} - Train FAISS clustering for an index
- POST /clusters/assign/{index} - Assign unclustered items to clusters
- GET /clusters/stats/{index} - Get cluster statistics
- GET /clusters/{index}/{cluster_id} - Get cluster members
- POST /clusters/rebalance/{index} - Force rebalance
- GET /clusters/albums - List auto-generated albums
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from src.core.dependencies import VisualSearchDep


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/clusters',
    tags=['Clustering & Albums'],
    default_response_class=ORJSONResponse,
)


# =============================================================================
# Response Models
# =============================================================================


class TrainResponse(BaseModel):
    """Response for cluster training endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    index_name: str = Field(..., description='Index that was trained')
    n_vectors: int = Field(..., description='Number of vectors trained')
    n_clusters: int = Field(..., description='Number of clusters created')
    avg_cluster_size: float = Field(..., description='Average cluster size')
    empty_clusters: int = Field(..., description='Number of empty clusters')
    documents_updated: int = Field(..., description='Documents with cluster assignments')
    training_time_s: float = Field(..., description='Training time in seconds')
    error: str | None = Field(default=None, description='Error message if failed')


class AssignResponse(BaseModel):
    """Response for cluster assignment endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    index_name: str = Field(..., description='Index processed')
    documents_found: int = Field(default=0, description='Unclustered documents found')
    documents_updated: int = Field(default=0, description='Documents assigned to clusters')
    message: str | None = Field(default=None, description='Status message')
    error: str | None = Field(default=None, description='Error message if failed')


class FaissStats(BaseModel):
    """FAISS index statistics."""

    is_trained: bool = Field(..., description='Whether index is trained')
    n_clusters: int = Field(..., description='Number of clusters')
    n_vectors: int = Field(..., description='Number of vectors indexed')
    avg_cluster_size: float = Field(..., description='Average cluster size')
    min_cluster_size: int = Field(..., description='Minimum cluster size')
    max_cluster_size: int = Field(..., description='Maximum cluster size')
    empty_clusters: int = Field(..., description='Number of empty clusters')
    trained_at: str | None = Field(default=None, description='Training timestamp')


class ClusterInfo(BaseModel):
    """Single cluster info from OpenSearch aggregation."""

    cluster_id: int = Field(..., description='Cluster ID')
    count: int = Field(..., description='Number of items in cluster')


class ClusterStatsResponse(BaseModel):
    """Response for cluster statistics endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    index_name: str = Field(..., description='Index name')
    faiss: FaissStats | None = Field(default=None, description='FAISS index statistics')
    opensearch_clusters: list[ClusterInfo] = Field(
        default_factory=list, description='Top clusters from OpenSearch'
    )
    total_clusters_in_opensearch: int = Field(
        default=0, description='Total clusters with documents'
    )
    error: str | None = Field(default=None, description='Error message if failed')


class ClusterMember(BaseModel):
    """Single member of a cluster."""

    image_id: str = Field(..., description='Image identifier')
    image_path: str | None = Field(default=None, description='Image file path')
    cluster_distance: float | None = Field(default=None, description='Distance to centroid')
    score: float | None = Field(default=None, description='Detection confidence (if applicable)')
    class_id: int | None = Field(default=None, description='COCO class ID (if applicable)')


class ClusterMembersResponse(BaseModel):
    """Response for cluster members endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    index_name: str = Field(..., description='Index name')
    cluster_id: int = Field(..., description='Cluster ID')
    page: int = Field(..., description='Page number')
    size: int = Field(..., description='Page size')
    count: int = Field(..., description='Number of members returned')
    members: list[ClusterMember] = Field(default_factory=list, description='Cluster members')
    error: str | None = Field(default=None, description='Error message if failed')


class BalanceCheckResponse(BaseModel):
    """Response for cluster balance check."""

    status: str = Field(..., description="'success' or 'error'")
    index_name: str = Field(..., description='Index name')
    is_balanced: bool = Field(..., description='Whether clusters are balanced')
    imbalance_ratio: float = Field(..., description='Max/min cluster size ratio')
    empty_ratio: float = Field(..., description='Ratio of empty clusters')
    vectors_since_training: int = Field(..., description='New vectors since training')
    needs_rebalance: bool = Field(..., description='Whether rebalancing is recommended')
    reason: str | None = Field(default=None, description='Reason for rebalance recommendation')
    error: str | None = Field(default=None, description='Error message if failed')


class AlbumInfo(BaseModel):
    """Single auto-generated album."""

    cluster_id: int = Field(..., alias='key', description='Cluster ID (album ID)')
    count: int = Field(..., alias='doc_count', description='Number of images in album')

    class Config:
        populate_by_name = True


class ListAlbumsResponse(BaseModel):
    """Response for list albums endpoint."""

    status: str = Field(..., description="'success' or 'error'")
    total_albums: int = Field(..., description='Total number of albums')
    albums: list[dict[str, Any]] = Field(default_factory=list, description='Album list')
    error: str | None = Field(default=None, description='Error message if failed')


# =============================================================================
# Valid Index Names
# =============================================================================

VALID_INDEXES = {'global', 'vehicles', 'people', 'faces'}


def validate_index_name(index: str) -> str:
    """Validate index name parameter."""
    index_lower = index.lower()
    if index_lower not in VALID_INDEXES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index '{index}'. Must be one of: {sorted(VALID_INDEXES)}",
        )
    return index_lower


# =============================================================================
# Endpoints
# =============================================================================


@router.post('/train/{index}', response_model=TrainResponse)
async def train_clusters(
    search_service: VisualSearchDep,
    index: Annotated[str, Path(description='Index to train: global, vehicles, people, or faces')],
    n_clusters: Annotated[
        int | None, Query(ge=16, le=8192, description='Number of clusters (auto if not set)')
    ] = None,
    max_samples: Annotated[
        int | None, Query(ge=100, description='Max samples for training (all if not set)')
    ] = None,
):
    """
    Train FAISS IVF clustering for an index.

    Extracts embeddings from OpenSearch and trains a FAISS IVF index.
    This creates visual clusters that can be used as auto-generated albums.

    Training time scales with embedding count:
    - 10K images: ~2s
    - 100K images: ~15s
    - 1M images: ~120s

    After training, all documents are assigned to their nearest cluster.

    Args:
        index: Index to train (global, vehicles, people, faces)
        n_clusters: Number of clusters (auto-calculated if not provided)
        max_samples: Maximum samples for training (all if not provided)

    Returns:
        Training statistics and timing.
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.train_clusters(
            index_name=index_name,
            n_clusters=n_clusters,
            max_samples=max_samples,
        )

        if result.get('status') == 'error':
            return TrainResponse(
                status='error',
                index_name=index_name,
                n_vectors=0,
                n_clusters=0,
                avg_cluster_size=0,
                empty_clusters=0,
                documents_updated=0,
                training_time_s=0,
                error=result.get('error'),
            )

        return TrainResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            n_vectors=result.get('n_vectors', 0),
            n_clusters=result.get('n_clusters', 0),
            avg_cluster_size=result.get('avg_cluster_size', 0),
            empty_clusters=result.get('empty_clusters', 0),
            documents_updated=result.get('documents_updated', 0),
            training_time_s=result.get('training_time_s', 0),
        )

    except Exception as e:
        logger.error(f'Cluster training failed for {index_name}: {e}')
        raise HTTPException(status_code=500, detail=f'Training failed: {e!s}') from e


@router.post('/assign/{index}', response_model=AssignResponse)
async def assign_unclustered(
    search_service: VisualSearchDep,
    index: Annotated[str, Path(description='Index to process: global, vehicles, people, or faces')],
):
    """
    Assign clusters to unclustered documents.

    Finds documents without a cluster_id and assigns them to their nearest
    cluster centroid. Use this after ingesting new images to update cluster
    assignments without full retraining.

    Requires: Index must be trained first via POST /clusters/train/{index}

    Args:
        index: Index to process (global, vehicles, people, faces)

    Returns:
        Assignment statistics.
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.assign_unclustered(index_name=index_name)

        if result.get('status') == 'error':
            return AssignResponse(
                status='error',
                index_name=index_name,
                error=result.get('error'),
            )

        return AssignResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            documents_found=result.get('documents_found', 0),
            documents_updated=result.get('documents_updated', 0),
            message=result.get('message'),
        )

    except Exception as e:
        logger.error(f'Cluster assignment failed for {index_name}: {e}')
        raise HTTPException(status_code=500, detail=f'Assignment failed: {e!s}') from e


@router.get('/stats/{index}', response_model=ClusterStatsResponse)
async def get_cluster_stats(
    search_service: VisualSearchDep,
    index: Annotated[str, Path(description='Index to query: global, vehicles, people, or faces')],
):
    """
    Get detailed cluster statistics.

    Returns FAISS index stats and OpenSearch cluster aggregations.

    Args:
        index: Index to query (global, vehicles, people, faces)

    Returns:
        Cluster statistics from both FAISS and OpenSearch.
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.get_cluster_stats(index_name=index_name)

        if result.get('status') == 'error':
            return ClusterStatsResponse(
                status='error',
                index_name=index_name,
                error=result.get('error'),
            )

        faiss_data = result.get('faiss', {})
        faiss_stats = FaissStats(
            is_trained=faiss_data.get('is_trained', False),
            n_clusters=faiss_data.get('n_clusters', 0),
            n_vectors=faiss_data.get('n_vectors', 0),
            avg_cluster_size=faiss_data.get('avg_cluster_size', 0),
            min_cluster_size=faiss_data.get('min_cluster_size', 0),
            max_cluster_size=faiss_data.get('max_cluster_size', 0),
            empty_clusters=faiss_data.get('empty_clusters', 0),
            trained_at=faiss_data.get('trained_at'),
        )

        os_clusters = [
            ClusterInfo(cluster_id=c.get('key', 0), count=c.get('doc_count', 0))
            for c in result.get('opensearch_clusters', [])
        ]

        return ClusterStatsResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            faiss=faiss_stats,
            opensearch_clusters=os_clusters,
            total_clusters_in_opensearch=result.get('total_clusters_in_opensearch', 0),
        )

    except Exception as e:
        logger.error(f'Get cluster stats failed for {index_name}: {e}')
        raise HTTPException(status_code=500, detail=f'Stats retrieval failed: {e!s}') from e


@router.get('/{index}/{cluster_id}', response_model=ClusterMembersResponse)
async def get_cluster_members(
    search_service: VisualSearchDep,
    index: Annotated[str, Path(description='Index to query: global, vehicles, people, or faces')],
    cluster_id: Annotated[int, Path(ge=0, description='Cluster ID to retrieve')],
    page: Annotated[int, Query(ge=0, description='Page number (0-indexed)')] = 0,
    size: Annotated[int, Query(ge=1, le=100, description='Page size')] = 50,
):
    """
    Get members of a specific cluster (album view).

    Retrieves all images/detections assigned to a cluster, sorted by distance
    to the cluster centroid (closest first).

    Args:
        index: Index to query (global, vehicles, people, faces)
        cluster_id: Cluster ID to retrieve
        page: Page number (0-indexed)
        size: Page size (max 100)

    Returns:
        Paginated list of cluster members.
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.get_cluster_members(
            index_name=index_name,
            cluster_id=cluster_id,
            page=page,
            size=size,
        )

        if result.get('status') == 'error':
            return ClusterMembersResponse(
                status='error',
                index_name=index_name,
                cluster_id=cluster_id,
                page=page,
                size=size,
                count=0,
                error=result.get('error'),
            )

        members = [
            ClusterMember(
                image_id=m.get('image_id', ''),
                image_path=m.get('image_path'),
                cluster_distance=m.get('cluster_distance'),
                score=m.get('score'),
                class_id=m.get('class_id'),
            )
            for m in result.get('members', [])
        ]

        return ClusterMembersResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            cluster_id=result.get('cluster_id', cluster_id),
            page=result.get('page', page),
            size=result.get('size', size),
            count=result.get('count', len(members)),
            members=members,
        )

    except Exception as e:
        logger.error(f'Get cluster members failed for {index_name}/{cluster_id}: {e}')
        raise HTTPException(status_code=500, detail=f'Members retrieval failed: {e!s}') from e


@router.post('/rebalance/{index}', response_model=TrainResponse)
async def rebalance_clusters(
    search_service: VisualSearchDep,
    index: Annotated[
        str, Path(description='Index to rebalance: global, vehicles, people, or faces')
    ],
):
    """
    Force rebalance clusters by re-training.

    Re-trains the FAISS index from current data. Use when:
    - Cluster distribution has become uneven
    - Significant new data has been added since training
    - Manual rebalancing is needed

    This is equivalent to calling train with auto-calculated parameters.

    Args:
        index: Index to rebalance (global, vehicles, people, faces)

    Returns:
        Training statistics (same as /clusters/train).
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.rebalance_clusters(index_name=index_name)

        if result.get('status') == 'error':
            return TrainResponse(
                status='error',
                index_name=index_name,
                n_vectors=0,
                n_clusters=0,
                avg_cluster_size=0,
                empty_clusters=0,
                documents_updated=0,
                training_time_s=0,
                error=result.get('error'),
            )

        return TrainResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            n_vectors=result.get('n_vectors', 0),
            n_clusters=result.get('n_clusters', 0),
            avg_cluster_size=result.get('avg_cluster_size', 0),
            empty_clusters=result.get('empty_clusters', 0),
            documents_updated=result.get('documents_updated', 0),
            training_time_s=result.get('training_time_s', 0),
        )

    except Exception as e:
        logger.error(f'Cluster rebalancing failed for {index_name}: {e}')
        raise HTTPException(status_code=500, detail=f'Rebalancing failed: {e!s}') from e


@router.get('/balance/{index}', response_model=BalanceCheckResponse)
async def check_cluster_balance(
    search_service: VisualSearchDep,
    index: Annotated[str, Path(description='Index to check: global, vehicles, people, or faces')],
):
    """
    Check if clusters need rebalancing.

    Analyzes cluster distribution and recommends rebalancing when:
    - Max cluster is >10x larger than min cluster
    - Empty clusters exceed 10% of total
    - Significant new data added since training

    Args:
        index: Index to check (global, vehicles, people, faces)

    Returns:
        Balance assessment with recommendation.
    """
    index_name = validate_index_name(index)

    try:
        result = await search_service.check_cluster_balance(index_name=index_name)

        if result.get('status') == 'error':
            return BalanceCheckResponse(
                status='error',
                index_name=index_name,
                is_balanced=False,
                imbalance_ratio=0,
                empty_ratio=0,
                vectors_since_training=0,
                needs_rebalance=False,
                error=result.get('error'),
            )

        return BalanceCheckResponse(
            status='success',
            index_name=result.get('index_name', index_name),
            is_balanced=result.get('is_balanced', False),
            imbalance_ratio=result.get('imbalance_ratio', 0),
            empty_ratio=result.get('empty_ratio', 0),
            vectors_since_training=result.get('vectors_since_training', 0),
            needs_rebalance=result.get('needs_rebalance', False),
            reason=result.get('reason'),
        )

    except Exception as e:
        logger.error(f'Balance check failed for {index_name}: {e}')
        raise HTTPException(status_code=500, detail=f'Balance check failed: {e!s}') from e


@router.get('/albums', response_model=ListAlbumsResponse)
async def list_albums(
    search_service: VisualSearchDep,
    min_size: Annotated[int, Query(ge=1, description='Minimum cluster size to include')] = 5,
):
    """
    List auto-generated albums from global index clusters.

    Albums are clusters of visually similar images. Each album is identified
    by its cluster_id, which can be used to retrieve members via
    GET /clusters/global/{cluster_id}.

    Args:
        min_size: Minimum images in album (default: 5)

    Returns:
        List of albums sorted by size (largest first).
    """
    try:
        result = await search_service.list_albums(min_size=min_size)

        if result.get('status') == 'error':
            return ListAlbumsResponse(
                status='error',
                total_albums=0,
                error=result.get('error'),
            )

        return ListAlbumsResponse(
            status='success',
            total_albums=result.get('total_albums', 0),
            albums=result.get('albums', []),
        )

    except Exception as e:
        logger.error(f'List albums failed: {e}')
        raise HTTPException(status_code=500, detail=f'List albums failed: {e!s}') from e
