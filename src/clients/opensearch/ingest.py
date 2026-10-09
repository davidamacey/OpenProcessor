"""Single-image ingestion and hash duplicate checks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.opensearch.names import (
    DetectionCategory,
    IndexName,
    get_category,
    get_class_name,
    get_cluster_index_name,
)


if TYPE_CHECKING:
    import numpy as np
    from opensearchpy import AsyncOpenSearch

    from src.services.clustering import ClusteringService


logger = logging.getLogger(__name__)


class IngestMixin:
    """Single-image ingestion and hash duplicate checks."""

    client: AsyncOpenSearch
    embedding_dim: int

    # =========================================================================
    # Multi-Index Ingestion (Category-Specific Routing)
    # =========================================================================

    async def ingest_image(
        self,
        image_id: str,
        image_path: str,
        global_embedding: np.ndarray,
        box_embeddings: np.ndarray | None = None,
        normalized_boxes: np.ndarray | None = None,
        det_classes: list[int] | None = None,
        det_scores: list[float] | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        metadata: dict[str, Any] | None = None,
        clustering_service: ClusteringService | None = None,
        imohash: str | None = None,
        file_size_bytes: int | None = None,
    ) -> dict[str, Any]:
        """
        Ingest image with auto-routing to category-specific indexes.

        Routes detections by class:
        - Person (class 0) -> visual_search_people
        - Vehicles (class 2,3,5,7,8) -> visual_search_vehicles
        - Global embedding -> visual_search_global

        When clustering_service is provided, each document gets:
        - cluster_id: FAISS IVF cluster assignment (~0.1ms overhead)
        - cluster_distance: Distance to cluster centroid

        Args:
            image_id: Unique identifier for the image
            image_path: File path or URL to the image
            global_embedding: 512-dim L2-normalized embedding for entire image
            box_embeddings: [N, 512] embeddings for detected objects
            normalized_boxes: [N, 4] boxes in [0, 1] range
            det_classes: [N] class IDs for detected objects
            det_scores: [N] detection confidence scores
            image_width: Image width in pixels
            image_height: Image height in pixels
            metadata: Optional dictionary for custom fields
            clustering_service: Optional ClusteringService for cluster assignment
            imohash: Optional imohash for duplicate detection
            file_size_bytes: Optional file size for additional duplicate verification

        Returns:
            Dict with ingestion results per index
        """
        results = {
            'global': False,
            'vehicles': 0,
            'people': 0,
            'skipped': 0,
            'clustered': 0,
            'errors': [],
        }
        timestamp = datetime.now(UTC).isoformat()

        # 1. Ingest global embedding
        try:
            global_doc = {
                'image_id': image_id,
                'image_path': image_path,
                'global_embedding': global_embedding.tolist(),
                'indexed_at': timestamp,
            }

            # Assign to cluster if clustering is available
            if clustering_service is not None:
                try:
                    cluster_idx = get_cluster_index_name(IndexName.GLOBAL)
                    if clustering_service.is_trained(cluster_idx):
                        assignment = clustering_service.assign_cluster(
                            cluster_idx, global_embedding
                        )
                        global_doc['cluster_id'] = assignment.cluster_id
                        global_doc['cluster_distance'] = assignment.distance
                        global_doc['clustered_at'] = timestamp
                        results['clustered'] += 1
                except Exception as e:
                    logger.warning(f'Cluster assignment failed for global: {e}')

            if image_width:
                global_doc['width'] = image_width
            if image_height:
                global_doc['height'] = image_height
            if metadata:
                global_doc['metadata'] = metadata
            if imohash:
                global_doc['imohash'] = imohash
            if file_size_bytes:
                global_doc['file_size_bytes'] = file_size_bytes

            await self.client.index(
                index=IndexName.GLOBAL.value,
                id=image_id,
                body=global_doc,
                refresh=False,
            )
            results['global'] = True
        except Exception as e:
            results['errors'].append(f'Global: {e!s}')

        # 2. Route box embeddings to category-specific indexes
        if box_embeddings is not None and det_classes is not None:
            num_boxes = box_embeddings.shape[0]

            for i in range(num_boxes):
                class_id = int(det_classes[i])
                category = get_category(class_id)
                detection_id = f'{image_id}_box_{i}'

                try:
                    if category == DetectionCategory.VEHICLE:
                        # Index vehicle detection
                        vehicle_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': box_embeddings[i].tolist(),
                            'class_id': class_id,
                            'class_name': get_class_name(class_id),
                            'indexed_at': timestamp,
                        }

                        # Assign to vehicle cluster
                        if clustering_service is not None:
                            try:
                                cluster_idx = get_cluster_index_name(IndexName.VEHICLES)
                                if clustering_service.is_trained(cluster_idx):
                                    assignment = clustering_service.assign_cluster(
                                        cluster_idx, box_embeddings[i]
                                    )
                                    vehicle_doc['cluster_id'] = assignment.cluster_id
                                    vehicle_doc['cluster_distance'] = assignment.distance
                                    vehicle_doc['clustered_at'] = timestamp
                                    results['clustered'] += 1
                            except Exception as e:
                                logger.warning(f'Cluster assignment failed for vehicle: {e}')

                        if normalized_boxes is not None:
                            vehicle_doc['box'] = normalized_boxes[i].tolist()
                        if det_scores is not None:
                            vehicle_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            vehicle_doc['metadata'] = metadata

                        await self.client.index(
                            index=IndexName.VEHICLES.value,
                            id=detection_id,
                            body=vehicle_doc,
                            refresh=False,
                        )
                        results['vehicles'] += 1

                    elif category == DetectionCategory.PERSON:
                        # Index person detection
                        person_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': box_embeddings[i].tolist(),
                            'has_face': False,  # Will be updated when face detection is added
                            'indexed_at': timestamp,
                        }

                        # Assign to people cluster
                        if clustering_service is not None:
                            try:
                                cluster_idx = get_cluster_index_name(IndexName.PEOPLE)
                                if clustering_service.is_trained(cluster_idx):
                                    assignment = clustering_service.assign_cluster(
                                        cluster_idx, box_embeddings[i]
                                    )
                                    person_doc['cluster_id'] = assignment.cluster_id
                                    person_doc['cluster_distance'] = assignment.distance
                                    person_doc['clustered_at'] = timestamp
                                    results['clustered'] += 1
                            except Exception as e:
                                logger.warning(f'Cluster assignment failed for person: {e}')

                        if normalized_boxes is not None:
                            person_doc['box'] = normalized_boxes[i].tolist()
                        if det_scores is not None:
                            person_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            person_doc['metadata'] = metadata

                        await self.client.index(
                            index=IndexName.PEOPLE.value,
                            id=detection_id,
                            body=person_doc,
                            refresh=False,
                        )
                        results['people'] += 1

                    else:
                        # Skip non-vehicle, non-person classes
                        results['skipped'] += 1

                except Exception as e:
                    results['errors'].append(f'{detection_id}: {e!s}')

        logger.info(
            f'Multi-index ingestion: global={results["global"]}, '
            f'vehicles={results["vehicles"]}, people={results["people"]}, '
            f'skipped={results["skipped"]}, clustered={results["clustered"]}'
        )
        return results

    async def check_duplicate_by_hash(self, imohash: str) -> dict[str, Any] | None:
        """
        Check if an image with the same imohash already exists.

        This is a fast duplicate check using constant-time hashing.
        The imohash samples only 48KB of the file (16KB x 3 locations),
        making it O(1) regardless of file size.

        Args:
            imohash: Hex string from imohash.hashbytes()

        Returns:
            Existing document if found, None if no duplicate
        """
        try:
            response = await self.client.search(
                index=IndexName.GLOBAL.value,
                body={
                    'size': 1,
                    'query': {'term': {'imohash': imohash}},
                    '_source': ['image_id', 'image_path', 'indexed_at', 'imohash'],
                },
            )

            hits = response['hits']['hits']
            if hits:
                return hits[0]['_source']
            return None

        except Exception as e:
            logger.warning(f'Duplicate check failed: {e}')
            return None

    async def check_duplicates_by_hash_batch(
        self, imohashes: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        """
        Batch check for duplicate images using msearch.

        Efficiently checks multiple hashes in a single request using OpenSearch
        multi-search API. ~10x faster than sequential checks for batch operations.

        Args:
            imohashes: List of hex strings from imohash.hashbytes()

        Returns:
            Dict mapping imohash -> existing document (or None if not found)
        """
        if not imohashes:
            return {}

        results: dict[str, dict[str, Any] | None] = dict.fromkeys(imohashes)

        try:
            # Build msearch request body
            body = []
            for imohash in imohashes:
                # Header line (index)
                body.append({'index': IndexName.GLOBAL.value})
                # Query line
                body.append(
                    {
                        'size': 1,
                        'query': {'term': {'imohash': imohash}},
                        '_source': ['image_id', 'image_path', 'indexed_at', 'imohash'],
                    }
                )

            response = await self.client.msearch(body=body)

            # Parse responses
            for i, resp in enumerate(response['responses']):
                if 'error' in resp:
                    logger.warning(f'msearch error for hash {imohashes[i]}: {resp["error"]}')
                    continue
                hits = resp.get('hits', {}).get('hits', [])
                if hits:
                    results[imohashes[i]] = hits[0]['_source']

            return results

        except Exception as e:
            logger.warning(f'Batch duplicate check failed: {e}')
            return results
