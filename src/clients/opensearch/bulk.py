"""Face indexing and bulk ingestion."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.helpers import async_bulk

from src.clients.opensearch.names import (
    DetectionCategory,
    IndexName,
    get_category,
    get_class_name,
    get_cluster_index_name,
)


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.clustering import ClusteringService


logger = logging.getLogger(__name__)


class BulkMixin:
    """Face indexing and bulk ingestion."""

    client: AsyncOpenSearch
    embedding_dim: int

    async def ingest_faces(
        self,
        image_id: str,
        image_path: str,
        faces: list[dict[str, Any]],
        embeddings: list[list[float]] | np.ndarray,
        person_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        clustering_service: ClusteringService | None = None,
    ) -> dict[str, Any]:
        """
        Ingest face detections with ArcFace embeddings.

        Args:
            image_id: Unique identifier for the source image
            image_path: File path or URL to the image
            faces: List of face detections with box, landmarks, score
            embeddings: [N, 512] ArcFace embeddings
            person_name: Optional name/label for all faces in image
            metadata: Optional metadata dictionary
            clustering_service: Optional ClusteringService for cluster assignment

        Returns:
            Dict with face ingestion results
        """
        results = {
            'faces': 0,
            'clustered': 0,
            'errors': [],
        }
        timestamp = datetime.now(UTC).isoformat()

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        for i, (face, embedding) in enumerate(zip(faces, embeddings, strict=False)):
            face_id = f'{image_id}_face_{i}'

            try:
                # Convert flat landmarks list to named object
                raw_landmarks = face.get('landmarks', [])
                if len(raw_landmarks) >= 10:
                    landmarks = {
                        'left_eye': [float(raw_landmarks[0]), float(raw_landmarks[1])],
                        'right_eye': [float(raw_landmarks[2]), float(raw_landmarks[3])],
                        'nose': [float(raw_landmarks[4]), float(raw_landmarks[5])],
                        'left_mouth': [float(raw_landmarks[6]), float(raw_landmarks[7])],
                        'right_mouth': [float(raw_landmarks[8]), float(raw_landmarks[9])],
                    }
                else:
                    landmarks = {}

                face_doc = {
                    'face_id': face_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'embedding': embedding,
                    'box': face.get('box', [0, 0, 1, 1]),
                    'landmarks': landmarks,
                    'confidence': face.get('score', 0.0),
                    'quality': face.get('quality', 0.0),
                    'indexed_at': timestamp,
                }

                if person_name:
                    face_doc['person_name'] = person_name

                # Assign to face cluster
                if clustering_service is not None:
                    try:
                        cluster_idx = get_cluster_index_name(IndexName.FACES)
                        if clustering_service.is_trained(cluster_idx):
                            emb_array = np.array(embedding, dtype=np.float32)
                            assignment = clustering_service.assign_cluster(cluster_idx, emb_array)
                            face_doc['cluster_id'] = assignment.cluster_id
                            face_doc['cluster_distance'] = assignment.distance
                            face_doc['clustered_at'] = timestamp
                            results['clustered'] += 1
                    except Exception as e:
                        logger.warning(f'Cluster assignment failed for face: {e}')

                if metadata:
                    face_doc['metadata'] = metadata

                await self.client.index(
                    index=IndexName.FACES.value,
                    id=face_id,
                    body=face_doc,
                    refresh=False,
                )
                results['faces'] += 1

            except Exception as e:
                results['errors'].append(f'{face_id}: {e!s}')

        logger.info(f'Face ingestion: faces={results["faces"]}, clustered={results["clustered"]}')
        return results

    async def index_face(
        self,
        face_id: str,
        image_id: str,
        image_path: str,
        embedding: list[float] | np.ndarray,
        box: list[float],
        landmarks: list[float] | None = None,
        confidence: float = 0.0,
        quality: float = 0.0,
        person_id: str | None = None,
        person_name: str | None = None,
    ) -> bool:
        """
        Index a single face detection with ArcFace embedding.

        Args:
            face_id: Unique identifier for this face
            image_id: Source image identifier
            image_path: Path to source image
            embedding: 512-dim ArcFace embedding
            box: [x1, y1, x2, y2] normalized bounding box
            landmarks: Optional [10] flat list of 5-point landmarks (x,y pairs)
            confidence: Face detection confidence score
            quality: Face quality score
            person_id: Optional person identity ID
            person_name: Optional person name label

        Returns:
            True if successfully indexed
        """
        timestamp = datetime.now(UTC).isoformat()

        # Convert embedding to list if numpy array
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()

        # Convert flat landmarks list to named object
        landmarks_obj = {}
        if landmarks and len(landmarks) >= 10:
            landmarks_obj = {
                'left_eye': [float(landmarks[0]), float(landmarks[1])],
                'right_eye': [float(landmarks[2]), float(landmarks[3])],
                'nose': [float(landmarks[4]), float(landmarks[5])],
                'left_mouth': [float(landmarks[6]), float(landmarks[7])],
                'right_mouth': [float(landmarks[8]), float(landmarks[9])],
            }

        face_doc = {
            'face_id': face_id,
            'image_id': image_id,
            'image_path': image_path,
            'embedding': embedding,
            'box': box,
            'landmarks': landmarks_obj,
            'confidence': confidence,
            'quality_score': quality,
            'indexed_at': timestamp,
        }

        if person_id:
            face_doc['person_id'] = person_id
        if person_name:
            face_doc['person_name'] = person_name

        await self.client.index(
            index=IndexName.FACES.value,
            id=face_id,
            body=face_doc,
            refresh=False,
        )
        return True

    async def bulk_index_faces(
        self,
        face_documents: list[dict[str, Any]],
        refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Bulk index multiple faces in a single OpenSearch request.

        ~10x faster than sequential index_face() calls for batch operations.

        Args:
            face_documents: List of dicts with keys:
                - face_id, image_id, image_path, embedding, box
                - Optional: landmarks, confidence, quality, person_id, person_name
            refresh: Whether to refresh index after bulk operation

        Returns:
            Dict with 'indexed' count and 'errors' list
        """
        if not face_documents:
            return {'indexed': 0, 'errors': []}

        timestamp = datetime.now(UTC).isoformat()
        actions = []

        for face_doc in face_documents:
            # Convert embedding to list if numpy array
            embedding = face_doc.get('embedding', [])
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()

            # Convert flat landmarks list to named object
            landmarks = face_doc.get('landmarks', [])
            landmarks_obj = {}
            if landmarks and len(landmarks) >= 10:
                landmarks_obj = {
                    'left_eye': [float(landmarks[0]), float(landmarks[1])],
                    'right_eye': [float(landmarks[2]), float(landmarks[3])],
                    'nose': [float(landmarks[4]), float(landmarks[5])],
                    'left_mouth': [float(landmarks[6]), float(landmarks[7])],
                    'right_mouth': [float(landmarks[8]), float(landmarks[9])],
                }

            doc = {
                'face_id': face_doc['face_id'],
                'image_id': face_doc['image_id'],
                'image_path': face_doc['image_path'],
                'embedding': embedding,
                'box': face_doc.get('box', [0, 0, 1, 1]),
                'landmarks': landmarks_obj,
                'confidence': face_doc.get('confidence', 0.0),
                'quality_score': face_doc.get('quality', 0.0),
                'indexed_at': face_doc.get('indexed_at', timestamp),
            }

            if face_doc.get('person_id'):
                doc['person_id'] = face_doc['person_id']
            if face_doc.get('person_name'):
                doc['person_name'] = face_doc['person_name']

            actions.append(
                {
                    '_index': IndexName.FACES.value,
                    '_id': face_doc['face_id'],
                    '_source': doc,
                }
            )

        try:
            success, errors = await async_bulk(
                self.client, actions, refresh=refresh, raise_on_error=False
            )
            error_msgs = []
            if errors:
                error_msgs = [str(e) for e in errors[:10]]  # Limit error messages
                logger.warning(f'Bulk face index had {len(errors)} errors')

            logger.info(f'Bulk indexed {success} faces')
            return {'indexed': success, 'errors': error_msgs}

        except Exception as e:
            logger.error(f'Bulk face index failed: {e}')
            return {'indexed': 0, 'errors': [str(e)]}

    async def bulk_ingest(
        self,
        documents: list[dict[str, Any]],
        refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Bulk ingest multiple images with auto-routing to category-specific indexes.

        Args:
            documents: List of document dicts with:
                - image_id, image_path, global_embedding
                - Optional: box_embeddings, normalized_boxes, det_classes, det_scores
            refresh: Whether to refresh indexes after bulk operation

        Returns:
            Dict with counts per index
        """
        global_actions = []
        vehicle_actions = []
        people_actions = []
        timestamp = datetime.now(UTC).isoformat()

        for doc in documents:
            image_id = doc['image_id']
            image_path = doc['image_path']
            global_embedding = doc['global_embedding']
            metadata = doc.get('metadata')

            # Global document
            global_doc = {
                'image_id': image_id,
                'image_path': image_path,
                'global_embedding': (
                    global_embedding.tolist()
                    if hasattr(global_embedding, 'tolist')
                    else global_embedding
                ),
                'indexed_at': timestamp,
            }
            if 'width' in doc:
                global_doc['width'] = doc['width']
            if 'height' in doc:
                global_doc['height'] = doc['height']
            if metadata:
                global_doc['metadata'] = metadata

            global_actions.append(
                {
                    '_index': IndexName.GLOBAL.value,
                    '_id': image_id,
                    '_source': global_doc,
                }
            )

            # Route box embeddings
            if 'box_embeddings' in doc and 'det_classes' in doc:
                box_embeddings = doc['box_embeddings']
                det_classes = doc['det_classes']
                normalized_boxes = doc.get('normalized_boxes')
                det_scores = doc.get('det_scores')

                num_boxes = len(box_embeddings)
                for i in range(num_boxes):
                    class_id = int(det_classes[i])
                    category = get_category(class_id)
                    detection_id = f'{image_id}_box_{i}'

                    embedding = (
                        box_embeddings[i].tolist()
                        if hasattr(box_embeddings[i], 'tolist')
                        else box_embeddings[i]
                    )

                    if category == DetectionCategory.VEHICLE:
                        vehicle_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'class_id': class_id,
                            'class_name': get_class_name(class_id),
                            'indexed_at': timestamp,
                        }
                        if normalized_boxes is not None:
                            vehicle_doc['box'] = (
                                normalized_boxes[i].tolist()
                                if hasattr(normalized_boxes[i], 'tolist')
                                else normalized_boxes[i]
                            )
                        if det_scores is not None:
                            vehicle_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            vehicle_doc['metadata'] = metadata

                        vehicle_actions.append(
                            {
                                '_index': IndexName.VEHICLES.value,
                                '_id': detection_id,
                                '_source': vehicle_doc,
                            }
                        )

                    elif category == DetectionCategory.PERSON:
                        person_doc = {
                            'detection_id': detection_id,
                            'image_id': image_id,
                            'image_path': image_path,
                            'embedding': embedding,
                            'has_face': False,
                            'indexed_at': timestamp,
                        }
                        if normalized_boxes is not None:
                            person_doc['box'] = (
                                normalized_boxes[i].tolist()
                                if hasattr(normalized_boxes[i], 'tolist')
                                else normalized_boxes[i]
                            )
                        if det_scores is not None:
                            person_doc['confidence'] = float(det_scores[i])
                        if metadata:
                            person_doc['metadata'] = metadata

                        people_actions.append(
                            {
                                '_index': IndexName.PEOPLE.value,
                                '_id': detection_id,
                                '_source': person_doc,
                            }
                        )

        # Execute bulk operations
        results = {'global': 0, 'vehicles': 0, 'people': 0, 'errors': []}

        try:
            if global_actions:
                success, _ = await async_bulk(
                    self.client, global_actions, refresh=refresh, raise_on_error=False
                )
                results['global'] = success

            if vehicle_actions:
                success, _ = await async_bulk(
                    self.client, vehicle_actions, refresh=refresh, raise_on_error=False
                )
                results['vehicles'] = success

            if people_actions:
                success, _ = await async_bulk(
                    self.client, people_actions, refresh=refresh, raise_on_error=False
                )
                results['people'] = success

        except Exception as e:
            results['errors'].append(str(e))

        logger.info(
            f'Bulk multi-index ingestion: global={results["global"]}, '
            f'vehicles={results["vehicles"]}, people={results["people"]}'
        )
        return results
