"""Category-specific and similarity search operations."""

import asyncio
import logging
from typing import Any

import numpy as np

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import DetectionCategory, get_category
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


def _detect_and_embed(image_bytes: bytes) -> dict[str, Any]:
    """YOLO detection plus per-box MobileCLIP vectors on the sync Triton client (blocking;
    callers run it off the event loop)."""
    from src.clients.triton_client import get_sync_triton_client
    from src.config import get_settings

    client = get_sync_triton_client(get_settings().triton_url)
    return client.infer_yolo_clip_cpu(image_bytes)


class SearchMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    # =========================================================================
    # Search Operations (Google Photos-like)
    # =========================================================================

    async def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Image-to-image similarity search (whole scene).

        Like Google Photos "Find similar images" - matches overall visual similarity.

        Pipeline:
        1. Encode query image via MobileCLIP
        2. k-NN search on visual_search_global index

        Args:
            image_bytes: Query image bytes
            top_k: Number of results to return
            min_score: Minimum similarity threshold

        Returns:
            List of similar images with scores
        """
        query_embedding = await asyncio.to_thread(
            self.inference.encode_image, image_bytes, use_cache=True
        )
        return await self.opensearch.search_global(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    async def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        min_score: float | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Text-to-image search using MobileCLIP text encoder.

        Like Google Photos search - "beach sunset", "birthday party", etc.

        Pipeline:
        1. Tokenize + encode text to embedding
        2. k-NN search on visual_search_global index

        Args:
            text: Query text string
            top_k: Number of results
            min_score: Minimum similarity
            use_cache: Use text embedding cache

        Returns:
            List of matching images with scores
        """
        query_embedding = await asyncio.to_thread(self.inference.encode_text, text, use_cache)
        return await self.opensearch.search_global(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    async def search_by_object(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Object-to-object search with auto-routing to category index.

        Automatically routes to vehicles or people index based on detection class. Any other
        class is refused with ``RequestRejectedError(422, object_class_not_indexed)``.

        Args:
            image_bytes: Query image bytes
            box_index: Index of detected box to use (default: 0)
            top_k: Number of results
            min_score: Minimum similarity
            class_filter: Filter by COCO class IDs

        Returns:
            dict with results and query box info
        """
        result = await asyncio.to_thread(_detect_and_embed, image_bytes)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        if box_index >= result['num_dets']:
            return {
                'status': 'error',
                'error': f'Box index {box_index} out of range (0-{result["num_dets"] - 1})',
                'results': [],
            }

        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[box_index]
        classes = result.get('classes', [])
        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        class_id = int(classes[box_index])
        category = get_category(class_id)

        query_info = {
            'box': boxes[box_index].tolist() if len(boxes) > box_index else None,
            'class_id': class_id,
            'category': category.value,
            'score': float(scores[box_index]) if len(scores) > box_index else None,
        }

        # Route to appropriate index
        if category == DetectionCategory.VEHICLE:
            results = await self.opensearch.search_vehicles(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                class_filter=class_filter,
            )
        elif category == DetectionCategory.PERSON:
            results = await self.opensearch.search_people(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            # A box vector belongs to the per-object indexes; the global index holds
            # whole-image vectors, so a query against it would compare two vector spaces.
            from src.core.exceptions import RequestRejectedError  # src.core imports this package

            raise RequestRejectedError(
                422,
                {
                    'error': 'object_class_not_indexed',
                    'message': (
                        f'class {class_id} (category {category.value!r}) has no per-object index; '
                        'only vehicles and people do. Use /search/image for whole-image similarity.'
                    ),
                },
            )

        return {'status': 'success', 'query_object': query_info, 'results': results}

    async def search_faces_by_image(
        self,
        image_bytes: bytes,
        face_index: int = 0,
        top_k: int = 10,
        min_score: float = 0.7,
    ) -> dict[str, Any]:
        """
        Face-to-face identity search.

        Pipeline:
        1. Run SCRFD face detection on query image
        2. Extract ArcFace embedding for selected face
        3. Search visual_search_faces index

        Args:
            image_bytes: Raw image bytes
            face_index: Which detected face to use as query (0-indexed)
            top_k: Number of results to return
            min_score: Minimum similarity score (0.7 recommended for identity)

        Returns:
            dict with query_face info and search results
        """
        # Run face detection + recognition to get embeddings via InferenceService
        result = await asyncio.to_thread(self.inference.recognize_faces, image_bytes)

        if result.get('num_faces', 0) == 0:
            return {
                'status': 'error',
                'error': 'No faces detected',
                'results': [],
                'query_face': None,
            }

        if face_index >= result['num_faces']:
            return {
                'status': 'error',
                'error': f'Face index {face_index} out of range (0-{result["num_faces"] - 1})',
                'results': [],
                'query_face': None,
            }

        # Get query face info and embedding
        query_embedding = np.array(result['face_embeddings'][face_index])
        face_data = result['faces'][face_index]
        query_face = {
            'box': face_data['box'],
            'landmarks': face_data['landmarks'],
            'score': float(face_data['score']),
            'quality': face_data.get('quality'),
        }

        # Search faces index
        results = await self.opensearch.search_faces(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return {
            'status': 'success',
            'query_face': query_face,
            'results': results,
        }
