"""Category-specific and similarity search operations."""

import logging
from typing import Any

import numpy as np

from src.clients.opensearch.client import OpenSearchClient
from src.clients.opensearch.names import DetectionCategory, get_category
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


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
        query_embedding = self.inference.encode_image(image_bytes, use_cache=True)
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
        query_embedding = self.inference.encode_text(text, use_cache)
        return await self.opensearch.search_global(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

    async def search_vehicles(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
        class_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Find similar vehicles across all images.

        Like "Find all red cars" or "Show me motorcycles like this one".

        Pipeline:
        1. Run YOLO detection + MobileCLIP encoding to get vehicle embedding
        2. k-NN search on visual_search_vehicles index

        Args:
            image_bytes: Query image with vehicle
            box_index: Which detected vehicle to use (0 = first)
            top_k: Number of results
            min_score: Minimum similarity
            class_filter: Filter by vehicle type [2=car, 3=motorcycle, 5=bus, 7=truck, 8=boat]

        Returns:
            dict with query vehicle info and matching vehicles
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        result = client.infer_yolo_clip_cpu(image_bytes)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        # Find vehicle detections
        classes = result.get('classes', [])
        vehicle_indices = [
            i for i, c in enumerate(classes) if get_category(int(c)) == DetectionCategory.VEHICLE
        ]

        if not vehicle_indices:
            return {'status': 'error', 'error': 'No vehicles detected', 'results': []}

        if box_index >= len(vehicle_indices):
            return {
                'status': 'error',
                'error': f'Vehicle index {box_index} out of range (0-{len(vehicle_indices) - 1})',
                'results': [],
            }

        actual_idx = vehicle_indices[box_index]
        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[actual_idx]

        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        query_info = {
            'box': boxes[actual_idx].tolist() if len(boxes) > actual_idx else None,
            'class_id': int(classes[actual_idx]),
            'score': float(scores[actual_idx]) if len(scores) > actual_idx else None,
        }

        results = await self.opensearch.search_vehicles(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            class_filter=class_filter,
        )

        return {'status': 'success', 'query_vehicle': query_info, 'results': results}

    async def search_people(
        self,
        image_bytes: bytes,
        box_index: int = 0,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Find similar people by appearance (clothing, pose).

        Like "Find people wearing similar outfits" - NOT identity matching.
        For identity matching, use search_faces (future - requires ArcFace).

        Pipeline:
        1. Run YOLO detection + MobileCLIP encoding to get person embedding
        2. k-NN search on visual_search_people index

        Args:
            image_bytes: Query image with person
            box_index: Which detected person to use (0 = first)
            top_k: Number of results
            min_score: Minimum similarity

        Returns:
            dict with query person info and matching people
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        result = client.infer_yolo_clip_cpu(image_bytes)

        if result['num_dets'] == 0:
            return {'status': 'error', 'error': 'No objects detected', 'results': []}

        # Find person detections
        classes = result.get('classes', [])
        person_indices = [
            i for i, c in enumerate(classes) if get_category(int(c)) == DetectionCategory.PERSON
        ]

        if not person_indices:
            return {'status': 'error', 'error': 'No people detected', 'results': []}

        if box_index >= len(person_indices):
            return {
                'status': 'error',
                'error': f'Person index {box_index} out of range (0-{len(person_indices) - 1})',
                'results': [],
            }

        actual_idx = person_indices[box_index]
        box_embeddings = np.array(result['box_embeddings'])
        query_embedding = box_embeddings[actual_idx]

        boxes = result.get('normalized_boxes', [])
        scores = result.get('scores', [])

        query_info = {
            'box': boxes[actual_idx].tolist() if len(boxes) > actual_idx else None,
            'class_id': int(classes[actual_idx]),
            'score': float(scores[actual_idx]) if len(scores) > actual_idx else None,
        }

        results = await self.opensearch.search_people(
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
        )

        return {'status': 'success', 'query_person': query_info, 'results': results}

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

        Automatically routes to vehicles or people index based on detection class.

        Args:
            image_bytes: Query image bytes
            box_index: Index of detected box to use (default: 0)
            top_k: Number of results
            min_score: Minimum similarity
            class_filter: Filter by COCO class IDs

        Returns:
            dict with results and query box info
        """
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        result = client.infer_yolo_clip_cpu(image_bytes)

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
            # For other classes, search global (fallback)
            results = await self.opensearch.search_global(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
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
        result = self.inference.recognize_faces(image_bytes)

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
