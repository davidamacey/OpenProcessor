"""Single-image ingestion with category routing, near-duplicate grouping and OCR."""

import logging
from typing import Any

import numpy as np

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class IngestSingleMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    # =========================================================================
    # Ingestion Operations (Auto-Routes to Category Indexes)
    # =========================================================================

    async def ingest_image(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        skip_duplicates: bool = True,
        detect_near_duplicates: bool = True,
        near_duplicate_threshold: float = 0.99,
        enable_ocr: bool = True,
        enable_detection: bool = True,
        enable_faces: bool = True,
        enable_clip: bool = True,
    ) -> dict[str, Any]:
        """
        Ingest single image with auto-routing to category indexes.

        Pipeline:
        1. Compute imohash for exact duplicate detection (if skip_duplicates=True)
        2. Check if image already exists by hash
        3. Run YOLO detection + MobileCLIP encoding
        4. Extract global + box embeddings
        5. Route to appropriate indexes:
           - Global embedding -> visual_search_global
           - Person detections -> visual_search_people
           - Vehicle detections -> visual_search_vehicles
        6. Check for near-duplicates and auto-assign to groups (if enabled)
        7. Run OCR and index text content (if enable_ocr=True)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            metadata: Optional metadata dictionary
            skip_duplicates: If True, skip processing if image hash already exists.
                           Set to False for benchmarks. Default: True (production).
            detect_near_duplicates: If True, check for visually similar images
                           and auto-assign to duplicate groups. Default: True.
            near_duplicate_threshold: Similarity threshold for grouping.
                           Default: 0.99 (matches Immich's maxDistance=0.01).
                           Range: 0.90 (similar content) to 0.99 (near-identical).
            enable_ocr: If True, run OCR and index detected text.
                           Default: True.

        Returns:
            dict with status, routing info, and counts per category
        """
        try:
            # 1. Compute imohash for duplicate detection
            import io

            import imohash

            from src.clients.fast_face_client import get_fast_face_client
            from src.clients.triton_client import get_triton_client
            from src.config import get_settings

            image_hash = imohash.hashfileobject(io.BytesIO(image_bytes)).hex()
            file_size = len(image_bytes)

            # 2. Check for duplicates if enabled
            if skip_duplicates:
                existing = await self.opensearch.check_duplicate_by_hash(image_hash)
                if existing:
                    logger.info(
                        f'Duplicate detected: {image_id} matches {existing.get("image_id")}'
                    )
                    return {
                        'status': 'duplicate',
                        'image_id': image_id,
                        'existing_image_id': existing.get('image_id'),
                        'existing_image_path': existing.get('image_path'),
                        'imohash': image_hash,
                        'message': 'Image already exists in index (same imohash)',
                    }

            settings = get_settings()
            client = get_triton_client(settings.triton_url)

            # Initialize results
            result = {'num_dets': 0, 'image_embedding': None}
            face_result = {'num_faces': 0}
            global_embedding = None

            # Run enabled pipelines in parallel
            from concurrent.futures import ThreadPoolExecutor

            futures = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                # YOLO detection + CLIP embedding (if either is enabled)
                if enable_detection or enable_clip:

                    def run_yolo_clip():
                        return client.infer_yolo_clip_cpu(image_bytes)

                    futures['yolo_clip'] = executor.submit(run_yolo_clip)

                # Face detection + embedding (if enabled)
                if enable_faces:
                    from src.clients.fast_face_client import get_fast_face_client

                    face_client = get_fast_face_client(settings.triton_url)

                    def run_face_detection():
                        return face_client.recognize(image_bytes, confidence=0.5)

                    futures['face'] = executor.submit(run_face_detection)

                # Collect results
                if 'yolo_clip' in futures:
                    result = futures['yolo_clip'].result()
                if 'face' in futures:
                    face_result = futures['face'].result()

            # Extract global embedding if CLIP is enabled
            if enable_clip and result.get('image_embedding') is not None:
                global_embedding = np.array(result['image_embedding'])
            else:
                # Create zero embedding if CLIP disabled (still need for indexing structure)
                global_embedding = np.zeros(512, dtype=np.float32)

            # Prepare box data (only if detection is enabled)
            box_embeddings = None
            normalized_boxes = None
            det_classes = None
            det_scores = None

            if enable_detection and result['num_dets'] > 0:
                box_embeddings = np.array(result.get('box_embeddings', []))
                normalized_boxes = np.array(result.get('normalized_boxes', []))
                det_classes = result.get('classes', [])
                det_scores = result.get('scores', [])

            # Index in OpenSearch (auto-routes by class)
            ingest_result = await self.opensearch.ingest_image(
                image_id=image_id,
                image_path=image_path or image_id,
                global_embedding=global_embedding,
                box_embeddings=box_embeddings,
                normalized_boxes=normalized_boxes,
                det_classes=det_classes,
                det_scores=det_scores,
                image_width=result.get('image_width'),
                image_height=result.get('image_height'),
                metadata=metadata,
                imohash=image_hash,
                file_size_bytes=file_size,
            )

            # Index faces if detected and enabled
            num_faces_indexed = 0
            if enable_faces and face_result.get('num_faces', 0) > 0:
                face_embeddings: list = face_result.get('face_embeddings', [])  # type: ignore[assignment]
                face_boxes: list = face_result.get('face_boxes', [])  # type: ignore[assignment]
                face_scores: list = face_result.get('face_scores', [])  # type: ignore[assignment]
                face_quality: list = face_result.get('face_quality', [])  # type: ignore[assignment]
                face_landmarks: list = face_result.get('face_landmarks', [])  # type: ignore[assignment]

                for i in range(min(len(face_embeddings), len(face_boxes))):
                    if len(face_embeddings[i]) > 0:
                        face_id = f'{image_id}_face_{i}'
                        try:
                            await self.opensearch.index_face(
                                face_id=face_id,
                                image_id=image_id,
                                image_path=image_path or image_id,
                                embedding=face_embeddings[i],
                                box=face_boxes[i].tolist()
                                if hasattr(face_boxes[i], 'tolist')
                                else list(face_boxes[i]),
                                landmarks=face_landmarks[i].tolist()
                                if hasattr(face_landmarks[i], 'tolist')
                                else list(face_landmarks[i])
                                if len(face_landmarks) > i
                                else [],
                                confidence=float(face_scores[i]) if len(face_scores) > i else 0.0,
                                quality=float(face_quality[i]) if len(face_quality) > i else 0.0,
                            )
                            num_faces_indexed += 1
                        except Exception as e:
                            logger.debug(f'Failed to index face {face_id}: {e}')

            # 6. Auto-detect and assign near-duplicates (runs in background, non-blocking)
            duplicate_info = None
            if detect_near_duplicates and ingest_result['global']:
                duplicate_info = await self._assign_to_duplicate_group(
                    image_id=image_id,
                    embedding=global_embedding,
                    threshold=near_duplicate_threshold,
                )

            response = {
                'status': 'success' if ingest_result['global'] else 'failed',
                'image_id': image_id,
                'num_detections': result['num_dets'],
                'num_faces': face_result.get('num_faces', 0),
                'embedding_norm': float(np.linalg.norm(global_embedding)),
                'imohash': image_hash,
                'indexed': {
                    'global': ingest_result['global'],
                    'vehicles': ingest_result['vehicles'],
                    'people': ingest_result['people'],
                    'faces': num_faces_indexed,
                    'skipped': ingest_result['skipped'],
                },
                'errors': ingest_result.get('errors', []),
            }

            if duplicate_info:
                response['near_duplicate'] = duplicate_info

            # 7. Run OCR and index text content (if enabled)
            ocr_info = None
            if enable_ocr:
                ocr_info = await self._process_ocr(
                    image_bytes=image_bytes,
                    image_id=image_id,
                    image_path=image_path,
                )
                if ocr_info:
                    response['ocr'] = ocr_info

            return response

        except Exception as e:
            logger.error(f'Failed to ingest image {image_id}: {e}')
            return {
                'status': 'error',
                'image_id': image_id,
                'error': str(e),
            }

    async def _assign_to_duplicate_group(
        self,
        image_id: str,
        embedding: np.ndarray,
        threshold: float = 0.99,
    ) -> dict[str, Any] | None:
        """
        Check for near-duplicates and assign to existing group or create new one.

        This runs automatically during ingestion to keep duplicate groups current.
        No periodic retraining needed - groups are updated in real-time.

        Algorithm:
        1. Search for similar images above threshold
        2. If found with existing group -> join that group
        3. If found without group -> create new group with both images
        4. If not found -> no action (unique image)

        Args:
            image_id: ID of newly ingested image
            embedding: CLIP embedding of the image
            threshold: Similarity threshold (default 0.99, matches Immich)

        Returns:
            Dict with duplicate info if assigned, None if unique
        """
        try:
            from src.services.duplicate_detection import DuplicateDetectionService

            dup_service = DuplicateDetectionService(self.opensearch)

            # Find near-duplicates (excluding self)
            duplicates = await dup_service.find_duplicates_by_embedding(
                embedding=embedding,
                threshold=threshold,
                max_results=10,
                exclude_image_id=image_id,
            )

            if not duplicates:
                return None  # Unique image, no duplicates

            # Check if any duplicate is already in a group
            existing_group_id = None
            for dup in duplicates:
                if dup.duplicate_group_id:
                    existing_group_id = dup.duplicate_group_id
                    break

            if existing_group_id:
                # Join existing group
                await self.opensearch.client.update(
                    index='visual_search_global',
                    id=image_id,
                    body={
                        'doc': {
                            'duplicate_group_id': existing_group_id,
                            'is_duplicate_primary': False,
                            'duplicate_score': duplicates[0].similarity,
                        }
                    },
                )
                logger.info(f'Added {image_id} to existing duplicate group {existing_group_id}')
                return {
                    'action': 'joined_group',
                    'group_id': existing_group_id,
                    'similarity': duplicates[0].similarity,
                    'matched_image': duplicates[0].image_id,
                }

            # Create new group with this image as primary and first duplicate
            best_match = duplicates[0]
            group_id = await dup_service.create_duplicate_group(
                primary_image_id=image_id,
                duplicate_image_ids=[best_match.image_id],
                duplicate_scores=[best_match.similarity],
            )
            logger.info(f'Created new duplicate group {group_id} for {image_id}')
            return {
                'action': 'created_group',
                'group_id': group_id,
                'similarity': best_match.similarity,
                'matched_image': best_match.image_id,
            }

        except Exception as e:
            # Don't fail ingestion if duplicate detection fails
            logger.warning(f'Near-duplicate detection failed for {image_id}: {e}')
            return None

    async def _process_ocr(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Process OCR for an image and index results.

        Non-blocking - failures don't affect main ingestion.

        Args:
            image_bytes: Raw image bytes
            image_id: Image identifier
            image_path: Optional file path

        Returns:
            OCR info dict or None if no text detected
        """
        try:
            from src.services.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()
            ocr_result = ocr_service.extract_text(image_bytes, filter_by_score=True)

            if ocr_result.get('status') != 'success' or ocr_result.get('num_texts', 0) == 0:
                return None  # No text detected or OCR failed

            # Index OCR results to OpenSearch
            full_text = ocr_service.get_full_text(ocr_result)
            await self.opensearch.index_ocr_results(
                image_id=image_id,
                image_path=image_path or image_id,
                texts=ocr_result['texts'],
                boxes=ocr_result['boxes'],
                boxes_normalized=ocr_result['boxes_normalized'],
                det_scores=ocr_result['det_scores'],
                rec_scores=ocr_result['rec_scores'],
                full_text=full_text,
            )

            logger.info(f'OCR indexed {ocr_result["num_texts"]} text regions for {image_id}')

            return {
                'num_texts': ocr_result['num_texts'],
                'full_text': full_text[:200]
                if len(full_text) > 200
                else full_text,  # Truncate for response
                'indexed': True,
            }

        except Exception as e:
            # Don't fail main ingestion if OCR fails
            logger.warning(f'OCR processing failed for {image_id}: {e}')
            return None
