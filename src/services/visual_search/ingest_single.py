"""Single-image ingestion with category routing, near-duplicate marking and OCR."""

import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from src.clients.opensearch.client import OpenSearchClient
from src.config.faces import faces_enabled
from src.services.inference import InferenceService
from src.services.visual_search.batch_documents import landmarks_for


logger = logging.getLogger(__name__)


def _imohash(image_bytes: bytes) -> str:
    import io

    import imohash

    typed: str = imohash.hashfileobject(io.BytesIO(image_bytes)).hex()
    return typed


def _run_pipelines(
    image_bytes: bytes, *, enable_detection: bool, enable_clip: bool, enable_faces: bool
) -> tuple[dict, dict]:
    """Run the enabled pipelines in parallel; returns ``(yolo+clip result, face result)``."""
    from concurrent.futures import ThreadPoolExecutor

    from src.clients.fast_face_client import get_fast_face_client
    from src.clients.triton_client import get_sync_triton_client
    from src.config import get_settings

    settings = get_settings()
    client = get_sync_triton_client(settings.triton_url)

    result: dict = {'num_dets': 0, 'image_embedding': None}
    face_result: dict = {'num_faces': 0}

    futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        # YOLO detection + CLIP embedding (if either is enabled)
        if enable_detection or enable_clip:
            futures['yolo_clip'] = executor.submit(client.infer_yolo_clip_cpu, image_bytes)

        # Face detection + embedding (if enabled)
        if enable_faces:
            face_client = get_fast_face_client(settings.triton_url)
            futures['face'] = executor.submit(face_client.recognize, image_bytes, confidence=0.5)

        # Collect results
        if 'yolo_clip' in futures:
            result = futures['yolo_clip'].result()
        if 'face' in futures:
            face_result = futures['face'].result()
    return result, face_result


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
        6. Mark the image for the core grouping job and list its near-duplicate candidates
           (if enabled; the group appears once ``near_dup_core_group`` has run)
        7. Run OCR and index text content (if enable_ocr=True)

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            metadata: Optional metadata dictionary
            skip_duplicates: If True, skip processing if image hash already exists.
                           Set to False for benchmarks. Default: True (production).
            detect_near_duplicates: If True, mark the image for the single-writer core
                           grouping job and list its near-duplicate candidates. Default: True.
            enable_ocr: If True, run OCR and index detected text.
                           Default: True.

        Returns:
            dict with status, routing info, and counts per category
        """
        # Imported here: the near-duplicate services import the app's dependency module, which
        # imports this one.
        from src.services.near_duplicates import core_candidates, core_pending

        enable_faces = enable_faces and faces_enabled()
        try:
            # 1. Compute imohash for duplicate detection
            image_hash = _imohash(image_bytes)
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

            result, face_result = _run_pipelines(
                image_bytes,
                enable_detection=enable_detection,
                enable_clip=enable_clip,
                enable_faces=enable_faces,
            )

            # Extract global embedding if CLIP is enabled
            if enable_clip and result.get('image_embedding') is not None:
                global_embedding = np.array(result['image_embedding'])
            else:
                # Create zero embedding if CLIP disabled (still need for indexing structure)
                global_embedding = np.zeros(512, dtype=np.float32)

            # Box data (only if detection is enabled)
            box_embeddings = None
            normalized_boxes = None
            det_classes: list[int] | None = None
            det_scores: list[float] | None = None

            if enable_detection and result['num_dets'] > 0:
                box_embeddings = np.array(result.get('box_embeddings', []))
                normalized_boxes = np.array(result.get('normalized_boxes', []))
                det_classes = result.get('classes', [])
                det_scores = result.get('scores', [])

            # The pending row goes before the index write it describes (an extra row costs one
            # idempotent rescore; a missing one would leave a stale group nobody knows about).
            if detect_near_duplicates:
                await core_pending.record_pending([image_id], 'added')

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
                num_faces_indexed = await self._index_faces(
                    face_result, image_id, image_path or image_id
                )

            candidates = (
                await core_candidates.find_candidates(
                    self.opensearch.client, {image_id: global_embedding}
                )
                if detect_near_duplicates and ingest_result['global']
                else {}
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

            if candidates.get(image_id):
                response['near_duplicate_candidates'] = [asdict(c) for c in candidates[image_id]]

            # 7. Run OCR and index text content (if enabled)
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

    async def _index_faces(self, face_result: dict, image_id: str, image_path: str) -> int:
        """Index each detected face; one that fails to index is skipped, not fatal."""
        face_embeddings: list = face_result.get('face_embeddings', [])
        face_boxes: list = face_result.get('face_boxes', [])
        face_scores: list = face_result.get('face_scores', [])
        face_quality: list = face_result.get('face_quality', [])
        face_landmarks: list = face_result.get('face_landmarks', [])

        indexed = 0
        for i in range(min(len(face_embeddings), len(face_boxes))):
            if len(face_embeddings[i]) > 0:
                face_id = f'{image_id}_face_{i}'
                if face_landmarks and i >= len(face_landmarks):
                    logger.warning(
                        f'Not indexing {face_id}: face_landmarks_length_mismatch: '
                        f'{len(face_landmarks)} landmark entries for face {i}'
                    )
                    continue
                try:
                    await self.opensearch.index_face(
                        face_id=face_id,
                        image_id=image_id,
                        image_path=image_path,
                        embedding=face_embeddings[i],
                        box=face_boxes[i].tolist()
                        if hasattr(face_boxes[i], 'tolist')
                        else list(face_boxes[i]),
                        landmarks=landmarks_for(face_landmarks, i),
                        confidence=float(face_scores[i]) if len(face_scores) > i else 0.0,
                        quality=float(face_quality[i]) if len(face_quality) > i else 0.0,
                    )
                    indexed += 1
                except Exception as e:
                    logger.debug(f'Failed to index face {face_id}: {e}')
        return indexed

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
