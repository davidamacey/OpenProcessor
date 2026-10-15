"""Batch and face ingestion."""

import functools
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService
from src.services.visual_search.batch_documents import build_documents
from src.services.visual_search.batch_inference import infer_batch


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


def _summary(*, total: int, processed: int, duplicates: int) -> dict[str, Any]:
    """The response when nothing was (or could be) indexed."""
    return {
        'status': 'success',
        'total': total,
        'processed': processed,
        'duplicates': duplicates,
        'errors_count': 0,
        'indexed': {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0, 'ocr': 0},
        'near_duplicates': 0,
    }


def _compute_hashes(images_bytes: list[bytes]) -> list[str]:
    """imohash of every image, computed in parallel."""
    import io
    from concurrent.futures import ThreadPoolExecutor

    import imohash

    def compute_hash(img_bytes: bytes) -> str:
        typed: str = imohash.hashfileobject(io.BytesIO(img_bytes)).hex()
        return typed

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(compute_hash, images_bytes))


def _split_duplicates(
    images_data: list[tuple[bytes, str, str | None]],
    hashes: list[str],
    existing_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[bytes, str, str | None]], list[int]]:
    """``(duplicates, images to process, their indices in the request)``."""
    duplicates: list[dict[str, Any]] = []
    to_process: list[tuple[bytes, str, str | None]] = []
    to_process_indices: list[int] = []
    for i, (img_data, image_hash) in enumerate(zip(images_data, hashes, strict=False)):
        existing = existing_map.get(image_hash)
        if existing:
            duplicates.append(
                {
                    'image_id': img_data[1],
                    'existing_image_id': existing.get('image_id'),
                    'imohash': image_hash,
                }
            )
        else:
            to_process.append(img_data)
            to_process_indices.append(i)
    return duplicates, to_process, to_process_indices


class IngestBatchMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

    if TYPE_CHECKING:
        # Implemented by IngestSingleMixin; declared so the mixin type-checks alone.
        async def _assign_to_duplicate_group(
            self, image_id: str, embedding: np.ndarray, threshold: float = 0.99
        ) -> dict[str, Any] | None: ...

    # =========================================================================
    # Batch Ingestion (HIGH THROUGHPUT - 300+ RPS)
    # =========================================================================

    async def ingest_batch(
        self,
        images_data: list[tuple[bytes, str, str | None]],
        skip_duplicates: bool = True,
        detect_near_duplicates: bool = True,
        near_duplicate_threshold: float = 0.99,
        enable_ocr: bool = True,
        enable_detection: bool = True,
        enable_faces: bool = True,
        enable_clip: bool = True,
        defer_heavy_ops: bool = False,
        _max_workers: int = 32,  # Reserved for future use (currently using adaptive worker count)
    ) -> dict[str, Any]:
        """
        Batch ingest multiple images with optimized parallel processing.

        Performance: 3-5x faster than individual /ingest calls.
        - Parallel hash computation
        - Batch duplicate checking (msearch - 10x faster)
        - Batch Triton inference (dynamic batching: 16-48 avg)
        - OpenSearch bulk indexing for images and faces
        - Parallel near-duplicate detection
        - Reduced HTTP overhead

        Target throughput: 300+ RPS with batch sizes of 32-64.

        Args:
            images_data: List of (image_bytes, image_id, image_path) tuples
            skip_duplicates: Skip processing if image hash exists
            detect_near_duplicates: Auto-assign to duplicate groups
            near_duplicate_threshold: Similarity threshold for grouping
            enable_ocr: Run OCR text extraction and indexing
            enable_detection: Run YOLO object detection
            enable_faces: Run face detection and embedding extraction
            enable_clip: Run MobileCLIP global image embedding
            defer_heavy_ops: If True, skip near-duplicate detection during high
                           throughput ingestion. These can be processed later
                           via /ingest/process-deferred when load is lower.
            max_workers: Max parallel threads for inference

        Returns:
            Summary with processed count, duplicates, and errors
        """
        import asyncio
        import time as _time

        from src.clients.fast_face_client import get_fast_face_client
        from src.clients.triton_client import get_sync_triton_client
        from src.config import get_settings

        if not images_data:
            return _summary(total=0, processed=0, duplicates=0)

        total = len(images_data)
        settings = get_settings()
        client = get_sync_triton_client(settings.triton_url)
        face_client = get_fast_face_client(settings.triton_url)

        ocr_service = None
        if enable_ocr:
            from src.services.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()

        # Step 1: Compute hashes in parallel
        hashes = _compute_hashes([img[0] for img in images_data])

        # Step 2: Check for duplicates if enabled (batch operation - 10x faster)
        _t_dup_start = _time.perf_counter()
        if skip_duplicates:
            # Use batch msearch for all hashes at once
            existing_map = await self.opensearch.check_duplicates_by_hash_batch(hashes)
            duplicates, to_process, to_process_indices = _split_duplicates(
                images_data, hashes, existing_map
            )
        else:
            duplicates = []
            to_process = images_data
            to_process_indices = list(range(len(images_data)))

        _dup_ms = (_time.perf_counter() - _t_dup_start) * 1000
        logger.info(f'[PROFILE] Duplicate check: {_dup_ms:.1f}ms for {len(hashes)} hashes')

        if not to_process:
            return {
                **_summary(total=total, processed=0, duplicates=len(duplicates)),
                'duplicate_details': duplicates,
            }

        # Step 3: Triton inference (pass raw JPEG bytes directly to unified pipeline)
        loop = asyncio.get_running_loop()

        # === PROFILING: Triton Inference ===
        _t_inference_start = _time.perf_counter()
        inference_results = await loop.run_in_executor(
            None,
            functools.partial(
                infer_batch,
                [img[0] for img in to_process],
                client=client,
                face_client=face_client,
                ocr_service=ocr_service,
                enable_detection=enable_detection,
                enable_clip=enable_clip,
                enable_faces=enable_faces,
                enable_ocr=enable_ocr,
            ),
        )
        _inference_ms = (_time.perf_counter() - _t_inference_start) * 1000
        _inference_per_img = _inference_ms / len(to_process) if to_process else 0
        logger.info(
            f'[PROFILE] Triton Inference: {_inference_ms:.1f}ms total, {_inference_per_img:.1f}ms/img'
        )

        # Step 4: Format documents for bulk indexing
        documents, face_documents, errors = build_documents(
            inference_results,
            to_process,
            hashes,
            to_process_indices,
            datetime.now(UTC).isoformat(),
        )

        # Step 5: Bulk index to OpenSearch (global, vehicles, people, faces, OCR)
        indexed = await self._bulk_index(documents, face_documents)
        indexed['ocr'] = 0
        if enable_ocr:
            indexed['ocr'] = await self._index_ocr_results(inference_results, to_process)

        # Step 6: Near-duplicate detection (parallel execution, can be deferred)
        near_duplicates: list[dict] = []
        deferred_count = 0
        if detect_near_duplicates and documents and not defer_heavy_ops:
            near_duplicates = await self._near_duplicates(documents, near_duplicate_threshold)
        elif detect_near_duplicates and documents and defer_heavy_ops:
            # Heavy operations deferred for later processing
            deferred_count = len(documents)
            logger.info(f'Deferred near-duplicate detection for {deferred_count} images')

        return {
            'status': 'success',
            'total': total,
            'processed': len(documents),
            'duplicates': len(duplicates),
            'errors_count': len(errors),
            'indexed': indexed,
            'near_duplicates': len(near_duplicates),
            'deferred_ops': deferred_count if defer_heavy_ops else 0,
            'duplicate_details': duplicates if duplicates else None,
            'error_details': errors if errors else None,
            'near_duplicate_details': near_duplicates if near_duplicates else None,
        }

    async def _bulk_index(
        self, documents: list[dict[str, Any]], face_documents: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Step 5: bulk index the image documents, then (10x faster than sequential) the faces."""
        indexed = {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0}
        if documents:
            bulk_result = await self.opensearch.bulk_ingest(documents, refresh=False)
            indexed = {
                'global': bulk_result.get('global', 0),
                'vehicles': bulk_result.get('vehicles', 0),
                'people': bulk_result.get('people', 0),
                'faces': 0,
            }
        if face_documents:
            face_result = await self.opensearch.bulk_index_faces(face_documents)
            indexed['faces'] = face_result.get('indexed', 0)
            if face_result.get('errors'):
                logger.debug(f'Face indexing errors: {face_result["errors"]}')
        return indexed

    async def _index_ocr_result(self, result: dict, img_id: str, img_path: str) -> bool:
        """Index one image's OCR output (already in its inference result)."""

        num_texts = result.get('num_texts', 0)
        if num_texts == 0:
            return False
        try:
            texts = result.get('texts', [])
            text_boxes = result.get('text_boxes', [])
            text_boxes_norm = result.get('text_boxes_normalized', [])
            det_scores = result.get('text_det_scores', [])
            rec_scores = result.get('text_rec_scores', [])

            # Convert numpy arrays to lists if needed
            if hasattr(text_boxes, 'tolist'):
                text_boxes = text_boxes.tolist()
            if hasattr(text_boxes_norm, 'tolist'):
                text_boxes_norm = text_boxes_norm.tolist()
            if hasattr(det_scores, 'tolist'):
                det_scores = det_scores.tolist()
            if hasattr(rec_scores, 'tolist'):
                rec_scores = rec_scores.tolist()

            full_text = ' '.join(texts) if texts else ''
            await self.opensearch.index_ocr_results(
                image_id=img_id,
                image_path=img_path,
                texts=texts,
                boxes=text_boxes,
                boxes_normalized=text_boxes_norm,
                det_scores=det_scores,
                rec_scores=rec_scores,
                full_text=full_text,
            )
            logger.info(f'OCR indexed {num_texts} text regions for {img_id}')
            return True
        except Exception as e:
            logger.debug(f'OCR index failed for {img_id}: {e}')
            return False

    async def _index_ocr_results(
        self,
        inference_results: list[dict],
        to_process: list[tuple[bytes, str, str | None]],
    ) -> int:
        """Step 5c: index every image's OCR output in parallel; returns how many indexed."""
        import asyncio

        ocr_tasks = [
            self._index_ocr_result(
                inference_results[i],
                to_process[i][1],
                to_process[i][2] or to_process[i][1],
            )
            for i in range(len(inference_results))
        ]
        ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        return sum(1 for r in ocr_results if r is True)

    async def _near_duplicates(
        self, documents: list[dict[str, Any]], threshold: float
    ) -> list[dict]:
        """Assign every indexed image to a duplicate group, all checks in parallel."""
        import asyncio

        async def check_near_dup(doc: dict) -> dict | None:
            dup_info = await self._assign_to_duplicate_group(
                image_id=doc['image_id'],
                embedding=doc['global_embedding'],
                threshold=threshold,
            )
            if dup_info:
                return {'image_id': doc['image_id'], **dup_info}
            return None

        dup_results = await asyncio.gather(
            *[check_near_dup(doc) for doc in documents],
            return_exceptions=True,
        )
        return [r for r in dup_results if isinstance(r, dict) and r]

    async def ingest_faces(
        self,
        image_bytes: bytes,
        image_id: str,
        image_path: str | None = None,
        person_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect faces in image and ingest with ArcFace embeddings.

        Pipeline:
        1. Run SCRFD face detection
        2. Extract ArcFace embeddings for each face
        3. Index faces to visual_search_faces

        Args:
            image_bytes: Raw JPEG/PNG bytes
            image_id: Unique identifier for the image
            image_path: Optional file path (for retrieval)
            person_name: Optional name/label for faces
            metadata: Optional metadata dictionary

        Returns:
            dict with status and face count
        """
        try:
            from src.clients.fast_face_client import get_fast_face_client
            from src.config import get_settings

            settings = get_settings()
            face_client = get_fast_face_client(settings.triton_url)

            # Run SCRFD face detection + Umeyama alignment + ArcFace embedding
            result = face_client.recognize(image_bytes, confidence=0.5)

            if result.get('num_faces', 0) == 0:
                return {
                    'status': 'success',
                    'image_id': image_id,
                    'num_faces': 0,
                    'indexed': 0,
                    'message': 'No faces detected',
                }

            num_faces = result['num_faces']
            faces = [
                {
                    'box': list(result['face_boxes'][i]),
                    'landmarks': list(result['face_landmarks'][i])
                    if result['face_landmarks']
                    else [0.0] * 10,
                    'score': float(result['face_scores'][i]),
                    'quality': float(result['face_quality'][i]) if result['face_quality'] else 0.0,
                }
                for i in range(num_faces)
            ]
            embeddings = result['face_embeddings']

            # Ingest to OpenSearch
            ingest_result = await self.opensearch.ingest_faces(
                image_id=image_id,
                image_path=image_path or image_id,
                faces=faces,
                embeddings=embeddings,
                person_name=person_name,
                metadata=metadata,
            )

            return {
                'status': 'success' if ingest_result['faces'] > 0 else 'failed',
                'image_id': image_id,
                'num_faces': result['num_faces'],
                'indexed': ingest_result['faces'],
                'errors': ingest_result.get('errors', []),
            }

        except Exception as e:
            logger.error(f'Failed to ingest faces from {image_id}: {e}')
            return {
                'status': 'error',
                'image_id': image_id,
                'error': str(e),
            }
