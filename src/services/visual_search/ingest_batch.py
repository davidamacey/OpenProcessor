"""Batch and face ingestion."""

import logging
from typing import Any

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService


logger = logging.getLogger(__name__)


class IngestBatchMixin:
    opensearch: OpenSearchClient
    inference: InferenceService

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
        from concurrent.futures import ThreadPoolExecutor

        import imohash

        from src.clients.fast_face_client import get_fast_face_client
        from src.clients.triton_client import get_triton_client
        from src.config import get_settings

        if not images_data:
            return {
                'status': 'success',
                'total': 0,
                'processed': 0,
                'duplicates': 0,
                'errors_count': 0,
                'indexed': {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0, 'ocr': 0},
                'near_duplicates': 0,
            }

        total = len(images_data)
        settings = get_settings()
        client = get_triton_client(settings.triton_url)
        face_client = get_fast_face_client(settings.triton_url)

        ocr_service = None
        if enable_ocr:
            from src.services.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()

        # Step 1: Compute hashes in parallel
        def compute_hash(img_bytes: bytes) -> str:
            import io

            return imohash.hashfileobject(io.BytesIO(img_bytes)).hex()

        hashes = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            hashes = list(executor.map(compute_hash, [img[0] for img in images_data]))

        # Step 2: Check for duplicates if enabled (batch operation - 10x faster)
        import time as _time

        _t_dup_start = _time.perf_counter()
        duplicates = []
        to_process = []
        to_process_indices = []

        if skip_duplicates:
            # Use batch msearch for all hashes at once
            existing_map = await self.opensearch.check_duplicates_by_hash_batch(hashes)

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
        else:
            to_process = images_data
            to_process_indices = list(range(len(images_data)))

        _t_dup_end = _time.perf_counter()
        _dup_ms = (_t_dup_end - _t_dup_start) * 1000
        logger.info(f'[PROFILE] Duplicate check: {_dup_ms:.1f}ms for {len(hashes)} hashes')

        if not to_process:
            return {
                'status': 'success',
                'total': total,
                'processed': 0,
                'duplicates': len(duplicates),
                'errors_count': 0,
                'indexed': {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0, 'ocr': 0},
                'near_duplicates': 0,
                'duplicate_details': duplicates,
            }

        # Step 3: Triton inference (pass raw JPEG bytes directly to unified pipeline)
        import time as _time

        loop = asyncio.get_running_loop()
        from concurrent.futures import ThreadPoolExecutor

        # === PROFILING: Triton Inference ===
        _t_inference_start = _time.perf_counter()

        def run_single_unified(img_bytes: bytes) -> dict:
            """Run selective inference based on enabled pipelines."""
            try:
                result: dict = {
                    'num_dets': 0,
                    'num_faces': 0,
                    'num_texts': 0,
                    'global_embedding': [],
                }

                # Run YOLO + MobileCLIP if detection or CLIP is enabled
                if enable_detection or enable_clip:
                    yolo_result = client.infer_yolo_clip_cpu(img_bytes)
                    if enable_detection:
                        result.update(
                            {
                                'num_dets': yolo_result.get('num_dets', 0),
                                'boxes': yolo_result.get('boxes', []),
                                'scores': yolo_result.get('scores', []),
                                'classes': yolo_result.get('classes', []),
                                'normalized_boxes': yolo_result.get('normalized_boxes', []),
                                'box_embeddings': yolo_result.get('box_embeddings', []),
                            }
                        )
                    if enable_clip:
                        result['global_embedding'] = yolo_result.get('image_embedding', [])
                    # Always get orig_shape for image dimensions
                    result['orig_shape'] = yolo_result.get('orig_shape')

                # Run face detection + embedding extraction if faces is enabled
                if enable_faces:
                    face_result = face_client.recognize(img_bytes, confidence=0.5)
                    result.update(
                        {
                            'num_faces': face_result.get('num_faces', 0),
                            'face_boxes': face_result.get('face_boxes', []),
                            'face_landmarks': face_result.get('face_landmarks', []),
                            'face_scores': face_result.get('face_scores', []),
                            'face_embeddings': face_result.get('face_embeddings', []),
                            'face_quality': face_result.get('face_quality', []),
                        }
                    )

                # Run OCR text extraction if enabled (DF2: this used to be
                # skipped entirely, so num_texts stayed 0 and the OCR
                # indexing step below never had anything to index).
                if enable_ocr:
                    ocr_result = ocr_service.extract_text(img_bytes, filter_by_score=True)
                    if ocr_result.get('status') == 'success':
                        result.update(
                            {
                                'num_texts': ocr_result.get('num_texts', 0),
                                'texts': ocr_result.get('texts', []),
                                'text_boxes': ocr_result.get('boxes', []),
                                'text_boxes_normalized': ocr_result.get('boxes_normalized', []),
                                'text_det_scores': ocr_result.get('det_scores', []),
                                'text_rec_scores': ocr_result.get('rec_scores', []),
                            }
                        )

                return result
            except Exception as e:
                logger.error(f'Unified pipeline inference failed: {e}', exc_info=True)
                return {'error': str(e), 'num_dets': 0, 'num_faces': 0, 'num_texts': 0}

        def run_inference_batch(images_bytes: list[bytes]) -> list[dict]:
            """Run unified pipeline on batch of images in parallel."""
            # Limit concurrent Triton inference to prevent gRPC "too_many_pings" errors
            # Single shared gRPC connection can handle ~8 concurrent streams reliably
            inference_workers = min(8, len(images_bytes))
            with ThreadPoolExecutor(max_workers=inference_workers) as executor:
                return list(executor.map(run_single_unified, images_bytes))

        inference_results = await loop.run_in_executor(
            None,
            run_inference_batch,
            [img[0] for img in to_process],
        )

        _t_inference_end = _time.perf_counter()
        _inference_ms = (_t_inference_end - _t_inference_start) * 1000
        _inference_per_img = _inference_ms / len(to_process) if to_process else 0
        logger.info(
            f'[PROFILE] Triton Inference: {_inference_ms:.1f}ms total, {_inference_per_img:.1f}ms/img'
        )

        # Extract face results from unified response (already included)
        face_results = [
            {
                'num_faces': r.get('num_faces', 0),
                'face_embeddings': r.get('face_embeddings', []),
                'face_boxes': r.get('face_boxes', []),
                'face_scores': r.get('face_scores', []),
                'face_quality': r.get('face_quality', []),
                'face_landmarks': r.get('face_landmarks', []),
            }
            for r in inference_results
        ]

        # Step 4: Format documents for bulk indexing
        from datetime import UTC, datetime

        import numpy as np

        documents = []
        face_documents = []
        errors = []
        timestamp = datetime.now(UTC).isoformat()

        for _i, (result, face_result, data_tuple, orig_idx) in enumerate(
            zip(inference_results, face_results, to_process, to_process_indices, strict=False)
        ):
            img_bytes, image_id, image_path = data_tuple
            image_hash = hashes[orig_idx]

            if 'error' in result:
                errors.append(
                    {
                        'image_id': image_id,
                        'error': result['error'],
                    }
                )
                continue

            # Handle both unified (global_embedding) and legacy (image_embedding) responses
            global_embedding = np.array(
                result.get('global_embedding', result.get('image_embedding', []))
            )

            # Get image dimensions from response or parse from JPEG
            orig_shape = result.get('orig_shape')
            if orig_shape:
                width, height = orig_shape[1], orig_shape[0]
            else:
                # Parse from JPEG header if not in response
                from src.utils.affine import get_jpeg_dimensions_fast

                try:
                    width, height = get_jpeg_dimensions_fast(img_bytes)
                except Exception:
                    width, height = 0, 0

            doc = {
                'image_id': image_id,
                'image_path': image_path or image_id,
                'global_embedding': global_embedding,
                'imohash': image_hash,
                'file_size_bytes': len(img_bytes),
                'width': width,
                'height': height,
            }

            # Add box data if detections exist
            if result.get('num_dets', 0) > 0:
                doc['box_embeddings'] = np.array(result.get('box_embeddings', []))
                doc['normalized_boxes'] = np.array(result.get('normalized_boxes', []))
                doc['det_classes'] = result.get('classes', [])
                doc['det_scores'] = result.get('scores', [])

            documents.append(doc)

            # Collect face documents for bulk indexing
            if face_result.get('num_faces', 0) > 0:
                face_embeddings = face_result.get('face_embeddings', [])
                face_boxes = face_result.get('face_boxes', [])
                face_scores = face_result.get('face_scores', [])
                face_quality = face_result.get('face_quality', [])
                face_landmarks = face_result.get('face_landmarks', [])

                for j in range(min(len(face_embeddings), len(face_boxes))):
                    if len(face_embeddings[j]) > 0:
                        face_id = f'{image_id}_face_{j}'
                        face_documents.append(
                            {
                                'face_id': face_id,
                                'image_id': image_id,
                                'image_path': image_path or image_id,
                                'embedding': face_embeddings[j],
                                'box': face_boxes[j].tolist()
                                if hasattr(face_boxes[j], 'tolist')
                                else list(face_boxes[j]),
                                'landmarks': face_landmarks[j].tolist()
                                if hasattr(face_landmarks[j], 'tolist')
                                else list(face_landmarks[j])
                                if len(face_landmarks) > j
                                else [],
                                'confidence': float(face_scores[j])
                                if len(face_scores) > j
                                else 0.0,
                                'quality': float(face_quality[j]) if len(face_quality) > j else 0.0,
                                'indexed_at': timestamp,
                            }
                        )

        # Step 5: Bulk index to OpenSearch (global, vehicles, people)
        indexed = {'global': 0, 'vehicles': 0, 'people': 0, 'faces': 0}
        if documents:
            bulk_result = await self.opensearch.bulk_ingest(documents, refresh=False)
            indexed = {
                'global': bulk_result.get('global', 0),
                'vehicles': bulk_result.get('vehicles', 0),
                'people': bulk_result.get('people', 0),
                'faces': 0,
            }

        # Step 5b: Bulk index faces (10x faster than sequential)
        if face_documents:
            face_result = await self.opensearch.bulk_index_faces(face_documents)
            indexed['faces'] = face_result.get('indexed', 0)
            if face_result.get('errors'):
                logger.debug(f'Face indexing errors: {face_result["errors"]}')

        # Step 5c: OCR indexing (results already from unified pipeline)
        indexed['ocr'] = 0
        if enable_ocr:
            # OCR results are already in inference_results from the analyze pipeline
            async def index_ocr_from_unified(result: dict, img_id: str, img_path: str) -> bool:
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

            # Run all OCR indexing in parallel
            ocr_tasks = [
                index_ocr_from_unified(
                    inference_results[i],
                    to_process[i][1],
                    to_process[i][2] or to_process[i][1],
                )
                for i in range(len(inference_results))
            ]
            ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
            indexed['ocr'] = sum(1 for r in ocr_results if r is True)

        # Step 6: Near-duplicate detection (parallel execution, can be deferred)
        near_duplicates = []
        deferred_count = 0
        if detect_near_duplicates and documents and not defer_heavy_ops:
            # Run all near-duplicate checks in parallel for better throughput
            async def check_near_dup(doc: dict) -> dict | None:
                dup_info = await self._assign_to_duplicate_group(
                    image_id=doc['image_id'],
                    embedding=doc['global_embedding'],
                    threshold=near_duplicate_threshold,
                )
                if dup_info:
                    return {'image_id': doc['image_id'], **dup_info}
                return None

            dup_results = await asyncio.gather(
                *[check_near_dup(doc) for doc in documents],
                return_exceptions=True,
            )
            near_duplicates = [r for r in dup_results if r and not isinstance(r, Exception)]
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
