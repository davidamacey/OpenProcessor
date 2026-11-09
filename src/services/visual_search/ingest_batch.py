"""Batch and face ingestion."""

import functools
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from src.clients.opensearch.client import OpenSearchClient
from src.services.inference import InferenceService
from src.services.near_duplicates import core_candidates
from src.services.visual_search.batch_documents import build_documents
from src.services.visual_search.batch_inference import infer_batch
from src.utils.task_group import run_settled


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
    from concurrent.futures import ThreadPoolExecutor

    from src.services.near_duplicates.content_key import content_imohash

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(content_imohash, images_bytes))


def _split_duplicates(
    images_data: list[tuple[bytes, str, str | None]],
    hashes: list[str],
    holders: list[Any],
) -> tuple[list[dict[str, Any]], list[tuple[bytes, str, str | None]], list[int]]:
    """``(duplicates, images to process, their indices in the request)``; ``holders`` holds, per
    image, the stored image with the same bytes or ``None``."""
    duplicates: list[dict[str, Any]] = []
    to_process: list[tuple[bytes, str, str | None]] = []
    to_process_indices: list[int] = []
    for i, (img_data, image_hash, holder) in enumerate(
        zip(images_data, hashes, holders, strict=True)
    ):
        if holder is not None:
            duplicates.append(
                {
                    'image_id': img_data[1],
                    'existing_image_id': holder.image_id,
                    'imohash': image_hash,
                }
            )
        else:
            to_process.append(img_data)
            to_process_indices.append(i)
    return duplicates, to_process, to_process_indices


async def _claim_contents(
    documents: list[dict[str, Any]],
    face_documents: list[dict[str, Any]],
    bytes_by_id: dict[str, bytes],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Store the exact content of each document in one transaction, before its images document.

    Returns ``(documents, face documents, duplicates, errors)`` without the images that lost:
    bytes a concurrent request (or an earlier image of this batch) stored first are duplicates of
    that image, and an image id that already holds other bytes is refused.
    """
    from src.services.near_duplicates import content_key

    claims = await content_key.claim_many_core(
        [
            content_key.CoreEntry(
                d['imohash'], d['image_id'], d['image_path'], bytes_by_id[d['image_id']]
            )
            for d in documents
        ]
    )
    lost = claims.duplicates.keys() | claims.conflicts.keys()
    duplicates = [
        {
            'image_id': image_id,
            'existing_image_id': verdict.holder,
            'imohash': next(d['imohash'] for d in documents if d['image_id'] == image_id),
        }
        for image_id, verdict in claims.duplicates.items()
    ]
    errors = [{'image_id': i, 'error': why} for i, why in claims.conflicts.items()]
    return (
        [d for d in documents if d['image_id'] not in lost],
        [f for f in face_documents if f['image_id'] not in lost],
        duplicates,
        errors,
    )


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
        - Near-duplicate grouping off the request: one pending row per image, grouped by the
          single-writer ``near_dup_core_group`` job; the response lists the candidates
        - Reduced HTTP overhead

        Target throughput: 300+ RPS with batch sizes of 32-64.

        Args:
            images_data: List of (image_bytes, image_id, image_path) tuples
            skip_duplicates: Skip processing if image hash exists
            detect_near_duplicates: Mark the indexed images for the core grouping job and list
                           their near-duplicate candidates
            enable_ocr: Run OCR text extraction and indexing
            enable_detection: Run YOLO object detection
            enable_faces: Run face detection and embedding extraction
            enable_clip: Run MobileCLIP global image embedding
            defer_heavy_ops: If True, skip the candidate lookup (reported as deferred). Grouping
                           is done by the ``near_dup_core_group`` job either way.
            max_workers: Max parallel threads for inference

        Returns:
            Summary with processed count, duplicates, and errors
        """
        import asyncio
        import time as _time

        from src.clients.fast_face_client import get_fast_face_client
        from src.clients.triton_client import get_sync_triton_client
        from src.config import get_settings
        from src.config.faces import faces_enabled
        from src.services.near_duplicates import core_content, core_pending

        enable_faces = enable_faces and faces_enabled()

        if not images_data:
            return _summary(total=0, processed=0, duplicates=0)

        total = len(images_data)
        settings = get_settings()
        client = get_sync_triton_client(settings.triton_url)
        face_client = get_fast_face_client(settings.triton_url) if enable_faces else None

        ocr_service = None
        if enable_ocr:
            from src.services.ocr_service import get_ocr_service

            ocr_service = get_ocr_service()

        # Step 1: Compute hashes in parallel
        hashes = _compute_hashes([img[0] for img in images_data])

        # Step 2: Check for duplicates if enabled (batch operation - 10x faster)
        _t_dup_start = _time.perf_counter()
        if skip_duplicates:
            # imohash finds the stored image, SHA-256 confirms (one query for the batch)
            holders = await core_content.resolve_holders(
                self.opensearch.client,
                [(h, img[0]) for img, h in zip(images_data, hashes, strict=True)],
            )
            duplicates, to_process, to_process_indices = _split_duplicates(
                images_data, hashes, holders
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

        # Step 4b: Store each image's exact content, one transaction for the batch. An image whose
        # bytes a concurrent request stored first is a duplicate of that image, not indexed twice.
        claimed_away: set[str] = set()
        if skip_duplicates and documents:
            offered = {d['image_id'] for d in documents}
            documents, face_documents, raced, refused = await _claim_contents(
                documents, face_documents, {img[1]: img[0] for img in to_process}
            )
            duplicates += raced
            errors += refused
            claimed_away = offered - {d['image_id'] for d in documents}

        # Step 5: Mark the images for the core grouping job, then bulk index to OpenSearch
        # (global, vehicles, people, faces, OCR). The pending row goes first: an extra row costs
        # one idempotent rescore, a missing one would leave a stale group nobody knows about.
        deferred_count = 0
        if detect_near_duplicates and documents:
            await core_pending.record_pending([doc['image_id'] for doc in documents], 'added')
            if defer_heavy_ops:
                deferred_count = len(documents)
        indexed = await self._bulk_index(documents, face_documents)
        indexed['ocr'] = 0
        if enable_ocr:
            kept = [i for i, img in enumerate(to_process) if img[1] not in claimed_away]
            indexed['ocr'] = await self._index_ocr_results(
                [inference_results[i] for i in kept], [to_process[i] for i in kept]
            )

        # Step 6: The near-duplicate candidates (a read; the group appears once the job has run).
        near_duplicates: list[dict[str, Any]] = []
        if detect_near_duplicates and documents and not defer_heavy_ops:
            found = await core_candidates.find_candidates(
                self.opensearch.client, {d['image_id']: d['global_embedding'] for d in documents}
            )
            near_duplicates = [
                {'image_id': image_id, 'candidates': [asdict(c) for c in items]}
                for image_id, items in found.items()
            ]

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
            'near_duplicate_candidates': near_duplicates or None,
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
        ocr_results = await run_settled(
            [
                self._index_ocr_result(
                    inference_results[i],
                    to_process[i][1],
                    to_process[i][2] or to_process[i][1],
                )
                for i in range(len(inference_results))
            ]
        )
        return sum(1 for r in ocr_results if r is True)
