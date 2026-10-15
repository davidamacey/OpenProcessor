"""OpenSearch documents for a batch: one image document per successful
inference (with its boxes) and one document per detected face."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.face_response import face_landmark_mismatch


def _image_size(result: dict, img_bytes: bytes) -> tuple[int, int]:
    """Width and height from the inference response, else from the JPEG header."""
    orig_shape = result.get('orig_shape')
    if orig_shape:
        return orig_shape[1], orig_shape[0]
    from src.utils.affine import get_jpeg_dimensions_fast

    try:
        return get_jpeg_dimensions_fast(img_bytes)
    except Exception:
        return 0, 0


def _image_document(
    result: dict, img_bytes: bytes, image_id: str, image_path: str | None, image_hash: str
) -> dict[str, Any]:
    # Handle both unified (global_embedding) and legacy (image_embedding) responses
    global_embedding = np.array(result.get('global_embedding', result.get('image_embedding', [])))
    width, height = _image_size(result, img_bytes)
    doc: dict[str, Any] = {
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
    return doc


def landmarks_for(face_landmarks: list, j: int) -> list:
    if len(face_landmarks) <= j:
        return []
    entry = face_landmarks[j]
    return entry.tolist() if hasattr(entry, 'tolist') else list(entry)


def _face_documents(
    face_result: dict, image_id: str, image_path: str | None, timestamp: str
) -> list[dict[str, Any]]:
    face_embeddings = face_result.get('face_embeddings', [])
    face_boxes = face_result.get('face_boxes', [])
    face_scores = face_result.get('face_scores', [])
    face_quality = face_result.get('face_quality', [])
    face_landmarks = face_result.get('face_landmarks', [])

    docs: list[dict[str, Any]] = []
    for j in range(min(len(face_embeddings), len(face_boxes))):
        if len(face_embeddings[j]) > 0:
            face_id = f'{image_id}_face_{j}'
            docs.append(
                {
                    'face_id': face_id,
                    'image_id': image_id,
                    'image_path': image_path or image_id,
                    'embedding': face_embeddings[j],
                    'box': face_boxes[j].tolist()
                    if hasattr(face_boxes[j], 'tolist')
                    else list(face_boxes[j]),
                    'landmarks': landmarks_for(face_landmarks, j),
                    'confidence': float(face_scores[j]) if len(face_scores) > j else 0.0,
                    'quality': float(face_quality[j]) if len(face_quality) > j else 0.0,
                    'indexed_at': timestamp,
                }
            )
    return docs


def build_documents(
    inference_results: list[dict],
    to_process: list[tuple[bytes, str, str | None]],
    hashes: list[str],
    to_process_indices: list[int],
    timestamp: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(image documents, face documents, per-image errors)`` for the batch."""
    documents: list[dict[str, Any]] = []
    face_documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for result, data_tuple, orig_idx in zip(
        inference_results, to_process, to_process_indices, strict=False
    ):
        img_bytes, image_id, image_path = data_tuple
        if 'error' in result:
            errors.append({'image_id': image_id, 'error': result['error']})
            continue
        # The face fields ride on the same inference response. An image whose
        # face response is unusable is refused whole: indexing it without its
        # faces would look complete.
        has_faces = result.get('num_faces', 0) > 0
        if has_faces and (reason := face_landmark_mismatch(result)):
            errors.append({'image_id': image_id, 'error': reason})
            continue
        documents.append(_image_document(result, img_bytes, image_id, image_path, hashes[orig_idx]))
        if has_faces:
            face_documents.extend(_face_documents(result, image_id, image_path, timestamp))
    return documents, face_documents, errors
