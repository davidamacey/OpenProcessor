"""Query-by-image face search and 1:N identification: detect, pick the ArcFace embeddings,
k-NN search the faces index. Returns plain dicts; the router owns the wire models."""

from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.exceptions import RequestRejectedError
from src.services.inference import InferenceService


if TYPE_CHECKING:
    from src.services.visual_search import VisualSearchService


def _query_face(face: dict[str, Any]) -> dict[str, Any]:
    return {
        'box': face['box'],
        'landmarks': face['landmarks'],
        'score': face['score'],
        'quality': face.get('quality'),
    }


def _match(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        'face_id': hit.get('face_id', hit.get('_id', '')),
        'image_id': hit.get('image_id', ''),
        'image_path': hit.get('image_path'),
        'score': hit.get('score', 0.0),
        'person_id': hit.get('person_id'),
        'person_name': hit.get('person_name'),
    }


async def _search(
    search_service: 'VisualSearchService', embedding: Any, top_k: int, min_score: float
) -> list[dict[str, Any]]:
    return await search_service.opensearch.search_faces(
        query_embedding=np.array(embedding), top_k=top_k, min_score=min_score
    )


async def search_by_image(
    search_service: 'VisualSearchService',
    image_bytes: bytes,
    *,
    face_index: int,
    top_k: int,
    min_score: float,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The query face (box, landmarks, score, quality) and its matches, each with the match
    fields plus ``box`` and ``confidence``."""
    detected = InferenceService().detect_faces(image_bytes, confidence=confidence)
    if detected['num_faces'] == 0:
        raise RequestRejectedError(400, 'No faces detected in query image')
    if face_index >= detected['num_faces']:
        raise RequestRejectedError(
            400, f'Face index {face_index} out of range (0-{detected["num_faces"] - 1})'
        )
    hits = await _search(search_service, detected['embeddings'][face_index], top_k, min_score)
    matches = [
        {**_match(h), 'box': h.get('box', [0, 0, 0, 0]), 'confidence': h.get('confidence', 0.0)}
        for h in hits
    ]
    return _query_face(detected['faces'][face_index]), matches


async def identify_by_image(
    search_service: 'VisualSearchService',
    image_bytes: bytes,
    *,
    top_k: int,
    threshold: float,
    confidence: float,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """For every detected face: its query face and its matches, best first."""
    detected = InferenceService().detect_faces(image_bytes, confidence=confidence)
    results = []
    for i in range(detected['num_faces']):
        hits = await _search(search_service, detected['embeddings'][i], top_k, threshold)
        results.append((_query_face(detected['faces'][i]), [_match(h) for h in hits]))
    return results
