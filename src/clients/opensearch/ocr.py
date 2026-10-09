"""OCR index writes and text search."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opensearchpy.helpers import async_bulk

from src.clients.opensearch.names import IndexName


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = logging.getLogger(__name__)


class OcrMixin:
    """OCR index writes and text search."""

    client: AsyncOpenSearch
    embedding_dim: int

    # =========================================================================
    # OCR Index Operations
    # =========================================================================

    async def index_ocr_results(
        self,
        image_id: str,
        image_path: str,
        texts: list[str],
        boxes: list[list[float]],  # [N, 8] quad coordinates
        boxes_normalized: list[list[float]],  # [N, 4] axis-aligned normalized
        det_scores: list[float],
        rec_scores: list[float],
        full_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Index OCR results for an image.

        Args:
            image_id: Source image ID
            image_path: File path or URL
            texts: List of detected text strings
            boxes: List of 8-coord quadrilateral boxes
            boxes_normalized: List of 4-coord axis-aligned boxes [x1,y1,x2,y2]
            det_scores: Detection confidence scores
            rec_scores: Recognition confidence scores
            full_text: Combined text from all regions (for full-text search)
            metadata: Optional metadata

        Returns:
            Dict with indexing results
        """
        if not texts:
            return {'indexed': 0, 'errors': []}

        indexed_at = datetime.now(UTC).isoformat()
        docs = []

        for i, (text, box, box_norm, det_score, rec_score) in enumerate(
            zip(texts, boxes, boxes_normalized, det_scores, rec_scores, strict=False)
        ):
            if not text.strip():  # Skip empty text
                continue

            ocr_id = f'{image_id}_ocr_{i}'
            doc = {
                '_index': IndexName.OCR.value,
                '_id': ocr_id,
                '_source': {
                    'ocr_id': ocr_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'text': text,
                    'text_raw': text,
                    'full_text': full_text or '',  # Combined text for full-text search
                    'box': box,
                    'box_normalized': box_norm,
                    'det_score': float(det_score),
                    # None (never a negative sentinel) when recognition
                    # failed for this line -- text is already '' and
                    # skipped above in that case, but stay defensive.
                    'rec_score': None if rec_score is None else float(rec_score),
                    'metadata': metadata or {},
                    'indexed_at': indexed_at,
                },
            }
            docs.append(doc)

        if not docs:
            return {'indexed': 0, 'errors': []}

        try:
            success, errors = await async_bulk(self.client, docs, raise_on_error=False)
            return {
                'indexed': success,
                'errors': [str(e) for e in errors] if errors else [],
            }
        except Exception as e:
            logger.error(f'Failed to index OCR results: {e}')
            return {'indexed': 0, 'errors': [str(e)]}

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        min_score: float = 0.0,
        exact_match: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for images containing specific text.

        Args:
            query_text: Text to search for
            top_k: Maximum number of results
            min_score: Minimum match score (0-1)
            exact_match: If True, use exact keyword match instead of fuzzy

        Returns:
            List of matching OCR results with image info
        """
        try:
            if exact_match:
                # Exact keyword match
                query = {
                    'query': {
                        'term': {'text_raw': query_text},
                    },
                    'size': top_k,
                }
            else:
                # Fuzzy text match with trigram analyzer
                query = {
                    'query': {
                        'bool': {
                            'should': [
                                {'match': {'text': {'query': query_text, 'boost': 1.0}}},
                                {'match_phrase': {'text': {'query': query_text, 'boost': 2.0}}},
                            ],
                        },
                    },
                    'size': top_k,
                    'min_score': min_score if min_score > 0 else None,
                }

            # Remove None values
            if query.get('min_score') is None:
                query.pop('min_score', None)

            response = await self.client.search(index=IndexName.OCR.value, body=query)

            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit.get('_score', 0),
                    'image_id': hit['_source'].get('image_id'),
                    'image_path': hit['_source'].get('image_path'),
                    'text': hit['_source'].get('text'),
                    'box': hit['_source'].get('box'),
                    'box_normalized': hit['_source'].get('box_normalized'),
                    'det_score': hit['_source'].get('det_score'),
                    'rec_score': hit['_source'].get('rec_score'),
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f'OCR text search failed: {e}')
            return []

    async def get_ocr_for_image(self, image_id: str) -> list[dict[str, Any]]:
        """
        Get all OCR results for a specific image.

        Args:
            image_id: Image identifier

        Returns:
            List of OCR results for the image
        """
        try:
            response = await self.client.search(
                index=IndexName.OCR.value,
                body={
                    'query': {'term': {'image_id': image_id}},
                    'size': 1000,  # Get all text from image
                    'sort': [{'det_score': 'desc'}],
                },
            )

            return [
                {
                    'text': hit['_source'].get('text'),
                    'box': hit['_source'].get('box'),
                    'box_normalized': hit['_source'].get('box_normalized'),
                    'det_score': hit['_source'].get('det_score'),
                    'rec_score': hit['_source'].get('rec_score'),
                }
                for hit in response['hits']['hits']
            ]

        except Exception as e:
            logger.error(f'Failed to get OCR for image {image_id}: {e}')
            return []

    async def delete_ocr_for_image(self, image_id: str) -> int:
        """
        Delete all OCR results for a specific image.

        Args:
            image_id: Image identifier

        Returns:
            Number of deleted documents
        """
        try:
            response = await self.client.delete_by_query(
                index=IndexName.OCR.value,
                body={'query': {'term': {'image_id': image_id}}},
            )
            return response.get('deleted', 0)
        except Exception as e:
            logger.error(f'Failed to delete OCR for image {image_id}: {e}')
            return 0

    async def search_ocr(
        self, query_text: str, top_k: int = 10, min_score: float = 0.3
    ) -> list[dict[str, Any]]:
        """Images whose OCR text matches ``query_text``, best first (one hit
        per image). See :meth:`search_ocr_page`."""
        results, _total = await self.search_ocr_page(query_text, size=top_k, min_score=min_score)
        return results

    async def search_ocr_page(
        self,
        query_text: str,
        *,
        offset: int = 0,
        size: int = 10,
        min_score: float = 0.0,
        exact: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """One page of OCR matches and the number of distinct images that
        match. The best-matching text line represents each image.

        ``exact`` matches the whole recognised line verbatim (``text_raw``);
        otherwise a line matches on any word of the query (prefix/substring
        up to the n-gram width) and a verbatim line ranks first.
        """
        if exact:
            should: list[dict[str, Any]] = [{'term': {'text_raw': {'value': query_text}}}]
        else:
            should = [
                {'match': {'text': {'query': query_text, 'boost': 2.0}}},
                {'term': {'text_raw': {'value': query_text, 'boost': 3.0}}},
            ]
        body: dict[str, Any] = {
            'from': offset,
            'size': size,
            'query': {'bool': {'should': should, 'minimum_should_match': 1}},
            'collapse': {'field': 'image_id'},
            'aggs': {'images': {'cardinality': {'field': 'image_id'}}},
            '_source': [
                'image_id',
                'image_path',
                'text',
                'text_raw',
                'det_score',
                'rec_score',
                'box_normalized',
                'metadata',
            ],
        }
        if min_score > 0:
            body['min_score'] = min_score
        try:
            response = await self.client.search(index=IndexName.OCR.value, body=body)
        except Exception as e:
            logger.error(f'OCR search failed: {e}')
            return [], 0

        results = [
            {
                'image_id': hit['_source'].get('image_id', ''),
                'image_path': hit['_source'].get('image_path'),
                'score': hit.get('_score', 0.0),
                'text': hit['_source'].get('text', ''),
                'box_normalized': hit['_source'].get('box_normalized'),
                'det_score': hit['_source'].get('det_score'),
                'rec_score': hit['_source'].get('rec_score'),
                'metadata': hit['_source'].get('metadata'),
            }
            for hit in response.get('hits', {}).get('hits', [])
        ]
        total = int(response.get('aggregations', {}).get('images', {}).get('value', 0))
        return results, total
