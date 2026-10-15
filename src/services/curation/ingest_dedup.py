"""Ingest dedup: the imohash lookups against the images index.

:class:`IngestDedupMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides
``opensearch`` and ``config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.config import CurationConfig


logger = get_logger(__name__)


class IngestDedupMixin:
    """Dedup lookups of :class:`~src.services.curation.ingest.CurationIngestService`."""

    opensearch: Any
    config: CurationConfig

    async def _lookup_image(self, image_hash: str) -> dict[str, Any] | None:
        """Term-query the images index on imohash. Returns the existing
        doc's ``image_id`` / ``image_path`` or None."""
        body = {
            'size': 1,
            'query': {'term': {'imohash': image_hash}},
            '_source': ['image_id', 'image_path', 'dataset_split'],
        }
        try:
            resp = await self.opensearch.search(index=self.config.images_index, body=body)
        except Exception as exc:
            logger.warning('ingest_dedup_query_failed', error=str(exc))
            return None
        hits = (resp.get('hits') or {}).get('hits') or []
        if not hits:
            return None
        source = hits[0].get('_source') or {}
        return source if source.get('image_id') else None

    async def _check_duplicate(self, image_hash: str) -> str | None:
        """Existing image_id for ``image_hash``, or None."""
        existing = await self._lookup_image(image_hash)
        return existing['image_id'] if existing is not None else None

    async def _check_duplicates_msearch(
        self,
        image_hashes: list[str],
    ) -> dict[str, str | None]:
        """Batch dedup via msearch. Maps imohash -> existing image_id (or None).

        Falls back to per-image lookup on an msearch failure so dedup is
        never silently skipped.
        """
        if not image_hashes:
            return {}
        body_lines: list[dict[str, Any]] = []
        for image_hash in image_hashes:
            body_lines.append({'index': self.config.images_index})
            body_lines.append(
                {'size': 1, 'query': {'term': {'imohash': image_hash}}, '_source': ['image_id']}
            )
        try:
            resp = await self.opensearch.msearch(body=body_lines)
        except Exception as exc:
            logger.warning('ingest_msearch_failed', error=str(exc), n=len(image_hashes))
            return {h: await self._check_duplicate(h) for h in image_hashes}
        out: dict[str, str | None] = {}
        responses = resp.get('responses', [])
        for image_hash, sub in zip(image_hashes, responses, strict=False):
            hits = ((sub or {}).get('hits') or {}).get('hits') or []
            out[image_hash] = (hits[0].get('_source') or {}).get('image_id') if hits else None
        return out
