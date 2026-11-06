"""Ingest dedup: the exact-content lookups against ``image_contents``.

:class:`IngestDedupMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides
``opensearch`` and ``config``. A content row without its images document (a crash between the
row and the document) is released and the content treated as new, so it can be ingested again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.optional_doc import get_doc_or_none
from src.core.logging import get_logger
from src.services.near_duplicates import content_key


if TYPE_CHECKING:
    from src.config import CurationConfig


logger = get_logger(__name__)


class IngestDedupMixin:
    """Dedup lookups of :class:`~src.services.curation.ingest.CurationIngestService`."""

    opensearch: Any
    config: CurationConfig

    async def _lookup_image(self, content_sha256: str) -> dict[str, Any] | None:
        """The stored image (``image_id``, ``image_path``, ``dataset_split``) that holds this
        content, or None."""
        return (await self._stored_holders([content_sha256]))[content_sha256]

    async def stored_image(self, image_id: str) -> dict[str, Any] | None:
        """The images doc ``image_id`` (``image_id``, ``image_path``, ``dataset_split``)
        read by id, so it sees a write the index has not refreshed yet; None when absent."""
        resp = await get_doc_or_none(self.opensearch, self.config.images_index, image_id)
        if resp is None:
            return None
        return _stored_view(image_id, resp['_source'])

    async def _check_duplicates(self, content_sha256s: list[str]) -> dict[str, str | None]:
        """Maps each content hash to the image id that holds it (or None)."""
        held = await self._stored_holders(content_sha256s)
        return {h: (doc['image_id'] if doc is not None else None) for h, doc in held.items()}

    async def _stored_holders(self, content_sha256s: list[str]) -> dict[str, dict[str, Any] | None]:
        """Maps each content hash to the images doc that holds it (or None).

        One query for the rows and one ``mget`` to confirm their documents exist; a row whose
        document is gone is released and answers None.
        """
        if not content_sha256s:
            return {}
        holders = await content_key.find_bound(content_sha256s)
        if not holders:
            return dict.fromkeys(content_sha256s)
        ids = sorted(set(holders.values()))
        resp = await self.opensearch.mget(index=self.config.images_index, body={'ids': ids})
        docs = {d['_id']: d.get('_source') or {} for d in resp.get('docs') or [] if d.get('found')}
        orphans = [i for i in ids if i not in docs]
        if orphans:
            logger.warning('ingest_content_row_without_image', n=len(orphans))
            await content_key.release_bound(orphans)
        return {
            h: _stored_view(holders[h], docs[holders[h]]) if holders.get(h) in docs else None
            for h in content_sha256s
        }


def _stored_view(image_id: str, source: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {'image_id': image_id}
    view.update({k: source[k] for k in ('image_path', 'dataset_split') if k in source})
    return view
