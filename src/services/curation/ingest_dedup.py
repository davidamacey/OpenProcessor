"""Ingest dedup: the exact-content lookups against ``image_contents``.

:class:`IngestDedupMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides
``opensearch`` and ``config``. A content row without its images document (a crash between the
row and the document) is released and the content treated as new, so it can be ingested again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.clients.optional_doc import get_doc_or_none
from src.core.logging import get_logger
from src.services.near_duplicates import content_key


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config import CurationConfig


logger = get_logger(__name__)


@dataclass(frozen=True)
class Resolved:
    """One file against the stored contents."""

    existing: dict[str, Any] | None
    """The images doc view (``image_id``, ``image_path``, ``dataset_split``) that holds these
    exact bytes, or None."""
    key: str
    """What a new image of these bytes derives its id from (:func:`content_key.content_image_id`)."""


class IngestDedupMixin:
    """Dedup lookups of :class:`~src.services.curation.ingest.CurationIngestService`."""

    opensearch: Any
    config: CurationConfig

    async def _lookup_image(self, imohash: str, data: bytes) -> Resolved:
        """The stored image that holds these exact bytes (if any) and the key a new one uses."""
        return (await self._resolve([(imohash, data)]))[0]

    async def stored_image(self, image_id: str) -> dict[str, Any] | None:
        """The images doc ``image_id`` (``image_id``, ``image_path``, ``dataset_split``)
        read by id, so it sees a write the index has not refreshed yet; None when absent."""
        resp = await get_doc_or_none(self.opensearch, self.config.images_index, image_id)
        if resp is None:
            return None
        return _stored_view(image_id, resp['_source'])

    async def _check_duplicates(self, entries: Sequence[tuple[str, bytes]]) -> list[str | None]:
        """The image id that holds each ``(imohash, bytes)`` file's content, or None."""
        resolved = await self._resolve(entries)
        return [r.existing['image_id'] if r.existing is not None else None for r in resolved]

    async def _resolve(self, entries: Sequence[tuple[str, bytes]]) -> list[Resolved]:
        """Each ``(imohash, bytes)`` file against the stored contents.

        One query finds the imohashes the project stores; only those files are compared by
        SHA-256 (:mod:`~src.services.near_duplicates.content_key`), and one ``mget`` confirms the
        holders' documents exist. A row whose document is gone is released and answers None.
        """
        if not entries:
            return []
        verdicts = await content_key.resolve_many_bound(entries)
        ids = sorted({v.holder for v in verdicts if v.holder is not None})
        docs: dict[str, dict[str, Any]] = {}
        if ids:
            resp = await self.opensearch.mget(index=self.config.images_index, body={'ids': ids})
            docs = {
                d['_id']: d.get('_source') or {} for d in resp.get('docs') or [] if d.get('found')
            }
            orphans = [i for i in ids if i not in docs]
            if orphans:
                logger.warning('ingest_content_row_without_image', n=len(orphans))
                await content_key.release_bound(orphans)
        return [
            Resolved(
                _stored_view(v.holder, docs[v.holder])
                if v.holder is not None and v.holder in docs
                else None,
                v.key,
            )
            for v in verdicts
        ]


def _stored_view(image_id: str, source: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {'image_id': image_id}
    view.update({k: source[k] for k in ('image_path', 'dataset_split') if k in source})
    return view
