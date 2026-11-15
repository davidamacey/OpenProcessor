"""Ingest dedup: the exact-content lookups against ``image_contents``.

:class:`IngestDedupMixin` is mixed into
:class:`~src.services.curation.ingest.CurationIngestService`, which provides
``opensearch`` and ``config``. A content row without its images document (a crash between the
row and the document) is released and the content treated as new, so it can be ingested again.

A stored image is a ``duplicate`` only when it is whole (:mod:`src.domain.image_completeness`):
every item its doc lists exists. One that is not is either repaired by this ingest (its writer
marked the failure, or it is older than :data:`REPAIR_AFTER_S`) or answered ``incomplete_write``
(a writer may still be writing it).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from src.clients.optional_doc import get_doc_or_none
from src.core.logging import get_logger
from src.domain.image_completeness import (
    INCOMPLETE_MARK_FIELD,
    INDEXED_AT_FIELD,
    ITEM_IDS_FIELD,
    listed_item_ids,
    repairable,
)
from src.services.curation.image_completeness import missing_items
from src.services.curation.ingest_models import ERROR_KIND_INCOMPLETE_WRITE, IngestResult
from src.services.curation.ingest_writer import REPAIR_AFTER_S
from src.services.curation.ops_metrics import count_whole_event
from src.services.near_duplicates import content_key


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config import CurationConfig


logger = get_logger(__name__)

Completeness = Literal['complete', 'repair', 'in_flight']


@dataclass(frozen=True)
class Resolved:
    """One file against the stored contents."""

    existing: dict[str, Any] | None
    """The images doc view (``image_id``, ``image_path``, ``dataset_split``) that holds these
    exact bytes, or None."""
    key: str
    """What a new image of these bytes derives its id from (:func:`content_key.content_image_id`)."""
    completeness: Completeness = 'complete'
    """Of ``existing``: ``complete`` (a duplicate), ``repair`` (an item is missing and no live
    writer can be writing it: ingest it again) or ``in_flight`` (an item is missing, a writer may
    still be writing it)."""
    listed: tuple[str, ...] = ()
    """The items ``existing`` lists, kept for a ``repair`` (the ones the rewrite no longer names are
    stale)."""


class IngestDedupMixin:
    """Dedup lookups of :class:`~src.services.curation.ingest.CurationIngestService`."""

    opensearch: Any
    config: CurationConfig

    async def _lookup_image(self, imohash: str, data: bytes) -> Resolved:
        """The stored image that holds these exact bytes (if any) and the key a new one uses."""
        return (await self.resolve_contents([(imohash, data)]))[0]

    async def stored_image(self, image_id: str) -> dict[str, Any] | None:
        """The images doc ``image_id`` (``image_id``, ``image_path``, ``dataset_split``)
        read by id, so it sees a write the index has not refreshed yet; None when absent."""
        resp = await get_doc_or_none(self.opensearch, self.config.images_index, image_id)
        if resp is None:
            return None
        return _stored_view(image_id, resp['_source'])

    async def resolve_contents(self, entries: Sequence[tuple[str, bytes]]) -> list[Resolved]:
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
            resp = await self.opensearch.mget(
                index=self.config.images_index,
                body={'ids': ids},
                _source_includes=[
                    'image_path',
                    'dataset_split',
                    ITEM_IDS_FIELD,
                    INDEXED_AT_FIELD,
                    INCOMPLETE_MARK_FIELD,
                ],
            )
            docs = {
                d['_id']: d.get('_source') or {} for d in resp.get('docs') or [] if d.get('found')
            }
            orphans = [i for i in ids if i not in docs]
            if orphans:
                logger.warning('ingest_content_row_without_image', n=len(orphans))
                await content_key.release_bound(orphans)
        gaps = await self._gaps(docs)
        now = datetime.now(UTC)
        return [
            self._resolved(v.holder, v.key, docs, gaps, now)
            if v.holder is not None and v.holder in docs
            else Resolved(None, v.key)
            for v in verdicts
        ]

    async def _gaps(self, docs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        """The listed items each stored image lacks (one ``mget`` for all of them)."""
        listed = {i: ids for i, src in docs.items() if (ids := listed_item_ids(src))}
        if not listed:
            return {}
        return await missing_items(self.opensearch, self.config.items_index, listed)

    @staticmethod
    def _resolved(
        holder: str,
        key: str,
        docs: dict[str, dict[str, Any]],
        gaps: dict[str, list[str]],
        now: datetime,
    ) -> Resolved:
        source = docs[holder]
        view = _stored_view(holder, source)
        if holder not in gaps:
            return Resolved(view, key)
        if repairable(source, now=now, repair_after_s=REPAIR_AFTER_S):
            return Resolved(view, key, 'repair', tuple(listed_item_ids(source) or ()))
        return Resolved(view, key, 'in_flight')


def held_back_result(
    resolved: Resolved, image_path: str, imohash: str, source_identifier: str | None
) -> IngestResult:
    """The answer for a photo whose stored image misses items a live writer may still be
    writing (``resolved.completeness == 'in_flight'``): not stored yet, send it again later."""
    count_whole_event('held_back')
    image_id = (resolved.existing or {}).get('image_id', '')
    return IngestResult(
        status='failed',
        image_id=image_id,
        image_path=image_path,
        source_identifier=source_identifier,
        imohash=imohash,
        error=f'image {image_id} is stored incomplete and may still be written; retry later',
        error_kind=ERROR_KIND_INCOMPLETE_WRITE,
    )


def _stored_view(image_id: str, source: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {'image_id': image_id}
    view.update({k: source[k] for k in ('image_path', 'dataset_split') if k in source})
    return view
