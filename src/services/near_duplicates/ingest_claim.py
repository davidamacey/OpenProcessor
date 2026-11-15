"""One commit group's content claims and pending rows, in the caller's transaction.

Per photo an ingest used to open a transaction for its content claim and two more for the pending
near-duplicate rows (item level, image level); each ended in a COMMIT that waits for the WAL.
:func:`claim_and_mark_in_session` does it for a whole group: the ``imohash`` locks of every claim
in sorted order, one read of the stored rows, the verdict of each photo (stored now, held by
another image, refused for this photo alone), one insert, and the pending rows of the photos that
were stored. The caller owns the transaction (the group transaction of
:func:`~src.services.curation.ingest_index.commit_group` reads the placement in the same one) and
its COMMIT. The order contract of the per-photo path stands: a photo whose content another image
holds, or whose claim is refused, writes no pending row; the pending rows exist before the
documents they describe.

A group whose transaction fails as a whole (the database, not a photo) is rolled back and run
again photo by photo through :func:`claim_and_mark_each`, so one photo's failure still fails that
photo alone; the caller counts and logs every such fall-back
(``op_ingest_batch_fallback_total{stage="claim"}``). A pool timeout is not retried: the request
answers 503 ``database_busy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from src.services.near_duplicates import content_key, pending
from src.services.near_duplicates.content_key import ClaimOutcome, ContentClaim


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class PhotoWrite:
    """What one photo needs stored before its documents are written."""

    item_ids: Sequence[str]
    """The photo's items: marked pending (reason ``added``) when the photo is stored."""
    claim: ContentClaim | None
    """The content claim of a new image; ``None`` for an image that already has its document
    (no claim, and no image-level pending row)."""


def _stored(outcome: ClaimOutcome) -> bool:
    return outcome.holder is None and outcome.error is None


async def claim_and_mark_in_session(
    session: AsyncSession, project_id: uuid.UUID, photos: Sequence[PhotoWrite]
) -> list[ClaimOutcome]:
    """The whole group in the caller's transaction of ``project_id`` (see the module docstring).

    Photos of one group claim distinct ``imohash`` values (:func:`content_key.claim_in_session`)."""
    claimed = [(i, p.claim) for i, p in enumerate(photos) if p.claim is not None]
    outcomes = [ClaimOutcome() for _ in photos]
    if claimed:
        decided = await content_key.claim_in_session(
            session, project_id, [claim for _, claim in claimed]
        )
        for (i, _), outcome in zip(claimed, decided, strict=True):
            outcomes[i] = outcome
    stored = [photo for photo, outcome in zip(photos, outcomes, strict=True) if _stored(outcome)]
    items = sorted({item for photo in stored for item in photo.item_ids})
    images = sorted({photo.claim.image_id for photo in stored if photo.claim is not None})
    if items:
        await pending.enqueue_pending(session, project_id, 'item', items, 'added')
    if images:
        await pending.enqueue_pending(session, project_id, 'image', images, 'added')
    return outcomes


async def claim_and_mark_each(photos: Sequence[PhotoWrite]) -> list[ClaimOutcome]:
    """The per-photo path: each photo claims, then marks its pending rows, in transactions of its
    own, one photo after the other. A photo that fails (the database, or a claim the content key
    refuses) is that photo's outcome and releases a claim it had taken."""
    outcomes: list[ClaimOutcome] = []
    for photo in photos:
        claimed = False
        try:
            if photo.claim is not None:
                holder = await content_key.claim_bound(
                    photo.claim.imohash,
                    photo.claim.image_id,
                    photo.claim.image_path,
                    photo.claim.data,
                )
                if holder is not None:
                    outcomes.append(ClaimOutcome(holder=holder))
                    continue
                claimed = True
            await pending.enqueue_pending_bound('item', photo.item_ids, 'added')
            if photo.claim is not None:
                await pending.enqueue_pending_bound('image', [photo.claim.image_id], 'added')
            outcomes.append(ClaimOutcome())
        except (SQLAlchemyError, OSError, ValueError) as exc:
            if claimed and photo.claim is not None:
                await content_key.release_bound([photo.claim.image_id])
            outcomes.append(ClaimOutcome(error=exc))
    return outcomes


__all__ = [
    'PhotoWrite',
    'claim_and_mark_each',
    'claim_and_mark_in_session',
]
