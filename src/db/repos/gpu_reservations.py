"""``gpu_reservations``: which job holds which card, keyed by the card's UUID (research S-7).

A card is held by at most one job (the primary key). ``reserve`` upserts the card's
``gpu_devices`` row in the same transaction, so a card with no GPU worker of its own can still be
reserved. Every function takes the caller's sync ``Session`` inside a scoped transaction and
never commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from src.db.repos import gpu_devices


if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from sqlalchemy.orm import Session


class CardHeldError(RuntimeError):
    """The card is held by another job."""

    def __init__(self, gpu_uuid: str) -> None:
        super().__init__(f'GPU {gpu_uuid} is reserved by another job')
        self.gpu_uuid = gpu_uuid


_INSERT = text(
    'INSERT INTO gpu_reservations (gpu_uuid, holder_job_id) VALUES (:card, :job) '
    'ON CONFLICT (gpu_uuid) DO NOTHING RETURNING gpu_uuid'
)
_MINE = text('SELECT 1 FROM gpu_reservations WHERE gpu_uuid = :card AND holder_job_id = :job')


def reserve(session: Session, job_id: uuid.UUID, cards: Sequence[gpu_devices.Card]) -> None:
    """Hold every card for ``job_id``; all or none (the caller's transaction rolls back on error).

    A repeat by the job that already holds a card is a no-op, so a redelivered job does not fail
    on its own reservation. ``CardHeldError`` names the first card another job holds.
    """
    gpu_devices.upsert(session, cards)
    for card in cards:
        params = {'card': card.gpu_uuid, 'job': job_id}
        if (
            session.execute(_INSERT, params).first() is None
            and session.execute(_MINE, params).first() is None
        ):
            raise CardHeldError(card.gpu_uuid)


def release(session: Session, job_id: uuid.UUID) -> int:
    """Free every card the job holds; returns how many."""
    freed = session.execute(
        text('DELETE FROM gpu_reservations WHERE holder_job_id = :job RETURNING gpu_uuid'),
        {'job': job_id},
    )
    return len(freed.all())
