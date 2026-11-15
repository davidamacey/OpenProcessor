"""A claim row and the subject it addresses (15a sections 2.2 and 3.4).

A claim is one human decision, appended and never changed. It names its subject by the photo's
key and the item's wire id with the geometry the person saw, so it finds its subject again after
every machine result was rebuilt. Time and ids are inputs: nothing here reads a clock.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.domain.claim_kinds import KINDS, ActorKind, ClaimKind, SubjectKind, actor_rank
from src.domain.claim_payloads import PAYLOAD_MODELS, Box4, Payload


_CHECK_TIME = datetime(1970, 1, 1, tzinfo=UTC)


class ClaimError(ValueError):
    """A claim that cannot be a row (a writer's bug; the row is never written)."""


@dataclass(frozen=True)
class Subject:
    """What a claim is about. ``bbox`` is the item or box geometry the person saw, ``machine_bbox``
    the detector's box at the time, ``frame`` the photo's decode frame when it was made."""

    kind: SubjectKind
    photo_key: int | None = None
    crop_id: str | None = None
    box_id: str | None = None
    bbox: Box4 | None = None
    machine_bbox: Box4 | None = None
    frame: int | None = None
    pair_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind in (SubjectKind.ITEM, SubjectKind.BOX):
            missing = [
                name
                for name in ('photo_key', 'crop_id', 'bbox', 'frame')
                if getattr(self, name) is None
            ]
            if missing:
                msg = f'a {self.kind.value} subject needs {missing}'
                raise ClaimError(msg)
        if self.kind is SubjectKind.BOX and self.box_id is None:
            msg = 'a box subject needs a box_id'
            raise ClaimError(msg)
        if self.kind is SubjectKind.PAIR and (self.pair_id is None or self.crop_id is None):
            msg = 'a pair subject needs both members'
            raise ClaimError(msg)


@dataclass(frozen=True)
class Claim:
    claim_id: int
    """The rising id: the order of claims within one rank."""
    at: datetime
    actor_kind: ActorKind
    actor: str
    op_id: str | None
    kind: ClaimKind
    subject: Subject
    payload: Payload
    withdraws_claim_id: int | None = None

    def __post_init__(self) -> None:
        info = KINDS[self.kind]
        if self.subject.kind is not info.subject_kind:
            msg = f'{self.kind.value} is about a {info.subject_kind.value}, not a {self.subject.kind.value}'
            raise ClaimError(msg)
        if not isinstance(self.payload, PAYLOAD_MODELS[self.kind]):
            msg = f'{self.kind.value} takes a {PAYLOAD_MODELS[self.kind].__name__} payload'
            raise ClaimError(msg)
        if (self.kind is ClaimKind.WITHDRAW) != (self.withdraws_claim_id is not None):
            msg = 'a withdraw names the claim it withdraws, and only a withdraw does'
            raise ClaimError(msg)

    @property
    def rank(self) -> int:
        return actor_rank(self.actor_kind)

    @property
    def order(self) -> tuple[int, int]:
        """Who made it before when: the sort key of every layer's deciding claim (B2)."""
        return (self.rank, self.claim_id)

    @property
    def import_id(self) -> str | None:
        """The import named by an import actor (``import:<id>``)."""
        if self.actor_kind is ActorKind.IMPORT and self.actor.startswith('import:'):
            return self.actor.removeprefix('import:')
        return None


@dataclass(frozen=True)
class ClaimDraft:
    """A claim before its row exists: what a write path decided to append. The repository gives it
    the rising ``claim_id`` and the time (:func:`commit`)."""

    actor_kind: ActorKind
    actor: str
    op_id: str | None
    kind: ClaimKind
    subject: Subject
    payload: Payload
    withdraws_claim_id: int | None = None

    def __post_init__(self) -> None:
        # The same checks as a stored claim: a draft that could not be a row never leaves a builder.
        Claim(
            0,
            _CHECK_TIME,
            self.actor_kind,
            self.actor,
            self.op_id,
            self.kind,
            self.subject,
            self.payload,
            self.withdraws_claim_id,
        )


def commit(draft: ClaimDraft, *, claim_id: int, at: datetime) -> Claim:
    """The row a repository inserts for ``draft``."""
    return Claim(
        claim_id,
        at,
        draft.actor_kind,
        draft.actor,
        draft.op_id,
        draft.kind,
        draft.subject,
        draft.payload,
        draft.withdraws_claim_id,
    )


def operation_key(claim: Claim | ClaimDraft) -> tuple[object, ...]:
    """The identity ``uq_claims_op_subject`` gives a claim: operation, kind and the whole subject.
    A chunk replayed after a crash meets the same keys and writes nothing twice (M4)."""
    s = claim.subject
    return (claim.op_id, claim.kind, s.photo_key, s.crop_id, s.box_id, s.pair_id)


def without_replays(existing: Iterable[Claim], drafts: Iterable[ClaimDraft]) -> list[ClaimDraft]:
    """The drafts whose operation key no live row of the log holds yet: what an
    ``INSERT ... ON CONFLICT DO NOTHING`` of the chunk would add."""
    held = {operation_key(c) for c in existing if c.kind is not ClaimKind.WITHDRAW and c.op_id}
    fresh: list[ClaimDraft] = []
    for draft in drafts:
        key = operation_key(draft)
        if draft.kind is ClaimKind.WITHDRAW or draft.op_id is None or key not in held:
            fresh.append(draft)
            held.add(key)
    return fresh
