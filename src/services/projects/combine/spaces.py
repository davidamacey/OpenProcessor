"""Which vectors a combine can carry: only those of a space the target also has.

A vector belongs to a space (``space_key`` = model and version). Two projects hold comparable
vectors only when both have a space with the same key, so a source's vector is copied exactly
when the target has a live space of that key for the same kind of vector (item, frame, box). The
rest are dropped and their items marked ``deferred`` for re-embedding (embeddings plan 5.4,
RD-33). Every field name comes from the registry through ``embedding_space``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.project_context import bind_project
from src.domain.embedding_spaces import EmbeddingSpaceError, ResolvedSpace, Target
from src.services.embedding_spaces.resolve import embedding_space


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


@dataclass(frozen=True)
class SpacePair:
    """A source's primary space of one kind of vector, and the target's space of the same key.

    ``target`` is ``None`` when the target has no live space with that key: the vector is not
    copied."""

    source: ResolvedSpace
    target: ResolvedSpace | None

    @property
    def copies(self) -> bool:
        return self.target is not None


@dataclass(frozen=True)
class SpaceCopy:
    """The pairs of one source: its item, frame (image) and box vectors."""

    item: SpacePair
    frame: SpacePair
    box: SpacePair


async def primary_space(record: ProjectRecord, target: Target) -> ResolvedSpace | None:
    """The project's primary space for ``target``; ``None`` when it has none."""
    with bind_project(record, read_only=True):
        try:
            return await embedding_space(target)
        except EmbeddingSpaceError:
            return None


async def _live_in(record: ProjectRecord, target: Target, key: str) -> ResolvedSpace | None:
    with bind_project(record, read_only=True):
        try:
            return await embedding_space(target, key)
        except EmbeddingSpaceError:
            return None


async def plan_space_copy(source: ProjectRecord, target: ProjectRecord) -> SpaceCopy:
    """What of ``source``'s vectors ``target`` can hold. A source without a primary space of a
    kind has no vectors of that kind to carry and fails the run closed."""
    pairs: dict[Target, SpacePair] = {}
    for kind in (Target.ITEM, Target.FRAME, Target.BOX):
        held = await primary_space(source, kind)
        if held is None:
            msg = f"project '{source.slug}' has no primary {kind.value} embedding space"
            raise EmbeddingSpaceError('embedding_space_not_found', msg, status=409)
        pairs[kind] = SpacePair(held, await _live_in(target, kind, held.space_key))
    return SpaceCopy(pairs[Target.ITEM], pairs[Target.FRAME], pairs[Target.BOX])


__all__ = ['SpaceCopy', 'SpacePair', 'plan_space_copy', 'primary_space']
