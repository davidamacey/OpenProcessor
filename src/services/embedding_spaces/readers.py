"""What a vector reader needs from the registry, in the shape its call site wants.

Every function answers for the bound project and fails closed (``EmbeddingSpaceError``) when it
has no such space. Clients never import this module: a service or router calls it and passes the
field names down (``src/clients`` may not import ``src.services``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.services.embedding_spaces.resolve import embedding_space, live_spaces


async def item_vector_field() -> str:
    """The vector field of the bound project's primary item space."""
    return (await embedding_space(Target.ITEM)).vector_field


async def vector_source_excludes(index_role: str = 'items') -> list[str]:
    """Every vector of one of the bound project's indexes, for a ``_source`` exclude list.

    A space's nested path stands for its whole box list, so one entry drops the nested vectors;
    the result is built from every live space, so a flipped or optional space is excluded too.
    ``index_role`` is ``items`` or ``images``.
    """
    return sorted(
        {
            (s.nested_path or s.vector_field)
            for s in await live_spaces()
            if s.index_role == index_role
        }
    )


@dataclass(frozen=True)
class WriteFields:
    """The physical fields a writer stores an item's vector and its state under."""

    vector_field: str
    state_field: str


async def item_write_fields() -> WriteFields:
    """Where the bound project's primary item space keeps a vector and its ``embedding_state``."""
    space = await embedding_space(Target.ITEM)
    if space.state_field is None:  # unreachable: an item space always has a state field
        raise EmbeddingSpaceError(
            'embedding_space_not_found',
            f'item space {space.space_key} has no state field',
            status=500,
        )
    return WriteFields(space.vector_field, space.state_field)


async def item_state_fields() -> dict[str, str]:
    """``{space_key: state field}`` of every live item space of the bound project.

    The item wire reports ``embedding_states`` for each of them (a flipped or optional space
    keeps its own state field), so a serializer is handed this map by the route that reads the
    items.
    """
    return {
        s.space_key: s.state_field
        for s in await live_spaces()
        if s.target is Target.ITEM and s.state_field is not None
    }


async def frame_vector_field() -> str:
    """The vector field of the bound project's primary frame space (the images index)."""
    return (await embedding_space(Target.FRAME)).vector_field
