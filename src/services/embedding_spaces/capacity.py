"""Does the OpenSearch data path have room for a new vector field?

A new space writes ``dim`` float32 per document plus the HNSW graph; the check asks for twice the
plain vector bytes. It fails closed: a cluster that cannot answer refuses the start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from src.domain.embedding_spaces import EmbeddingSpaceError


if TYPE_CHECKING:
    from src.domain.embedding_spaces import Target

BYTES_PER_FLOAT = 4
HEADROOM_FACTOR = 2


class SpaceCapacity(Protocol):
    async def doc_count(self, target: Target) -> int: ...

    async def free_bytes(self) -> int:
        """The smallest free space over the data nodes' data paths."""
        ...


def bytes_estimate(docs: int, dim: int) -> int:
    return docs * dim * BYTES_PER_FLOAT


async def require_room(capacity: SpaceCapacity, target: Target, dim: int) -> int:
    """The estimated bytes of the new field; 409 ``embedding_insufficient_storage`` if short."""
    try:
        estimate = bytes_estimate(await capacity.doc_count(target), dim)
        free = await capacity.free_bytes()
    except Exception as exc:  # fail closed: an unanswered question is a refusal
        raise EmbeddingSpaceError(
            'embedding_capacity_unknown', f'cannot read the index size or free space: {exc}'
        ) from exc
    if free < HEADROOM_FACTOR * estimate:
        raise EmbeddingSpaceError(
            'embedding_insufficient_storage',
            f'the new field needs about {estimate} bytes and the data path has {free} free; '
            f'{HEADROOM_FACTOR}x the estimate is required',
        )
    return estimate


class OpenSearchCapacity:
    """``SpaceCapacity`` over the project-guarded client and the bound project's index names."""

    def __init__(self, client: Any, items_index: str, images_index: str) -> None:
        self._client = client
        self._indexes = {'frame': images_index}
        self._items = items_index

    async def doc_count(self, target: Target) -> int:
        index = self._indexes.get(target.value, self._items)
        return int((await self._client.count(index=index))['count'])

    async def free_bytes(self) -> int:
        stats = await self._client.nodes.stats(metric='fs')
        nodes = stats['nodes'].values()
        return min(int(n['fs']['total']['available_in_bytes']) for n in nodes)
