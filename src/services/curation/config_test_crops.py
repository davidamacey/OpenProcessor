"""Stored-crop read for the pack / region-profile test routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.crops import mget_crops
from src.config.curation_accessors import items_index
from src.services.curation.wire import item_source_excludes
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def fetch_test_crop_docs(
    opensearch: AsyncOpenSearch, crop_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """The bound project's item documents for ``crop_ids`` (vectors excluded),
    keyed by id; ids with no document are absent."""
    return await mget_crops(
        opensearch,
        crop_ids,
        index=items_index(),
        source_excludes=item_source_excludes(await vector_source_excludes()),
    )
