"""Role-to-body table and idempotent index creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import config, logger
from src.clients.curation_opensearch.bodies_core import _images_body, _items_body
from src.config import CurationConfig, IndexRole, index_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


INDEX_BODIES: dict[IndexRole, dict[str, Any]] = {
    IndexRole.IMAGES: _images_body(),
    IndexRole.ITEMS: _items_body(),
}


# =============================================================================
# Index lifecycle
# =============================================================================


async def _create_one(
    client: AsyncOpenSearch,
    index: str,
    body: dict[str, Any],
    force_recreate: bool,
) -> bool:
    """Create a single index idempotently."""
    try:
        exists = await client.indices.exists(index=index)
        if exists:
            if force_recreate:
                logger.info('curation_index_delete', index=index)
                await client.indices.delete(index=index)
            else:
                logger.info('curation_index_exists', index=index)
                return True
        try:
            await client.indices.create(index=index, body=body)
            logger.info('curation_index_created', index=index)
        except Exception as create_err:
            # Concurrent ingest workers can race here: exists() returns False
            # for everyone, then they all try to create. OpenSearch returns
            # resource_already_exists_exception for the losers -- treat it
            # as success to keep logs clean and behavior idempotent.
            if 'resource_already_exists_exception' in str(create_err):
                logger.debug('curation_index_create_race_won_by_peer', index=index)
            else:
                raise
        return True
    except OpenSearchException as e:
        logger.error('curation_index_create_failed', index=index, error=str(e))
        return False


async def create_curation_indexes(
    client: AsyncOpenSearch,
    cfg: CurationConfig | None = None,
    force_recreate: bool = False,
) -> dict[str, bool]:
    """Create every curation index idempotently.

    Args:
        client: an ``AsyncOpenSearch`` client (already configured).
        cfg: deployment config to resolve index names against. Defaults
            to the module-level singleton.
        force_recreate: drop + recreate every index. **Destructive** — only use
            during bootstrap or in tests.

    Returns:
        Dict of index name -> creation success.
    """
    results: dict[str, bool] = {}
    for name, body in _bodies_by_name(cfg or config).items():
        results[name] = await _create_one(client, name, body, force_recreate)
    return results


def _bodies_by_name(cfg: CurationConfig) -> dict[str, dict[str, Any]]:
    """The body of each index of ``cfg``'s project, by index name."""
    return {index_name(cfg, role): body for role, body in INDEX_BODIES.items()}


def curation_index_names(cfg: CurationConfig | None = None) -> list[str]:
    """The index names of the bound project (or of ``cfg``), in creation order."""
    return list(_bodies_by_name(cfg or config))
