"""Role-to-body table and idempotent index creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.base import config, logger
from src.clients.curation_opensearch.bodies_core import _images_body, _items_body
from src.clients.curation_opensearch.bodies_other import (
    _classes_body,
    _configs_body,
    _labels_confirmed_body,
    _settings_body,
    _umap_state_body,
    _umap_viz_state_body,
)
from src.config import CurationConfig, IndexRole, index_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


INDEX_BODIES: dict[IndexRole, dict[str, Any]] = {
    IndexRole.IMAGES: _images_body(),
    IndexRole.ITEMS: _items_body(),
    IndexRole.LABELS_CONFIRMED: _labels_confirmed_body(),
    IndexRole.CLASSES: _classes_body(),
    IndexRole.SETTINGS: _settings_body(),
    IndexRole.UMAP_STATE: _umap_state_body(),
    IndexRole.UMAP_VIZ_STATE: _umap_viz_state_body(),
    IndexRole.CONFIGS: _configs_body(),
}


async def get_curation_index_settings() -> dict[str, dict[str, Any]]:
    """Return the schema dict keyed by index name (string).

    Used by tests + introspection routes.
    """
    return {index_name(config, role): body for role, body in INDEX_BODIES.items()}


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
    except Exception as e:
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
    active_cfg = cfg or config
    # A project may fold several roles onto one index name (shard folding,
    # owner D4 -- SETTINGS / UMAP_VIZ_STATE onto CONFIGS for a project
    # created after W2). Merge their mapping properties into one body per
    # distinct name rather than creating the name once with only
    # whichever role's body happened to be seen first.
    bodies_by_name: dict[str, dict[str, Any]] = {}
    for role, body in INDEX_BODIES.items():
        name = index_name(active_cfg, role)
        if name in bodies_by_name:
            bodies_by_name[name] = _merge_index_bodies(bodies_by_name[name], body)
        else:
            bodies_by_name[name] = body
    results: dict[str, bool] = {}
    for name, body in bodies_by_name.items():
        results[name] = await _create_one(client, name, body, force_recreate)
    return results


def _merge_index_bodies(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Union two index bodies' mapping properties (settings/shard config
    taken from ``first``) -- used when shard folding maps more than one
    ``IndexRole`` onto the same index name."""
    merged_properties = {
        **first.get('mappings', {}).get('properties', {}),
        **second.get('mappings', {}).get('properties', {}),
    }
    return {
        'settings': first.get('settings', {}),
        'mappings': {
            'dynamic': first.get('mappings', {}).get(
                'dynamic', second.get('mappings', {}).get('dynamic')
            ),
            'properties': merged_properties,
        },
    }
