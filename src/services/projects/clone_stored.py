"""Copy one project's stored configs of one kind into another (the
``prompt_packs`` and ``open_vocab`` clone axes): current revision only,
revision history is not carried over. Split out of ``clone.py`` for the
700-LOC ratchet; the source is always read under a read-only bind."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.config_store.index import ConfigKind
    from src.services.config_store.store import StoredConfig


async def copy_stored_configs(
    client: Any,
    *,
    target_record: ProjectRecord,
    source: ProjectRecord,
    kind: ConfigKind,
    stored_field: str,
) -> dict[str, StoredConfig]:
    """Copy every STORED config of ``kind`` (``stored_field`` names the
    snapshot dict holding them) from ``source`` into ``target_record``.
    A no-op when the source has none; an axis's activation (if also cloned)
    still owns which one is active.

    Returns ``{name: StoredConfig}`` for exactly what was written, so a
    caller can tell, without re-reading the target store's cache, whether
    the config it needs to activate was already written here, and with what
    body."""
    from src.config import get_curation_config
    from src.services.config_store import get_config_store
    from src.services.config_store.index import save_config
    from src.services.config_store.store import StoredConfig

    with bind_project(source, read_only=True):
        source_store = get_config_store()
        await source_store.ensure_fresh(client)
        stored = dict(getattr(source_store.current, stored_field))

    written: dict[str, StoredConfig] = {}
    if not stored:
        return written
    with bind_project(target_record):
        target_index = get_curation_config().configs_index
        for name, config in stored.items():
            doc = await save_config(
                client,
                target_index,
                kind=kind,
                name=name,
                body=config.body,
                expected_revision=None,
                description=config.description,
                cloned_from=f'{source.slug}:{name}@{config.revision}',
            )
            written[name] = StoredConfig(
                kind=kind,
                name=name,
                revision=int(doc['revision']),
                body=config.body,
                description=config.description,
            )
    return written


__all__ = ['copy_stored_configs']
