"""Save and delete a stored config and apply the result to this process's snapshot.

The four stored kinds (prompt packs, region profiles, open-vocabulary sets, ingest profiles) share
this write path; each kind's module wraps it with its own record type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.config_store import documents
from src.services.config_store.activation_apply import LOCAL_PATCH_KEY
from src.services.config_store.store import get_config_store


if TYPE_CHECKING:
    from src.services.config_store.documents import CloneSource, ConfigKind
    from src.services.config_store.snapshot import StoredConfig


async def save_and_apply(
    kind: ConfigKind,
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: CloneSource | None = None,
) -> StoredConfig:
    """Create or save a new revision, and show it in the bound project's store at once."""
    item, epoch = await documents.save_config(
        kind=kind,
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    get_config_store().apply_local(config_revision=epoch, **{LOCAL_PATCH_KEY[kind]: item})
    return item


async def delete_and_apply(kind: ConfigKind, *, name: str, expected_revision: int) -> None:
    epoch = await documents.delete_config(kind=kind, name=name, expected_revision=expected_revision)
    get_config_store().apply_local(
        config_revision=epoch, **{LOCAL_PATCH_KEY[kind]: None, 'name': name}
    )


__all__ = ['delete_and_apply', 'save_and_apply']
