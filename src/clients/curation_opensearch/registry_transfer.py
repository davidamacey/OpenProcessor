"""Moving a class registry document between homes: from a legacy file, to a clone.

These are the document-level operations the project clone and ``migrate-registry`` need beyond
ordinary class edits. Each goes through the store's versioned seams (``create_document``,
``mutate``), never a blind write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.curation_opensearch.base import logger
from src.clients.curation_opensearch.registry import ClassRegistryError, RegistryDoc
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistryFile
    from src.clients.curation_opensearch.registry import ClassRegistry


def _next_id_after(file_doc: ClassRegistryFile) -> int:
    return max((c.class_id for c in file_doc.classes), default=-1) + 1


async def copy_into(client: AsyncOpenSearch, target: ClassRegistry, source: RegistryDoc) -> None:
    """Give ``target`` the classes and id counter of ``source`` (project clone). The document is
    created; one already there is replaced by a versioned mutation."""
    doc = source.model_copy(deep=True)
    doc.registry_revision = 0
    doc.migrated_from_file_at = None
    doc.reversed_to_file_at = None
    if await target.create_document(client, doc):
        return
    logger.info('curation_registry_copy_replaces', project=target.project)

    def replace(current: RegistryDoc) -> None:
        current.classes = doc.classes
        current.next_class_id = doc.next_class_id

    await target.mutate(client, replace)


async def seed_from_file(
    client: AsyncOpenSearch, store: ClassRegistry, file_doc: ClassRegistryFile
) -> bool:
    """Create the document from a legacy file, every class id exactly as in the file.
    ``False`` when a document appeared meanwhile (the caller re-reads)."""
    doc = RegistryDoc(
        classes=[c.model_copy(deep=True) for c in file_doc.classes],
        updated_at=now_iso(),
        migrated_from_file_at=now_iso(),
        next_class_id=_next_id_after(file_doc),
    )
    return await store.create_document(client, doc)


async def replace_from_file(
    client: AsyncOpenSearch, store: ClassRegistry, file_doc: ClassRegistryFile
) -> None:
    """Replace the classes with the file's: a forward run after a reverse, when the file is the
    authority of the downgrade period. The id counter never goes back."""
    if await store.read_versioned(client) is None:
        msg = 'cannot replace a registry document that does not exist'
        raise ClassRegistryError(msg)

    def fn(doc: RegistryDoc) -> None:
        doc.classes = [c.model_copy(deep=True) for c in file_doc.classes]
        doc.next_class_id = max(doc.next_class_id, _next_id_after(file_doc))
        doc.reversed_to_file_at = None
        doc.migrated_from_file_at = now_iso()

    await store.mutate(client, fn)


async def mark_reversed(client: AsyncOpenSearch, store: ClassRegistry) -> None:
    """Record that the classes were written back to the file (rollback)."""

    def fn(doc: RegistryDoc) -> None:
        doc.reversed_to_file_at = now_iso()

    await store.mutate(client, fn)
