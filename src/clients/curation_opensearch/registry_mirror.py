"""Mirror of the class registry document into the per-class ``classes`` index.

The ``classes`` index holds one document per class id for term filters and dashboards. The
registry document is the authority; a mirror failure is logged and the index bootstrap re-syncs
from the document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException

from src.clients.curation_opensearch.base import logger
from src.clients.curation_opensearch.lifecycle import INDEX_BODIES
from src.clients.opensearch.bulk import failed_bulk_items
from src.config import IndexRole


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.clients.curation_opensearch import ClassRegistryFile, RegistryClassEntry


def _bulk_body(classes_index: str, entries: list[RegistryClassEntry]) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = []
    for entry in entries:
        body.append({'index': {'_index': classes_index, '_id': str(entry.class_id)}})
        body.append(entry.model_dump())
    return body


async def mirror_entries(
    client: AsyncOpenSearch, classes_index: str, entries: list[RegistryClassEntry]
) -> None:
    """Upsert ``entries`` into the ``classes`` index after a registry write."""
    if not entries:
        return
    try:
        resp = await client.bulk(body=_bulk_body(classes_index, entries), refresh=False)
    except OpenSearchException as exc:
        logger.warning('curation_registry_mirror_failed', error=str(exc), n=len(entries))
        return
    failed = failed_bulk_items(resp) if isinstance(resp, dict) else []
    if failed:
        logger.warning(
            'curation_registry_mirror_partial',
            failed=len(failed),
            n=len(entries),
            sample=failed[:3],
        )


async def sync_all(
    client: AsyncOpenSearch, classes_index: str, doc: ClassRegistryFile
) -> dict[str, int]:
    """Upsert every class (deprecated ones stay, flagged), creating the index if it is missing."""
    if not await client.indices.exists(index=classes_index):
        await client.indices.create(index=classes_index, body=INDEX_BODIES[IndexRole.CLASSES])
        logger.info('curation_index_created_on_sync', index=classes_index)
    failed = 0
    if doc.classes:
        resp = await client.bulk(body=_bulk_body(classes_index, doc.classes), refresh=False)
        failed_items = failed_bulk_items(resp) if isinstance(resp, dict) else []
        failed = len(failed_items)
        if failed:
            logger.warning(
                'curation_registry_sync_partial_errors', failed=failed, sample=failed_items[:3]
            )
    await client.indices.refresh(index=classes_index)
    upserted = len(doc.classes) - failed
    logger.info('curation_registry_sync', upserted=upserted, n_classes=len(doc.classes))
    return {'upserted': upserted, 'n_classes': len(doc.classes)}
