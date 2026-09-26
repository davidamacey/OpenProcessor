"""Idempotent startup bootstrap of the ``default`` project record.

This is the *only* "migration" P1 performs, and it touches no data
index -- it just upserts the registry doc so ``default`` shows up in
``GET /projects`` and ``bind_default_project`` has a record to bind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.config.curation import base_curation_config
from src.config.projects import DEFAULT_SLUG, ProjectRecord, resources_for_default
from src.core.logging import get_logger
from src.services.projects.registry import REVISION_DOC_ID, projects_index, record_to_doc


logger = get_logger(__name__)


async def bootstrap_default_project(client: Any) -> ProjectRecord:
    """Create the ``default`` project doc if it does not exist yet, or
    return the existing one unchanged. Never overwrites an existing
    record (a later env change must not remap a live project), and never
    issues a reindex/update_by_query against any data index."""
    doc_id = f'project:{DEFAULT_SLUG}'
    try:
        existing = await client.get(index=projects_index(), id=doc_id)
        if existing.get('found', True):
            from src.services.projects.registry import doc_to_record

            return doc_to_record(existing['_source'])
    except Exception:  # nosec B110 - the client raises on a missing doc; that's first-boot, not an error
        logger.debug('no existing default project doc; bootstrapping one')

    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug=DEFAULT_SLUG,
        display_name='Default',
        description='The original, unscoped dataset workspace.',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_default(base_curation_config()),
    )
    await client.index(index=projects_index(), id=doc_id, body=record_to_doc(record))
    await _bump_revision(client)
    logger.info('bootstrapped default project record')
    return record


async def _bump_revision(client: Any) -> None:
    try:
        current = await client.get(index=projects_index(), id=REVISION_DOC_ID)
        revision = int((current.get('_source') or {}).get('revision', 0)) + 1
    except Exception:
        revision = 1
    await client.index(index=projects_index(), id=REVISION_DOC_ID, body={'revision': revision})
