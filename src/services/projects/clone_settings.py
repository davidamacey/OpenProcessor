"""The ``settings_defaults`` axis of project ``clone_settings``: the source's
settings document, i.e. the strategy defaults and the ingest policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.curation.ingest_policy import IngestPolicyBody
from src.services.curation.ingest_policy_store import get_ingest_policy, put_ingest_policy


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def clone_settings_document(
    client: Any, *, source: ProjectRecord, target_record: ProjectRecord
) -> None:
    """Copy the source's strategy defaults and ingest policy into the target
    (a source that never wrote a policy leaves the target on the defaults)."""
    from src.clients.curation_opensearch import get_curation_settings, update_curation_settings

    with bind_project(source, read_only=True):
        source_settings = await get_curation_settings(client)
        source_policy = await get_ingest_policy(client)
    with bind_project(target_record):
        await update_curation_settings(client, dict(source_settings.get('defaults', {})))
        if source_policy.revision:
            await put_ingest_policy(
                client,
                IngestPolicyBody(detect=source_policy.detect, embedding=source_policy.embedding),
                expected_revision=(await get_ingest_policy(client)).revision,
            )
