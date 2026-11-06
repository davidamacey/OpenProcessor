"""The ``settings_defaults`` axis of project ``clone_settings``: the source's
settings document, i.e. the strategy defaults, the ingest policy and the VLM policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.project_context import bind_project
from src.services.curation.ingest_policy import IngestPolicyBody
from src.services.curation.ingest_policy_store import get_ingest_policy, put_ingest_policy
from src.services.curation.vlm_policy import VlmPolicyBody
from src.services.curation.vlm_policy_store import get_vlm_policy, put_vlm_policy


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def clone_settings_document(*, source: ProjectRecord, target_record: ProjectRecord) -> None:
    """Copy the source's strategy defaults, ingest policy and VLM policy into the
    target (a source that never wrote a policy leaves the target on the defaults)."""
    from src.services.curation.settings_store import get_curation_settings, update_curation_settings

    with bind_project(source, read_only=True):
        source_settings = await get_curation_settings(fresh=True)
        source_policy = await get_ingest_policy()
        source_vlm_policy = await get_vlm_policy()
    with bind_project(target_record):
        target_settings = await get_curation_settings(fresh=True)
        await update_curation_settings(
            dict(source_settings.get('defaults', {})),
            expected_revision=target_settings['revision'],
        )
        if source_policy.revision:
            await put_ingest_policy(
                IngestPolicyBody(**source_policy.model_dump(exclude={'revision'})),
                expected_revision=(await get_ingest_policy()).revision,
            )
        if source_vlm_policy.revision:
            await put_vlm_policy(
                VlmPolicyBody(**source_vlm_policy.model_dump(exclude={'revision'})),
                expected_revision=(await get_vlm_policy()).revision,
            )
