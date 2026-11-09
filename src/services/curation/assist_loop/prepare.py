"""Turn an :class:`AssistLoopRequest` into a loop that can start: validate it, pin the VLM
endpoints and the prompt pack, and survey the roots.

Everything here only reads, so a dry run is exactly this and nothing more. Refusals are
structured: 422 ``invalid_request`` (a start without a budget), ``no_class_names``,
``no_roots``; 409 ``vlm_not_configured``, ``vlm_external_*`` (the main or the audit
endpoint), ``clustering_running`` (a clustering run or projection is live: its activation would
re-project the items under the loop).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.project_context import current_project_id
from src.db import session as db
from src.services.clustering.job import PROJECTION_KIND, RUN_KIND
from src.services.curation.assist.prepare import images_per_call_for, require_item_classes
from src.services.curation.assist_loop.audit import resolve_audit_vlm
from src.services.curation.assist_loop.partition import (
    NoServingPartitionError,
    PartitionReader,
    PartitionScope,
    RootSize,
)
from src.services.curation.autolabel.run_params import resolve_run_prompt_pack
from src.services.curation.autolabel.run_vlm import RunVlm, require_run_vlm, resolve_run_vlm
from src.services.jobs import status as job_status
from src.services.labeling.run_labeler import resolve_pack
from src.services.rejections import rejection


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.curation.assist_loop import AssistLoopRequest
    from src.services.labeling.vlm_prompts import PromptPack


@asynccontextmanager
async def _scoped(session: AsyncSession) -> AsyncIterator[AsyncSession]:
    async with db.project_tx(session, current_project_id()) as scoped:
        yield scoped


@dataclass
class PreparedLoop:
    request: AssistLoopRequest
    vlm: RunVlm
    audit_vlm: RunVlm | None
    pack: PromptPack
    pack_revision: int | None
    images_per_call: int
    roots: list[RootSize]


_LIVE = ('queued', 'running', 'paused', 'cancelling')


async def clustering_is_live(session: AsyncSession) -> bool:
    """Whether a clustering run or projection of the bound project is queued, running, paused
    or cancelling: either may flip the serving partition under a loop."""
    page = await job_status.list_jobs(
        session,
        project=current_project_id(),
        kinds=(RUN_KIND, PROJECTION_KIND),
        statuses=_LIVE,
        limit=1,
    )
    return bool(page.items)


async def prepare_loop(
    opensearch: AsyncOpenSearch, session: AsyncSession, request: AssistLoopRequest
) -> PreparedLoop:
    if not request.dry_run and request.budget is None:
        raise rejection(
            422,
            'invalid_request',
            'budget.max_vlm_crops is required to start a loop: it has no default',
        )
    await require_item_classes()
    pack_name, pack_revision = await resolve_run_prompt_pack(opensearch, request.prompt_pack)
    pack = resolve_pack(pack_name, pack_revision)
    vlm = await resolve_run_vlm(
        request.vlm, pack=pack, acknowledge_external=request.acknowledge_external
    )
    require_run_vlm(vlm)
    audit_vlm = await resolve_audit_vlm(
        request.audit.vlm,
        pack=pack,
        acknowledge_external=request.audit.acknowledge_external,
    )
    reader = PartitionReader(opensearch, lambda: _scoped(session), project=current_project_id())
    try:
        await reader.pin()
    except NoServingPartitionError as exc:
        raise rejection(422, 'no_roots', str(exc)) from None
    roots = await reader.survey(
        PartitionScope(
            cluster_uids=request.scope.cluster_uids, min_root_size=request.scope.min_root_size
        )
    )
    if not roots:
        raise rejection(
            422,
            'no_roots',
            'no cluster is big enough to work on: lower scope.min_root_size or name clusters',
        )
    if not request.dry_run and await clustering_is_live(session):
        raise rejection(
            409,
            'clustering_running',
            'a clustering run is live and would re-project the clusters under the loop; '
            'wait for it or cancel it',
        )
    return PreparedLoop(
        request=request,
        vlm=vlm,
        audit_vlm=audit_vlm,
        pack=pack,
        pack_revision=pack_revision,
        images_per_call=images_per_call_for(vlm.endpoint),
        roots=roots,
    )


__all__ = ['PreparedLoop', 'clustering_is_live', 'prepare_loop']
