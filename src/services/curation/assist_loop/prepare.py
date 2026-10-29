"""Turn an :class:`AssistLoopRequest` into a loop that can start: validate it, pin the VLM
endpoints and the prompt pack, and survey the roots.

Everything here only reads, so a dry run is exactly this and nothing more. Refusals are
structured: 422 ``invalid_request`` (a start without a budget), ``no_class_names``,
``no_roots``; 409 ``vlm_not_configured``, ``vlm_external_*`` (the main or the audit
endpoint), ``clustering_running`` (an auto-label job is live, whose retrain would move the
clusters the loop works on).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.curation.assist.prepare import images_per_call_for, require_item_classes
from src.services.curation.assist_loop.audit import resolve_audit_vlm
from src.services.curation.assist_loop.partition import (
    ClusterFieldPartition,
    PartitionScope,
    RootSize,
)
from src.services.curation.autolabel import job as auto_label_job
from src.services.curation.autolabel.run_params import resolve_run_prompt_pack
from src.services.curation.autolabel.run_vlm import RunVlm, require_run_vlm, resolve_run_vlm
from src.services.labeling.run_labeler import resolve_pack
from src.services.rejections import rejection


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.schemas.curation.assist_loop import AssistLoopRequest
    from src.services.labeling.vlm_prompts import PromptPack


@dataclass
class PreparedLoop:
    request: AssistLoopRequest
    vlm: RunVlm
    audit_vlm: RunVlm | None
    pack: PromptPack
    pack_revision: int | None
    images_per_call: int
    roots: list[RootSize]


async def prepare_loop(opensearch: AsyncOpenSearch, request: AssistLoopRequest) -> PreparedLoop:
    if not request.dry_run and request.budget is None:
        raise rejection(
            422,
            'invalid_request',
            'budget.max_vlm_crops is required to start a loop: it has no default',
        )
    require_item_classes()
    pack_name, pack_revision = await resolve_run_prompt_pack(opensearch, request.prompt_pack)
    pack = resolve_pack(pack_name, pack_revision)
    vlm = await resolve_run_vlm(
        opensearch, request.vlm, pack=pack, acknowledge_external=request.acknowledge_external
    )
    require_run_vlm(vlm)
    audit_vlm = await resolve_audit_vlm(
        opensearch,
        request.audit.vlm,
        pack=pack,
        acknowledge_external=request.audit.acknowledge_external,
    )
    roots = await ClusterFieldPartition(opensearch).survey(
        PartitionScope(
            cluster_ids=request.scope.cluster_ids, min_root_size=request.scope.min_root_size
        )
    )
    if not roots:
        raise rejection(
            422,
            'no_roots',
            'no cluster is big enough to work on: lower scope.min_root_size or name clusters',
        )
    if not request.dry_run and auto_label_job.is_busy():
        raise rejection(
            409,
            'clustering_running',
            'an auto-label job is running and would retrain the clusters under the loop; '
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


__all__ = ['PreparedLoop', 'prepare_loop']
