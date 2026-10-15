"""Turn an :class:`AssistRunRequest` into a run that can start: validate it,
resolve the scope and the sample, pin the VLM endpoint and the prompt pack.

Everything here is read-only, so a dry run is exactly this and nothing more.
Refusals are structured: 409 ``vlm_not_configured`` (the project's VLM is off
and none was named), 422 ``pack_declares_no_attributes`` and ``no_class_names``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.assist.models import AssistRunRequest, ResolvedScope, validate_request
from src.services.curation.assist.scope import ScopeResolution, resolve_scope
from src.services.curation.autolabel.run_params import resolve_run_prompt_pack
from src.services.curation.autolabel.run_vlm import RunVlm, require_run_vlm, resolve_run_vlm
from src.services.labeling.run_labeler import resolve_pack
from src.services.labeling.vlm_attributes import specs_for
from src.services.labeling.vlm_client import DEFAULT_OPEN_IMAGES_PER_CALL
from src.services.rejections import rejection


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.labeling.vlm_prompts import PromptPack


@dataclass
class PreparedRun:
    request: AssistRunRequest
    scope: ScopeResolution
    vlm: RunVlm
    pack: PromptPack
    pack_name: str | None
    pack_revision: int | None
    images_per_call: int

    def resolved(self) -> ResolvedScope:
        n = len(self.scope.sampled_ids)
        endpoint = self.vlm.endpoint
        return ResolvedScope(
            scope_count=len(self.scope.members),
            scope_digest=self.scope.digest,
            sampled=n,
            sample_seed=self.scope.seed,
            strata=self.scope.strata,
            skipped=self.scope.skipped,
            vlm={
                'name': self.vlm.name,
                'revision': self.vlm.revision,
                'model': endpoint.model_id if endpoint is not None else None,
            },
            prompt_pack={'name': self.pack.name, 'revision': self.pack_revision},
            estimated_vlm_calls=math.ceil(n / self.images_per_call) if n else 0,
        )


def _check_pack(request: AssistRunRequest, pack: PromptPack) -> None:
    if 'attributes' not in request.actions:
        return
    blank = [
        f
        for f in ('assist_system', 'assist_user_template')
        if 'classify' in request.actions and not getattr(pack, f).strip()
    ] + [
        f
        for f in ('attributes_system', 'attributes_user_template')
        if 'classify' not in request.actions and not getattr(pack, f).strip()
    ]
    if not specs_for(pack.attributes, 'item') or blank:
        raise rejection(
            422,
            'pack_declares_no_attributes',
            f'prompt pack {pack.name!r} declares no item attributes'
            + (f' or has no {", ".join(blank)}' if blank else ''),
        )


def _check_classes(request: AssistRunRequest) -> None:
    if 'classify' not in request.actions:
        return
    from src.clients.curation_opensearch.registry import get_class_registry
    from src.services.curation.region_class import item_classes

    if not item_classes(get_class_registry().load().classes):
        raise rejection(
            422,
            'no_class_names',
            'classify needs item classes: the project registry has none; create a class first',
        )


async def prepare_run(opensearch: AsyncOpenSearch, request: AssistRunRequest) -> PreparedRun:
    validate_request(request)
    pack_name, pack_revision = await resolve_run_prompt_pack(opensearch, request.prompt_pack)
    pack = resolve_pack(pack_name, pack_revision)
    _check_pack(request, pack)
    _check_classes(request)
    vlm = await resolve_run_vlm(
        opensearch,
        request.vlm,
        pack=pack,
        acknowledge_external=request.acknowledge_external,
    )
    require_run_vlm(vlm)
    scope = await resolve_scope(opensearch, request)
    limit = _endpoint_images(vlm)
    return PreparedRun(
        request=request,
        scope=scope,
        vlm=vlm,
        pack=pack,
        pack_name=pack_name,
        pack_revision=pack_revision,
        images_per_call=max(1, min(DEFAULT_OPEN_IMAGES_PER_CALL, limit)),
    )


def _endpoint_images(vlm: RunVlm) -> int:
    body: Any = vlm.endpoint.body if vlm.endpoint is not None else None
    value = getattr(body, 'max_images_per_call', None)
    return int(value) if isinstance(value, int) and value > 0 else DEFAULT_OPEN_IMAGES_PER_CALL


__all__ = ['PreparedRun', 'prepare_run']
