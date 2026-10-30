"""Start a loop: pin what it will run against into its job directory and queue it for the
worker. The state the API writes is the whole of what the loop is accepted as (the
normalized request, the endpoints and pack with their revisions, the plan); the worker
resolves exactly those pins by id."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.schemas.curation.assist_loop import (
    AssistLoopState,
    LoopPackPin,
    LoopPinned,
    LoopPlan,
    LoopVlmPin,
)
from src.services.curation.assist_loop import job
from src.services.curation.assist_loop.plan import build_plan
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from src.services.curation.assist_loop.prepare import PreparedLoop
    from src.services.curation.autolabel.run_vlm import RunVlm


def _pin(vlm: RunVlm | None) -> LoopVlmPin | None:
    if vlm is None or vlm.endpoint is None:
        return None
    return LoopVlmPin(name=vlm.name, revision=vlm.revision, model=vlm.endpoint.model_id)


def _pinned(prepared: PreparedLoop) -> LoopPinned:
    primary = _pin(prepared.vlm)
    assert primary is not None  # require_run_vlm refused a project with its VLM off
    return LoopPinned(
        vlm=primary,
        prompt_pack=LoopPackPin(name=prepared.pack.name, revision=prepared.pack_revision),
        audit_vlm=_pin(prepared.audit_vlm),
    )


def plan_of(prepared: PreparedLoop) -> LoopPlan:
    return build_plan(prepared.request, prepared.roots, images_per_call=prepared.images_per_call)


def dry_run_state(prepared: PreparedLoop) -> AssistLoopState:
    return AssistLoopState(
        status='dry_run',
        request=prepared.request,
        pinned=_pinned(prepared),
        plan=plan_of(prepared),
    )


def start_loop(prepared: PreparedLoop) -> dict[str, Any]:
    """Queue the loop (409 ``assist_loop_busy`` while another is live); its state."""
    request = prepared.request.model_dump(mode='json')
    loop = job.create_loop(
        state={
            'status': 'queued',
            'created_at': now_iso(),
            'request': request,
            'pinned': _pinned(prepared).model_dump(mode='json'),
            'plan': plan_of(prepared).model_dump(mode='json'),
        },
        request=request,
    )
    return loop.read()


__all__ = ['dry_run_state', 'plan_of', 'start_loop']
