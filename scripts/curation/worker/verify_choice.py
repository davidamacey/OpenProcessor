"""Which way an item is verified: by the VLM, or accepted without one.

The choice is per item (``region_run_verify``, stamped by an assist run that
re-queued it) and per project (``region_verify`` in the VLM policy); the item's wins.
``none`` takes the no-VLM acceptance path even when a VLM is active. ``vlm`` is the
default and means "verify with the project's VLM when it has one" (the behaviour
before the choice existed); an *explicit* per-item ``vlm`` with no VLM available fails
closed, writing ``detection_failed`` / ``vlm_unavailable`` instead of accepting an
unverified box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.curation.worker.region_text_stage import accept_without_vlm
from scripts.curation.worker.state import _ItemTask, vlm_unavailable_update


if TYPE_CHECKING:
    from scripts.curation.worker.pipeline import PipelineContext


def effective_verify(t: _ItemTask) -> str:
    """``vlm`` or ``none``: the item's own choice, else the project's policy."""
    return t.region_run_verify or t.policy_verify or 'vlm'


async def dispatch_verification(ctx: PipelineContext, rt: Any, t: _ItemTask) -> None:
    """Candidates are set on ``t``: one combined VLM call, or straight to the writer."""
    choice = effective_verify(t)
    if choice == 'vlm' and rt.vlm_available:
        await ctx.combined_q.put(t)
        return
    if choice == 'vlm' and t.region_run_verify == 'vlm':
        t.update_doc = vlm_unavailable_update(t)
        await ctx.out_q.put(t)
        return
    await accept_without_vlm(t, ocr=rt.ocr_recognizer, profile=rt.profile, rules=rt.text_rules)
    await ctx.out_q.put(t)


__all__ = ['dispatch_verification', 'effective_verify']
