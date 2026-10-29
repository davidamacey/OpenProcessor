"""The dry-run sizing of a loop: how many clusters and eligible items it would work on and
what its first draws cost, from the exact sample sizes of the gate. Reads nothing more than
the survey; it estimates the root level only, because how deep the loop splits depends on the
VLM's answers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.domain.loop_stats import required_sample
from src.schemas.curation.assist_loop import LoopPlan, LoopRequiredSample


if TYPE_CHECKING:
    from src.schemas.curation.assist_loop import AssistLoopRequest
    from src.services.curation.assist_loop.partition import RootSize

TABLE_ROWS = 25
"""Distinct cluster sizes the table lists, largest first."""


def build_plan(
    request: AssistLoopRequest, roots: list[RootSize], *, images_per_call: int
) -> LoopPlan:
    gate = request.gate
    alpha = gate.alpha / (gate.max_dissent + 1)  # the level each look is tested at
    sizes = sorted({r.eligible for r in roots if r.eligible > 0}, reverse=True)
    table = [
        LoopRequiredSample(
            n_members=n,
            required_by_dissent=[
                required_sample(n, gate.theta, alpha, d) for d in range(gate.max_dissent + 1)
            ],
        )
        for n in sizes[:TABLE_ROWS]
    ]
    draws = [required_sample(r.eligible, gate.theta, alpha, 0) for r in roots if r.eligible > 0]
    crops = sum(draws)
    return LoopPlan(
        roots=len(roots),
        eligible_items=sum(r.eligible for r in roots),
        required_sample_table=table,
        estimated_vlm_crops_root_level=crops,
        estimated_vlm_requests_root_level=sum(math.ceil(d / images_per_call) for d in draws),
        fits_budget=None if request.budget is None else crops <= request.budget.max_vlm_crops,
    )


__all__ = ['TABLE_ROWS', 'build_plan']
