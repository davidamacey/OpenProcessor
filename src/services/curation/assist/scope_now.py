"""A run's scope as it is now, against the members the run resolved."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.assist.models import AssistRunRequest, ClusterTarget, CropsTarget
from src.services.curation.assist.proposal_models import ScopeNow
from src.services.curation.assist.scope import filter_members, membership_overlap, proposal_scope


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


def _label(request: AssistRunRequest) -> str:
    target = request.target
    if isinstance(target, ClusterTarget):
        scope = target
    elif isinstance(target, CropsTarget) and target.context is not None:
        scope = target.context  # type: ignore[assignment]
    else:
        return 'the scope'
    return f'cluster {scope.cluster_uid}'


async def current_members(
    opensearch: AsyncOpenSearch, request: AssistRunRequest
) -> list[tuple[str, dict[str, Any]]] | None:
    """The visible members of the run's cluster scope now, with the fields the lock
    rule reads; ``None`` when the run has no cluster scope (a bare selection or a
    filter). A scope with no members now is an empty list."""
    flt = proposal_scope(request.target)
    if flt is None:
        return None
    from src.core.exceptions import RequestRejectedError

    try:
        members, _ = await filter_members(opensearch, flt, label=_label(request))
    except RequestRejectedError as exc:
        if exc.status_code == 404:
            return []
        raise
    return members


async def scope_now(
    opensearch: AsyncOpenSearch,
    request: AssistRunRequest,
    *,
    before: list[str] | None,
    members: list[tuple[str, dict[str, Any]]] | None = None,
) -> ScopeNow | None:
    """How the scope changed since the run resolved it (``before``: the member ids the run
    kept, from :func:`~src.services.curation.assist.job.read_members`), or ``None`` when the
    run has no cluster scope or kept no member list. ``members`` is the caller's
    ``current_members`` when it already has them."""
    now = members if members is not None else await current_members(opensearch, request)
    if now is None or before is None:
        return None
    return ScopeNow(**membership_overlap(before, [cid for cid, _ in now]))


__all__ = ['current_members', 'scope_now']
