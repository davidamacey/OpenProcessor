"""Apply a run's proposed class to the scope: the one path from a suggestion to labels.

An explicit user action with a dry-run count first. By default the write is the
human relabel service ``PUT /crops/batch_label`` uses (one writer), so the items
become human-validated, as if the user had labelled them. ``label_source:
"propagated"`` writes the machine kind instead (unvalidated, the run id in the
provenance), the same document the auto-label loop writes. It acts on the scope's
*current* members, refuses when the scope is not the one the run saw
(``scope_changed``), and leaves every member whose class is locked, validated,
holdout or excluded alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.curation.assist.apply_write import ClassWrite, write_class_to_ids
from src.services.curation.assist.proposal_models import (
    ApplyConflict,
    AssistApplyRequest,
    AssistApplyResponse,
    ScopeNow,
)
from src.services.curation.assist.scope_now import current_members, scope_now
from src.services.curation.class_label import ItemLabel
from src.services.curation.class_write_guard import class_write_locked
from src.services.rejections import rejection
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.assist.models import AssistRunState

SAME_SCOPE_JACCARD = 0.9
"""Overlap below which the scope is not the one the run saw."""
WRITER = 'human:assist_apply'


def _registry_class(class_name: str) -> tuple[int, str]:
    from src.clients.curation_opensearch.registry import get_class_registry
    from src.services.curation.region_class import item_classes

    wanted = normalize_class_name(class_name)
    for entry in item_classes(get_class_registry().load().classes):
        if normalize_class_name(entry.class_name) == wanted:
            return entry.class_id, entry.class_name
    raise rejection(
        409,
        'class_not_in_registry',
        f'{class_name!r} is not a class of this project; create it first',
        class_name=class_name,
    )


def _label(body: AssistApplyRequest, run_id: str, class_id: int, class_name: str) -> ItemLabel:
    if body.label_source == 'propagated':
        return ItemLabel.propagated(
            class_id=class_id,
            class_name=class_name,
            run_id=run_id,
            writer=f'assist_loop:{run_id}',
        )
    return ItemLabel.human(
        class_id=class_id, class_name=class_name, label_source=body.label_source, writer=WRITER
    )


async def apply_proposal(
    opensearch: AsyncOpenSearch, run_id: str, state: AssistRunState, body: AssistApplyRequest
) -> AssistApplyResponse:
    members = await current_members(opensearch, state.request)
    if members is None:
        raise rejection(
            409,
            'run_has_no_scope',
            'this run has no cluster or sub-cluster scope to apply a class to',
            run_id=run_id,
        )
    class_id, class_name = _registry_class(body.class_name)
    now = await scope_now(opensearch, run_id, state.request, members=members)
    if now is None:
        raise rejection(409, 'run_has_no_scope', 'this run kept no scope members', run_id=run_id)
    if now.jaccard < SAME_SCOPE_JACCARD and not body.allow_changed_scope:
        raise rejection(
            409,
            'scope_changed',
            f'only {now.jaccard:.0%} of the scope is what the run saw (a re-refine or relabel '
            'moved items); run again, or set allow_changed_scope',
            run_id=run_id,
            scope_now=now.model_dump(),
        )
    eligible = [cid for cid, src in members if not class_write_locked(src)]
    skipped = len(members) - len(eligible)
    if body.dry_run or not eligible:
        return _response(body, class_name, eligible, skipped, now, written=None)
    written = await write_class_to_ids(
        opensearch, eligible, _label(body, run_id, class_id, class_name)
    )
    return _response(
        body, class_name, eligible, skipped + len(written.skipped_locked), now, written
    )


def _response(
    body: AssistApplyRequest,
    class_name: str,
    eligible: list[str],
    skipped: int,
    now: ScopeNow,
    written: ClassWrite | None,
) -> AssistApplyResponse:
    return AssistApplyResponse(
        dry_run=body.dry_run,
        class_name=class_name,
        selected=len(eligible),
        skipped_locked=skipped,
        updated=len(written.updated_ids) if written else 0,
        updated_ids=list(written.updated_ids) if written else [],
        conflicts=[ApplyConflict(**c) for c in written.conflicts] if written else [],
        scope_now=now,
    )


__all__ = ['SAME_SCOPE_JACCARD', 'apply_proposal']
