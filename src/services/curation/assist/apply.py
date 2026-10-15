"""Apply a run's proposed class to the scope: the one path from a suggestion to labels.

An explicit user action with a dry-run count first. The write is the human
relabel service ``PUT /crops/batch_label`` uses (one writer), so the items become
human-validated, as if the user had labelled them. It acts on the scope's
*current* members, refuses when the scope is not the one the run saw
(``scope_changed``), and leaves every member whose class is locked, validated,
holdout or excluded alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.assist.proposal_models import (
    ApplyConflict,
    AssistApplyRequest,
    AssistApplyResponse,
    ScopeNow,
)
from src.services.curation.assist.scope_now import current_members, scope_now
from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.class_write_guard import class_write_locked
from src.services.curation.crop_bulk import occ_bulk_human_relabel
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
    locked_at_write: set[str] = set()

    def merge(current: dict[str, Any]) -> dict[str, Any]:
        # A human or an import got there between the read and the write.
        if class_write_locked(current):
            locked_at_write.add(str(current.get('crop_id', '')))
            return {}
        return class_label_update(
            current,
            ItemLabel.human(
                class_id=class_id,
                class_name=class_name,
                label_source=body.label_source,
                writer=WRITER,
            ),
        )

    written = await occ_bulk_human_relabel(opensearch, eligible, merge, writer_id=WRITER)
    written['updated_ids'] = [i for i in written['updated_ids'] if i not in locked_at_write]
    written['updated'] = len(written['updated_ids'])
    return _response(body, class_name, eligible, skipped + len(locked_at_write), now, written)


def _response(
    body: AssistApplyRequest,
    class_name: str,
    eligible: list[str],
    skipped: int,
    now: ScopeNow,
    written: dict[str, Any] | None,
) -> AssistApplyResponse:
    written = written or {}
    return AssistApplyResponse(
        dry_run=body.dry_run,
        class_name=class_name,
        selected=len(eligible),
        skipped_locked=skipped,
        updated=int(written.get('updated', 0)),
        updated_ids=list(written.get('updated_ids', [])),
        conflicts=[ApplyConflict(**c) for c in written.get('conflicts', [])],
        scope_now=now,
    )


__all__ = ['SAME_SCOPE_JACCARD', 'apply_proposal']
