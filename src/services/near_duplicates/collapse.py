"""Collapse near-duplicate frames out of an export from the stored groups of the current run.

The export used to recompute whole-frame groups at a request-time cosine cut; it now reads the
`dedup` groups of the project's current image-level run, so the training set and the review
queue agree on what a duplicate is. With no current run there are no groups to read, so the
export refuses instead of silently exporting every frame.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.db.repos import near_dup as repo, near_dup_export
from src.db.session import async_session_factory, project_tx
from src.domain import near_dup as rules
from src.services.config_store.bound_project import bound_project_id


class NoCurrentRunError(RuntimeError):
    """The project has no current image-level near-duplicate run to read groups from."""


class CollapsibleRow(Protocol):
    """What the collapse needs of an export row: its frame and whether it is held out."""

    image_id: str
    has_test_crop: bool


async def collapse_near_duplicates[R: CollapsibleRow](
    rows: list[R],
) -> tuple[list[R], dict[str, Any]]:
    """``rows`` without the near-duplicate frames, and the stats the manifest records.

    Rows are grouped by ``image_id``; each dedup group keeps one frame and every row of a
    surviving frame stays (distinct items of one frame are never duplicates of each other). A
    frame with no seat in a group, and a row with no image id, are kept untouched.
    """
    project_id = await bound_project_id()
    by_frame: dict[str, list[R]] = {}
    for row in rows:
        if row.image_id:
            by_frame.setdefault(row.image_id, []).append(row)
    async with async_session_factory()() as session, project_tx(session, project_id):
        run = await repo.current_run(session, project_id, 'image')
        if run is None:
            msg = 'collapse_near_duplicates needs a near-duplicate run; start a full build first'
            raise NoCurrentRunError(msg)
        seats = await near_dup_export.dedup_seats(session, run.id, by_frame)
        pending = await repo.pending_count(session, project_id, 'image')
    preferred = {
        f for f, frame_rows in by_frame.items() if any(r.has_test_crop for r in frame_rows)
    }
    dropped = rules.collapse_drops(seats, preferred)
    kept = [r for r in rows if r.image_id not in dropped]
    groups = {seats[f].group_id for f in dropped}
    return kept, {
        'enabled': True,
        'run_id': str(run.id),
        'pending_unscored': pending,
        'n_input_rows': len(rows),
        'n_frames': len(by_frame),
        'near_dup_groups': len(groups),
        'frames_dropped': len(dropped),
        'rows_dropped': len(rows) - len(kept),
        'n_output_rows': len(kept),
    }
