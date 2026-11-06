"""The project-bound inputs of the segmenter gate for the open-vocabulary pass:
the vision model for tier 2 and the persisted hit-rate windows for tier 3.

The decision itself is :func:`~src.services.detection.segmenter_gate.decide`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from sqlalchemy.exc import SQLAlchemyError

from src.core.logging import get_logger
from src.db.repos import settings as settings_repo
from src.db.repos.config import RevisionConflictError
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id
from src.services.detection.segmenter_gate import HitRateTracker


logger = get_logger(__name__)

VlmVisibleFn = Callable[[bytes, str], Awaitable[bool | None]]

#: The tier-3 windows are the ``project_state`` row of this key.
STATE_KEY = 'open_vocab_gate'


async def active_vlm_visible() -> VlmVisibleFn | None:
    """The tier-2 question, asked of the bound project's active vision model;
    ``None`` when the project has none (tier 2 then cannot run, and the call
    proceeds)."""
    from src.services.labeling.vlm_endpoints import (
        VlmEndpointUnavailableError,
        active_vlm_endpoint,
        refresh_vlm_state,
    )
    from src.services.labeling.vlm_factory import labeler_for
    from src.services.labeling.vlm_prompt_resolution import active_prompt_pack

    try:
        await refresh_vlm_state()
        endpoint = active_vlm_endpoint()
    except VlmEndpointUnavailableError:
        return None
    if endpoint is None:
        return None
    labeler = labeler_for(endpoint, active_prompt_pack())

    async def ask(jpeg: bytes, prompt: str) -> bool | None:
        return await labeler.prompt_visible(jpeg, prompt)

    return ask


async def load_tracker() -> HitRateTracker:
    """The bound project's windows; empty when none were ever saved."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_state(session, project_id, STATE_KEY)
    return HitRateTracker() if row is None else HitRateTracker.load(row.value.get('windows'))


async def save_tracker(tracker: HitRateTracker) -> None:
    """Persist the windows. Statistics, not state of record: a failed write is
    logged and never fails the pass. The last writer wins, so the revision read here is the
    one written against and a concurrent pass only costs one pass of windows."""
    try:
        project_id = await bound_project_id()
        async with async_session_factory()() as session, project_tx(session, project_id):
            row = await settings_repo.get_state(session, project_id, STATE_KEY)
            await settings_repo.put_state(
                session,
                project_id,
                STATE_KEY,
                {'windows': tracker.dump()},
                expected_revision=0 if row is None else row.revision,
            )
    except (SQLAlchemyError, RevisionConflictError) as exc:
        logger.warning('open_vocab_gate_stats_save_failed', error=str(exc))


__all__ = ['STATE_KEY', 'VlmVisibleFn', 'active_vlm_visible', 'load_tracker', 'save_tracker']
