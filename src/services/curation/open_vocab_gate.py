"""The project-bound inputs of the segmenter gate for the open-vocabulary pass:
the vision model for tier 2 and the persisted hit-rate windows for tier 3.

The decision itself is :func:`~src.services.detection.segmenter_gate.decide`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import NotFoundError

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.detection.segmenter_gate import HitRateTracker


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

VlmVisibleFn = Callable[[bytes, str], Awaitable[bool | None]]

#: The tier-3 windows live in the project's configs index (never in the
#: config snapshot, which only reads ``doc_type: config`` rows).
TRACKER_DOC_ID = 'gate_stats:open_vocab'


async def active_vlm_visible(opensearch: AsyncOpenSearch) -> VlmVisibleFn | None:
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
        await refresh_vlm_state(opensearch)
        endpoint = active_vlm_endpoint()
    except VlmEndpointUnavailableError:
        return None
    if endpoint is None:
        return None
    labeler = labeler_for(endpoint, active_prompt_pack())

    async def ask(jpeg: bytes, prompt: str) -> bool | None:
        return await labeler.prompt_visible(jpeg, prompt)

    return ask


async def load_tracker(opensearch: AsyncOpenSearch) -> HitRateTracker:
    try:
        doc = await opensearch.get(index=get_curation_config().configs_index, id=TRACKER_DOC_ID)
    except NotFoundError:
        return HitRateTracker()
    return HitRateTracker.load((doc['_source'].get('body') or {}).get('windows'))


async def save_tracker(opensearch: AsyncOpenSearch, tracker: HitRateTracker) -> None:
    """Persist the windows. Statistics, not state of record: a failed write is
    logged and never fails the pass."""
    body: dict[str, Any] = {'doc_type': 'gate_stats', 'body': {'windows': tracker.dump()}}
    try:
        await opensearch.index(
            index=get_curation_config().configs_index, id=TRACKER_DOC_ID, body=body
        )
    except Exception as exc:
        logger.warning('open_vocab_gate_stats_save_failed', error=str(exc))


__all__ = ['TRACKER_DOC_ID', 'VlmVisibleFn', 'active_vlm_visible', 'load_tracker', 'save_tracker']
