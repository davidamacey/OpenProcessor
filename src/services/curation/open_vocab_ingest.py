"""The ingest-time opt-in of the full-image pass (``run_on_ingest``, off by
default).

Never inline in the ingest request: a segmenter call takes seconds per target.
After the ingest response is built, newly created images of an active set with
``run_on_ingest`` are stamped ``open_vocab_status: pending`` (the durable
record) and drained by a background task of this process, one image at a time
across all requests. An image the drain does not reach (segmenter down, process
restart) stays ``pending`` until the sweeper
(:mod:`~src.services.curation.open_vocab_sweeper`) or a ``POST /reprocess`` with
the image selector ``open_vocab_status: ["pending"]`` and scope ``open_vocab``
picks it up.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.open_vocab_run import SegmentImage, stamp_open_vocab_status
from src.services.curation.reprocess_models import ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass, current_active_set
from src.services.curation.reprocess_targets import existing_images
from src.services.detection.segmenter_http import segment_image_http


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.detection.open_vocab_set import OpenVocabSet

logger = get_logger(__name__)

#: One ingest-time pass at a time per process, so ingest bursts queue behind
#: each other instead of multiplying the segmenter load.
_PASS_LOCK = asyncio.Lock()
_tasks: set[asyncio.Task[None]] = set()


async def drain(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    ov: OpenVocabSet,
    revision: int | None,
    image_ids: list[str],
    segment: SegmentImage,
) -> None:
    async with _PASS_LOCK:
        docs = await existing_images(
            opensearch, image_ids, index=get_curation_config().images_index
        )
        run = await OpenVocabPass.start(
            opensearch, ov, revision, segment, ReprocessScopeResult(scope='open_vocab')
        )
        await run.run_images(opensearch, service, {i: docs[i] for i in image_ids if i in docs})
        if run.tripped:
            logger.warning('open_vocab_ingest_pass_stopped_segmenter_down')
        await run.finish(opensearch)


async def schedule_open_vocab_after_ingest(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    image_ids: list[str],
    *,
    segment: SegmentImage | None = None,
) -> bool:
    """Queue the active set for ``image_ids``. ``False`` (nothing done) when
    there are no images, no active set, or the set does not opt in. A failure
    here is logged and never fails the ingest that just succeeded."""
    if not image_ids:
        return False
    try:
        active = await current_active_set(opensearch)
        if active is None or not active[0].run_on_ingest:
            return False
        ov, revision = active
        for image_id in image_ids:
            await stamp_open_vocab_status(opensearch, image_id, 'pending')
    except Exception as exc:
        logger.warning('open_vocab_ingest_schedule_failed', error=str(exc))
        return False
    coro = drain(opensearch, service, ov, revision, image_ids, segment or segment_image_http)
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


async def wait_for_scheduled() -> None:
    """Await every scheduled drain (tests, graceful shutdown)."""
    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)


__all__ = ['drain', 'schedule_open_vocab_after_ingest', 'wait_for_scheduled']
