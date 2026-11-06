"""The ingest-time opt-in of the full-image pass (``run_on_ingest``, off by default).

Never inline in the ingest request: a segmenter call takes seconds per target. After the ingest
response is built, newly created images of an active set with ``run_on_ingest`` are stamped
``open_vocab_status: pending`` (the durable record) and the project's ``open_vocab_drain`` job is
started. The job is exclusive per project: a start while one is queued or running is answered
"already pending" and adds nothing, because the job takes every ``pending`` image of the project,
and rescans until none is left that it has not tried. An image the drain does not reach (segmenter
down) stays ``pending`` for the next drain, which the periodic sweep starts, or for a
``POST /reprocess`` with the image selector ``open_vocab_status: ["pending"]`` and scope
``open_vocab``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import get_curation_config
from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.session import async_session_factory
from src.services.curation.open_vocab_run import SegmentImage, stamp_open_vocab_status
from src.services.curation.reprocess_models import ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass, current_active_set
from src.services.curation.reprocess_targets import existing_images, scan_items
from src.services.jobs.service import start_job


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService

logger = get_logger(__name__)

#: Images one batch takes on; the drain rescans for the next batch.
DRAIN_BATCH = 500


async def start_drain() -> bool:
    """Start the bound project's ``open_vocab_drain`` job.

    ``True`` when a job was started, ``False`` when one is already queued or running ("already
    pending": the unique index of the exclusive group coalesces the starts). A broker or database
    failure raises; the pending stamps stay, so a later start loses nothing.
    """
    async with async_session_factory()() as session:
        try:
            await start_job(session, 'open_vocab_drain', current_project_id())
        except RequestRejectedError as exc:
            if exc.detail.get('error') != 'job_busy':
                raise
            return False
    return True


async def schedule_open_vocab_after_ingest(
    opensearch: AsyncOpenSearch, image_ids: list[str]
) -> bool:
    """Stamp ``image_ids`` pending and start the drain. ``False`` (nothing done) when there are
    no images, no active set, or the set does not opt in. A failure here is logged and never
    fails the ingest that just succeeded."""
    if not image_ids:
        return False
    try:
        active = await current_active_set()
        if active is None or not active[0].run_on_ingest:
            return False
        for image_id in image_ids:
            await stamp_open_vocab_status(opensearch, image_id, 'pending')
        await start_drain()
    except Exception as exc:
        logger.warning('open_vocab_ingest_schedule_failed', error=str(exc))
        return False
    return True


async def drain_pending(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    segment: SegmentImage,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    on_batch: Callable[[int], Awaitable[None]] | None = None,
) -> dict[str, int]:
    """Run the active set over every ``pending`` image of the bound project.

    Takes ``DRAIN_BATCH`` images at a time and rescans, so images stamped while the drain ran are
    reached; an image already tried in this drain is not taken twice, so one that fails and stays
    ``pending`` cannot loop it. Stops when the segmenter is down (the images stay ``pending``) or
    ``should_stop`` says so. Returns ``{'drained': n, 'segmenter_down': 0|1}``.
    """
    active = await current_active_set()
    if active is None or not active[0].run_on_ingest:
        return {'drained': 0, 'segmenter_down': 0}
    ov, revision = active
    index = get_curation_config().images_index
    tried: set[str] = set()
    tripped = False
    while not should_stop():
        pending = await scan_items(
            opensearch,
            {'term': {'open_vocab_status': 'pending'}},
            index=index,
            includes=['open_vocab_status_at'],
            id_field='image_id',
            max_docs=DRAIN_BATCH + len(tried),
        )
        ids = [i for i, _ in pending if i not in tried][:DRAIN_BATCH]
        if not ids:
            break
        tried.update(ids)
        docs = await existing_images(opensearch, ids, index=index)
        run = await OpenVocabPass.start(
            ov, revision, segment, ReprocessScopeResult(scope='open_vocab')
        )
        await run.run_images(
            opensearch, service, {i: docs[i] for i in ids if i in docs}, should_cancel=should_stop
        )
        await run.finish()
        if on_batch is not None:
            await on_batch(len(tried))
        if run.tripped:
            logger.warning('open_vocab_ingest_pass_stopped_segmenter_down')
            tripped = True
            break
    return {'drained': len(tried), 'segmenter_down': int(tripped)}


__all__ = ['drain_pending', 'schedule_open_vocab_after_ingest', 'start_drain']
