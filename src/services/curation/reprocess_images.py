"""Run the image-unit scopes (``detect``, ``embed``) over a list of images.

Shared by the in-request path and the file-backed job, so both execute the
same code on the same inputs: the images doc and the items on it are read
fresh per chunk, so a job that runs long after the request that planned it
works from current state, not from anything the API process held.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.ingest_policy_store import get_ingest_policy
from src.services.curation.reprocess_detect import redetect_image
from src.services.curation.reprocess_embed import ALL_PARTS, new_image_gate, reembed_items
from src.services.curation.reprocess_embed_spec import EmbedSpec, embed_targets
from src.services.curation.reprocess_gpu_embed import gpu_embed_route
from src.services.curation.reprocess_models import ReprocessScope, ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass, active_set_for_run
from src.services.curation.reprocess_targets import existing_images
from src.services.detection.segmenter_http import segment_image_http


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.open_vocab_run import SegmentImage
    from src.services.curation.reprocess_embed import EmbedTarget
    from src.services.curation.reprocess_gpu_embed import GpuEmbedRoute

logger = get_logger(__name__)

CHUNK = 64
EMBED_CHUNKS_IN_FLIGHT = 2
"""Chunks of a pure ``embed`` pass embedded at once: the next chunk starts on its images while the
last ones of this chunk finish and its vectors are written, so no chunk boundary idles the encoder
(#259). They share one in-flight bound of images, so memory does not grow with the window."""
IMAGE_SCOPES: tuple[ReprocessScope, ...] = ('detect', 'open_vocab', 'embed')


def _failed_images(results: dict[ReprocessScope, ReprocessScopeResult]) -> int:
    return max(r.failed + r.not_found for r in results.values())


@dataclass
class ChunkRead:
    """What one chunk needs before it is processed: its images docs and, for the ``embed`` scope,
    the targets read from them. A failed targets read is kept, not raised, so it fails only its
    own chunk when that chunk is reached."""

    docs: dict[str, dict[str, Any]]
    targets: list[EmbedTarget] | BaseException | None = None


async def read_chunk(
    opensearch: AsyncOpenSearch, chunk: list[str], embed: EmbedSpec | None
) -> ChunkRead:
    docs = await existing_images(opensearch, chunk, index=get_curation_config().images_index)
    if embed is None or not docs:
        return ChunkRead(docs)
    (targets,) = await asyncio.gather(
        embed_targets(opensearch, docs, embed), return_exceptions=True
    )
    return ChunkRead(docs, targets)


async def read_chunks(
    opensearch: AsyncOpenSearch, image_ids: list[str], embed: EmbedSpec | None
) -> AsyncGenerator[tuple[list[str], ChunkRead]]:
    """``(chunk ids, their read)`` in order, reading chunk N+1 while the caller handles chunk N,
    so the pass does not wait on a read between chunks (#259). Closing it cancels the read ahead."""

    def start(at: int) -> asyncio.Task[ChunkRead]:
        return asyncio.create_task(read_chunk(opensearch, image_ids[at : at + CHUNK], embed))

    ahead = start(0) if image_ids else None
    try:
        for at in range(0, len(image_ids), CHUNK):
            assert ahead is not None
            read = await ahead
            ahead = start(at + CHUNK) if at + CHUNK < len(image_ids) else None
            yield image_ids[at : at + CHUNK], read
    finally:
        if ahead is not None:
            ahead.cancel()
            await asyncio.gather(ahead, return_exceptions=True)


async def embed_image_chunk(
    opensearch: AsyncOpenSearch,
    pe: Any,
    docs: dict[str, dict[str, Any]],
    spec: EmbedSpec,
    res: ReprocessScopeResult,
    targets: list[EmbedTarget] | BaseException | None = None,
    gpu: GpuEmbedRoute | None = None,
    gate: asyncio.Semaphore | None = None,
) -> None:
    """The ``embed`` scope over one chunk of existing images (``docs`` maps
    image id to its images doc), accumulating into ``res``. ``targets`` is the
    chunk's already-read targets (:func:`read_chunk`); read here when absent.
    ``gpu`` embeds large photos on the GPU decode; ``gate`` is the pass's shared in-flight bound
    of images. An encoder failure counts
    every image of the chunk as failed."""
    try:
        if isinstance(targets, BaseException):
            raise targets
        policy = await get_ingest_policy(opensearch)
        counts = await reembed_items(
            opensearch,
            pe,
            targets if targets is not None else await embed_targets(opensearch, docs, spec),
            parts=spec.part_set,
            only_missing=spec.only_missing,
            place_in_clusters=policy.clustering.automatic,
            gpu=gpu,
            gate=gate,
        )
    except Exception as exc:
        logger.warning('reprocess_embed_failed', error=str(exc))
        res.failed += len(docs)
        return
    res.queued += counts['images']
    res.failed += counts['missing_image']
    for key in ('items', 'crop_written', 'frame_written', 'region_written'):
        res.add_count(key, counts[key])


async def _detect_chunk(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    docs: dict[str, dict[str, Any]],
    res: ReprocessScopeResult,
) -> None:
    for image_id, doc in docs.items():
        try:
            counts = await redetect_image(opensearch, service, image_id, doc)
        except Exception as exc:
            logger.warning('reprocess_detect_failed', image_id=image_id, error=str(exc))
            res.failed += 1
            continue
        res.queued += 1
        res.locked_skipped += counts['locked_untouched']
        for key in ('merged', 'refreshed', 'replaced', 'created', 'removed'):
            res.add_count(key, counts[key])


class _ChunkWindow:
    """The chunks being embedded at once and the images done so far, counted in chunk order."""

    def __init__(
        self,
        size: int,
        results: dict[ReprocessScope, ReprocessScopeResult],
        on_progress: Callable[[int, int], None] | None,
    ) -> None:
        self.size = size
        self.done = 0
        self._results = results
        self._on_progress = on_progress
        self._running: deque[tuple[asyncio.Task[None], int]] = deque()

    def start(self, chunk_size: int, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._running.append((task, chunk_size))
        return task

    def count(self, size: int) -> None:
        self.done += size
        if self._on_progress is not None:
            self._on_progress(self.done, _failed_images(self._results))

    async def settle(self, keep: int) -> None:
        """Wait for the oldest chunks until at most ``keep`` run."""
        while len(self._running) > keep:
            task, size = self._running.popleft()
            await task
            self.count(size)

    async def abandon(self) -> None:
        """Cancel what still runs (the job ended abnormally) and wait for it to stop."""
        tasks = [task for task, _ in self._running]
        self._running.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def process_images(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    *,
    scopes: list[ReprocessScope],
    image_ids: list[str],
    should_cancel: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int], None] | None = None,
    embed: EmbedSpec | None = None,
    segment: SegmentImage | None = None,
) -> tuple[list[ReprocessScopeResult], bool]:
    """Run ``scopes`` (a subset of :data:`IMAGE_SCOPES`) over ``image_ids``.

    ``embed`` says which items and parts the ``embed`` scope covers (default:
    every item on the images, every part, rewritten).

    Returns ``(per-scope results, cancelled)``. An image that does not exist
    counts ``not_found``; one whose path is unservable or that raises counts
    ``failed`` for that scope; the rest continue.
    """
    embed_spec = embed or EmbedSpec(parts=sorted(ALL_PARTS))
    results = {s: ReprocessScopeResult(scope=s, selected=len(image_ids)) for s in scopes}
    ov_pass: OpenVocabPass | None = None
    if 'open_vocab' in scopes:
        ov, revision = await active_set_for_run(opensearch)
        ov_pass = await OpenVocabPass.start(
            opensearch, ov, revision, segment or segment_image_http, results['open_vocab']
        )
    cancelled = False
    # Targets are read ahead only when nothing else runs first on a chunk: detect and open-vocab
    # change the items, and the embed must see them as they are after. The same rule lets chunks
    # overlap: only then is a chunk's embed independent of the one before it.
    pure_embed = set(scopes) == {'embed'}
    chunk_embed = embed_spec if pure_embed else None
    window = _ChunkWindow(EMBED_CHUNKS_IN_FLIGHT if pure_embed else 1, results, on_progress)
    gpu = await gpu_embed_route(service) if 'embed' in scopes else None
    gate = new_image_gate() if 'embed' in scopes else None
    try:
        async with aclosing(read_chunks(opensearch, image_ids, chunk_embed)) as chunks:
            async for chunk, read in chunks:
                if should_cancel():
                    cancelled = True
                    break
                docs = read.docs
                missing = [i for i in chunk if i not in docs]
                for scope in scopes:
                    results[scope].not_found += len(missing)
                if 'detect' in scopes:
                    await _detect_chunk(opensearch, service, docs, results['detect'])
                if ov_pass is not None:
                    finished = 0

                    def _image_done(base: int = window.done) -> None:
                        nonlocal finished
                        finished += 1
                        if on_progress is not None:
                            on_progress(base + finished, _failed_images(results))

                    cancelled |= await ov_pass.run_images(
                        opensearch, service, docs, should_cancel=should_cancel, on_image=_image_done
                    )
                if 'embed' in scopes and docs:
                    embedding = embed_image_chunk(
                        opensearch,
                        service.pe_encoder,
                        docs,
                        embed_spec,
                        results['embed'],
                        read.targets,
                        gpu,
                        gate,
                    )
                    window.start(len(chunk), embedding)
                    await window.settle(window.size - 1)
                else:
                    await window.settle(0)
                    window.count(len(chunk))
        await window.settle(0)
    except BaseException:
        await window.abandon()
        raise
    if ov_pass is not None:
        await ov_pass.finish(opensearch)
    return [results[s] for s in scopes], cancelled


__all__ = ['CHUNK', 'IMAGE_SCOPES', 'embed_image_chunk', 'process_images']
