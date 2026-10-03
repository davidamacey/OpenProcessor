"""Run the image-unit scopes (``detect``, ``embed``) over a list of images.

Shared by the in-request path and the file-backed job, so both execute the
same code on the same inputs: the images doc and the items on it are read
fresh per chunk, so a job that runs long after the request that planned it
works from current state, not from anything the API process held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.curation.reprocess_detect import redetect_image
from src.services.curation.reprocess_embed import ALL_PARTS, reembed_items
from src.services.curation.reprocess_embed_spec import EmbedSpec, embed_targets
from src.services.curation.reprocess_models import ReprocessScope, ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass, active_set_for_run
from src.services.curation.reprocess_targets import existing_images
from src.services.detection.segmenter_http import segment_image_http


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.open_vocab_run import SegmentImage

logger = get_logger(__name__)

CHUNK = 20
IMAGE_SCOPES: tuple[ReprocessScope, ...] = ('detect', 'open_vocab', 'embed')


def _failed_images(results: dict[ReprocessScope, ReprocessScopeResult]) -> int:
    return max(r.failed + r.not_found for r in results.values())


async def embed_image_chunk(
    opensearch: AsyncOpenSearch,
    pe: Any,
    docs: dict[str, dict[str, Any]],
    spec: EmbedSpec,
    res: ReprocessScopeResult,
) -> None:
    """The ``embed`` scope over one chunk of existing images (``docs`` maps
    image id to its images doc), accumulating into ``res``. An encoder failure
    counts every image of the chunk as failed."""
    try:
        counts = await reembed_items(
            opensearch,
            pe,
            await embed_targets(opensearch, docs, spec),
            parts=spec.part_set,
            only_missing=spec.only_missing,
        )
    except Exception as exc:
        logger.warning('reprocess_embed_failed', error=str(exc))
        res.failed += len(docs)
        return
    res.queued += counts['images']
    res.failed += counts['missing_image']
    for key in ('items', 'crop_written', 'frame_written', 'region_written'):
        res.add_count(key, counts[key])


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
    done = failed_images = 0
    cancelled = False
    for start in range(0, len(image_ids), CHUNK):
        if should_cancel():
            cancelled = True
            break
        chunk = image_ids[start : start + CHUNK]
        docs = await existing_images(opensearch, chunk, index=get_curation_config().images_index)
        missing = [i for i in chunk if i not in docs]
        for scope in scopes:
            results[scope].not_found += len(missing)
        if 'detect' in scopes:
            res = results['detect']
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
        if ov_pass is not None:
            finished = 0

            def _image_done(base: int = done) -> None:
                nonlocal finished
                finished += 1
                if on_progress is not None:
                    on_progress(base + finished, _failed_images(results))

            cancelled |= await ov_pass.run_images(
                opensearch, service, docs, should_cancel=should_cancel, on_image=_image_done
            )
        if 'embed' in scopes and docs:
            await embed_image_chunk(
                opensearch, service.pe_encoder, docs, embed_spec, results['embed']
            )
        done += len(chunk)
        failed_images = _failed_images(results)
        if on_progress is not None:
            on_progress(done, failed_images)
    if ov_pass is not None:
        await ov_pass.finish(opensearch)
    return [results[s] for s in scopes], cancelled


__all__ = ['CHUNK', 'IMAGE_SCOPES', 'embed_image_chunk', 'process_images']
