"""Run the image-unit scopes (``detect``, ``embed``) over a list of images.

Shared by the in-request path and the file-backed job, so both execute the
same code on the same inputs: the images doc and the items on it are read
fresh per chunk, so a job that runs long after the request that planned it
works from current state, not from anything the API process held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.region_fields import get_region_fields
from src.core.logging import get_logger
from src.services.curation.reprocess_detect import redetect_image
from src.services.curation.reprocess_embed import ALL_PARTS, EmbedTarget, reembed_items
from src.services.curation.reprocess_models import ReprocessScope, ReprocessScopeResult
from src.services.curation.reprocess_open_vocab import OpenVocabPass, active_set_for_run
from src.services.curation.reprocess_targets import existing_images, items_by_terms
from src.services.detection.segmenter_http import segment_image_http


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch

    from src.services.curation.ingest import CurationIngestService
    from src.services.curation.open_vocab_run import SegmentImage

logger = get_logger(__name__)

CHUNK = 20
IMAGE_SCOPES: tuple[ReprocessScope, ...] = ('detect', 'open_vocab', 'embed')


async def _embed_targets(
    opensearch: AsyncOpenSearch, docs: dict[str, dict[str, Any]]
) -> list[EmbedTarget]:
    F = get_region_fields()
    items = await items_by_terms(
        opensearch,
        'image_id',
        list(docs),
        index=get_curation_config().items_index,
        includes=['image_id', 'bbox_norm', F.boxes],
    )
    by_image: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for crop_id, src in items:
        by_image.setdefault(src.get('image_id') or '', []).append((crop_id, src))
    return [
        EmbedTarget(
            image_id=image_id,
            image_path=doc.get('image_path') or '',
            items=by_image.get(image_id, []),
        )
        for image_id, doc in docs.items()
    ]


def _failed_images(results: dict[ReprocessScope, ReprocessScopeResult]) -> int:
    return max(r.failed + r.not_found for r in results.values())


async def process_images(
    opensearch: AsyncOpenSearch,
    service: CurationIngestService,
    *,
    scopes: list[ReprocessScope],
    image_ids: list[str],
    should_cancel: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int], None] | None = None,
    segment: SegmentImage | None = None,
) -> tuple[list[ReprocessScopeResult], bool]:
    """Run ``scopes`` (a subset of :data:`IMAGE_SCOPES`) over ``image_ids``.

    Returns ``(per-scope results, cancelled)``. An image that does not exist
    counts ``not_found``; one whose path is unservable or that raises counts
    ``failed`` for that scope; the rest continue.
    """
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
                    res.detail[key] = res.detail.get(key, 0) + counts[key]
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
            try:
                counts = await reembed_items(
                    opensearch,
                    service.pe_encoder,
                    await _embed_targets(opensearch, docs),
                    parts=ALL_PARTS,
                )
            except Exception as exc:
                logger.warning('reprocess_embed_failed', error=str(exc))
                results['embed'].failed += len(docs)
            else:
                res = results['embed']
                res.queued += counts['images']
                res.failed += counts['missing_image']
                for key in ('items', 'crop_written', 'frame_written', 'region_written'):
                    res.detail[key] = res.detail.get(key, 0) + counts[key]
        done += len(chunk)
        failed_images = _failed_images(results)
        if on_progress is not None:
            on_progress(done, failed_images)
    if ov_pass is not None:
        await ov_pass.finish(opensearch)
    return [results[s] for s in scopes], cancelled


__all__ = ['CHUNK', 'IMAGE_SCOPES', 'process_images']
