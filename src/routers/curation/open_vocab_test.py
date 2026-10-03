"""``POST /open_vocab/test``: try one unsaved target on one image.

Runs the real pass's read-only half (segment, select, dedup against the
image's items) and returns every candidate with its box, outline and what
selection did with it. Writes nothing. The client draws the boxes: the
backend serves clean data, no overlay.
"""

from __future__ import annotations

from src.routers.curation._common import OpenSearchDep, images_index, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._open_vocab_models import (
    OpenVocabTestHit,
    OpenVocabTestImage,
    OpenVocabTestRequest,
    OpenVocabTestResponse,
)
from src.routers.curation.open_vocab import validation_inputs
from src.services.config_store.open_vocab_validation import validate_open_vocab
from src.services.curation.open_vocab_test_run import (
    TrialImageError,
    decode_upload,
    load_stored_image,
    run_open_vocab_test,
)
from src.services.curation.reprocess_targets import existing_images
from src.services.detection.segmenter_http import SegmenterCallError, segment_image_http


@router.post('/open_vocab/test', response_model=OpenVocabTestResponse)
async def test_open_vocab_target(
    body: OpenVocabTestRequest, opensearch: OpenSearchDep
) -> OpenVocabTestResponse:
    """404 ``image_not_found``; 422 ``validation_failed`` (the target, or not
    exactly one image source); 502 ``segmenter_error`` when the segmenter
    cannot answer (never reported as "found nothing"). Writes nothing."""
    if (body.image_id is None) == (body.image_base64 is None):
        raise api_error(422, 'validation_failed', 'give exactly one of image_id and image_base64')
    draft = {
        'targets': [body.target.model_dump()],
        'image_max_side': body.image_max_side,
        'dedup_iou': body.dedup_iou,
    }
    report = await validate_open_vocab(None, draft, **validation_inputs())
    if not report.ok:
        raise api_error(422, 'validation_failed', 'the target has errors', report=report)

    image_id = body.image_id or ''
    try:
        if body.image_id is not None:
            found = await existing_images(opensearch, [body.image_id], index=images_index())
            if body.image_id not in found:
                raise api_error(404, 'image_not_found', f'no image {body.image_id}')
            pil = await load_stored_image(found[body.image_id])
        else:
            assert body.image_base64 is not None
            pil = decode_upload(body.image_base64)
    except TrialImageError as exc:
        raise api_error(422, 'validation_failed', str(exc)) from exc
    try:
        outcome = await run_open_vocab_test(
            opensearch,
            image_id=image_id,
            pil=pil,
            target_body=body.target.model_dump(),
            image_max_side=body.image_max_side,
            dedup_iou=body.dedup_iou,
            segment=segment_image_http,
        )
    except SegmenterCallError as exc:
        raise api_error(502, 'segmenter_error', str(exc)) from exc
    return OpenVocabTestResponse(
        image=OpenVocabTestImage(width=outcome.width, height=outcome.height),
        prompt=body.target.prompt,
        class_name=body.target.class_name,
        hits=[
            OpenVocabTestHit(
                bbox_norm=list(h.bbox_norm),
                score=h.score,
                mask_polygon=[list(p) for p in h.mask_polygon] if h.mask_polygon else None,
                selected=h.selected,
                drop_reason=h.drop_reason,
            )
            for h in outcome.hits
        ],
        elapsed_ms=outcome.elapsed_ms,
        validation=report,
    )
