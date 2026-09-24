"""B-PR5 + Phase C wireup tests — ensure the broadened low-class-confidence
cohort takes the single combined VLM call path instead of the legacy
verify + class chain.

Deferred here from Chunk 7 because it exercises
``scripts.curation.sam_worker_main``'s
``_process_crop`` cohort-routing logic, which needs the worker package
that lands in this chunk.

Cohort detection criterion (mirrors ``combined._is_combined_cohort``):

* ``task.class_source == 'coco_yolo11_proposal'`` (the primary
  classifier missed entirely), OR
* ``task.class_source in {'item_model', 'cluster_majority_agreement'}``
  AND ``task.class_confidence < 0.80`` (the primary classifier fired
  but low-confidence — class still needs VLM clarification),
* AND a region candidate exists (either pending_verification with an
  existing primary-detector bbox, OR pending_detection where the
  primary detector / secondary segmenter produced one).

When the cohort fires, ``VlmLabeler.label_combined`` must be invoked
exactly once and the legacy ``verify_plate`` / ``label_or_propose_batch``
paths must NOT be called for that crop. The bulk-update doc must carry
``vlm_verify_completed_at``, ``class_source='vlm'``, and the
make/model fields when reported.

Note: the reference file's ``TestPipelineSkipFilter`` class (asserting
the reference pipeline router's unvalidated-crops query excludes
recent ``vlm_verify_completed_at`` writes) is NOT ported here — that
module (``src/routers/curation/pipeline.py``) is out of this chunk's
scope (Chunk 9 in the plan's target layout). Those two cases stay
unported until the pipeline router lands.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import scripts.curation.sam_worker_main as worker
from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import VlmCombinedReply, VlmRegionVerdict


# The cascade needs an active region profile; the default is none.
pytestmark = pytest.mark.usefixtures('reference_region_profile')


F = get_region_fields()


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_task(
    *,
    crop_id: str = 'crop-1',
    status: str | None = 'pending_verification',
    class_source: str = 'coco_yolo11_proposal',
    class_confidence: float = 0.0,
    lpr_in_source: tuple[float, float, float, float] | None = (0.20, 0.30, 0.30, 0.34),
    lpr_score: float = 0.91,
) -> worker._ItemTask:
    return worker._ItemTask(
        crop_id=crop_id,
        image_path='/dev/null/never-read',
        vehicle_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        plate_status=status,
        class_name='',
        group='cars',
        class_source=class_source,
        class_confidence=class_confidence,
        lpr_plate_in_source=lpr_in_source,
        lpr_score=lpr_score,
        crop_jpeg=_make_jpeg(),
    )


def _gemma_with_combined(
    *,
    reply: VlmCombinedReply,
    class_names: list[str] | None = None,
) -> MagicMock:
    g = MagicMock()
    g.class_names = class_names or ['sedan', 'pickup', 'audi']
    # Authoritative class_name -> class_id map the worker reads via
    # ``gemma.name_to_id`` to resolve the combined reply's class. Without a
    # real dict here, MagicMock's auto-attr makes ``name_to_id.get(...)``
    # return a mock instead of the resolved id.
    g.name_to_id = {name: i for i, name in enumerate(g.class_names)}
    g.label_combined = AsyncMock(return_value=reply)
    g.verify_plate = AsyncMock(
        return_value=VlmRegionVerdict(
            crop_id='ignored', is_region=False, confidence='low', reason='unused'
        )
    )
    g.label_or_propose_batch = AsyncMock(return_value=[])
    g.aclose = AsyncMock()
    return g


class TestCohortRouting:
    @pytest.mark.asyncio
    async def test_pending_verification_cohort_uses_combined_call(self) -> None:
        """primary-missed + existing bbox → one label_combined call, no verify_plate."""
        reply = VlmCombinedReply(
            img_id='crop-1',
            class_id=2,  # 'audi'
            class_confidence='high',
            plate_visible=True,
            plate_bbox_correct=True,
            plate_text='XYZ-9999',
            plate_confidence='high',
            make='Audi',
            model='A4',
        )
        gemma = _gemma_with_combined(reply=reply)
        task = _make_task()
        await worker._process_crop(
            task,
            lpr=MagicMock(detect_batch=AsyncMock(return_value=[])),
            sam3=MagicMock(segment_plate=AsyncMock(return_value=None), aclose=AsyncMock()),
            ocr_recognizer=MagicMock(
                detect_regions=AsyncMock(return_value=[]),
                pick_best_plate_region=MagicMock(return_value=None),
            ),
            gemma=gemma,
        )

        gemma.label_combined.assert_awaited_once()
        gemma.verify_plate.assert_not_awaited()
        gemma.label_or_propose_batch.assert_not_awaited()

        doc = task.update_doc
        assert doc[F.status] == 'detected'
        assert doc['class_source'] == 'vlm'
        assert doc['class_id'] == 2
        assert doc['class_name'] == 'audi'
        assert doc['vlm_item_make'] == 'Audi'
        assert doc['vlm_item_model'] == 'A4'
        assert 'vlm_verify_completed_at' in doc

    @pytest.mark.asyncio
    async def test_broadened_cohort_v6_low_conf_with_plate_uses_label_combined(self) -> None:
        """Phase C: primary-classifier + low conf + candidate → label_combined."""
        reply = VlmCombinedReply(
            img_id='crop-low',
            class_id=1,
            class_confidence='medium',
            plate_visible=True,
            plate_bbox_correct=True,
            plate_text='LOW-0001',
            plate_confidence='medium',
        )
        gemma = _gemma_with_combined(reply=reply)
        task = _make_task(
            crop_id='crop-low',
            class_source='item_model',
            class_confidence=0.60,
            status='pending_verification',
            lpr_in_source=(0.20, 0.30, 0.30, 0.34),
        )
        await worker._process_crop(
            task,
            lpr=MagicMock(detect_batch=AsyncMock(return_value=[])),
            sam3=MagicMock(segment_plate=AsyncMock(return_value=None), aclose=AsyncMock()),
            ocr_recognizer=MagicMock(
                detect_regions=AsyncMock(return_value=[]),
                pick_best_plate_region=MagicMock(return_value=None),
            ),
            gemma=gemma,
        )
        gemma.label_combined.assert_awaited_once()
        gemma.verify_plate.assert_not_awaited()
        gemma.label_or_propose_batch.assert_not_awaited()
        assert task.update_doc[F.status] == 'detected'
        assert task.update_doc['class_source'] == 'vlm'
        assert 'vlm_verify_completed_at' in task.update_doc

    @pytest.mark.asyncio
    async def test_high_conf_v6_skips_label_combined(self) -> None:
        """Phase C: primary classifier with conf >= 0.80 → legacy 2-call path."""
        gemma = _gemma_with_combined(reply=VlmCombinedReply(img_id='x', class_id=None))
        gemma.verify_plate = AsyncMock(
            return_value=VlmRegionVerdict(
                crop_id='c', is_region=True, confidence='high', reason='ok'
            )
        )
        task = _make_task(
            class_source='item_model',
            class_confidence=0.95,
            status='pending_verification',
            lpr_in_source=(0.20, 0.30, 0.30, 0.34),
        )
        await worker._process_crop(
            task,
            lpr=MagicMock(detect_batch=AsyncMock(return_value=[])),
            sam3=MagicMock(segment_plate=AsyncMock(return_value=None), aclose=AsyncMock()),
            ocr_recognizer=MagicMock(
                detect_regions=AsyncMock(return_value=[]),
                pick_best_plate_region=MagicMock(return_value=None),
            ),
            gemma=gemma,
        )
        gemma.label_combined.assert_not_awaited()
        gemma.verify_plate.assert_awaited()
        assert task.update_doc[F.status] == 'detected'
        assert 'vlm_verify_completed_at' not in task.update_doc

    @pytest.mark.asyncio
    async def test_no_region_candidate_uses_class_only_path(self) -> None:
        """Phase C: cohort crop without a candidate → no combined call.

        The curation worker only handles the region-detection cascade;
        class-only VLM runs are the pipeline's job. A pending_detection
        cohort crop where the primary detector + secondary segmenter
        both miss results in a terminal ``no_region_box`` write with no
        combined call (and no label_or_propose_batch, since class-only
        labeling belongs to the pipeline, not this worker).
        """
        gemma = _gemma_with_combined(reply=VlmCombinedReply(img_id='x', class_id=None))
        task = _make_task(
            class_source='item_model',
            class_confidence=0.60,
            status='pending_detection',
            lpr_in_source=None,
            lpr_score=0.0,
        )
        await worker._process_crop(
            task,
            lpr=MagicMock(detect_batch=AsyncMock(return_value=[])),
            sam3=MagicMock(segment_plate=AsyncMock(return_value=None), aclose=AsyncMock()),
            ocr_recognizer=MagicMock(
                detect_regions=AsyncMock(return_value=[]),
                pick_best_plate_region=MagicMock(return_value=None),
            ),
            gemma=gemma,
        )
        gemma.label_combined.assert_not_awaited()
        gemma.label_or_propose_batch.assert_not_awaited()
        assert task.update_doc[F.status] == 'no_region_box'

    @pytest.mark.asyncio
    async def test_pending_detection_cohort_lpr_hit_uses_combined(self) -> None:
        """primary-missed + pending_detection + primary-detector candidate → one combined call."""
        lpr_cand = RegionCandidate(
            bbox_norm=(0.3, 0.4, 0.5, 0.45),
            score=0.82,
            source='lpr_nanov11_640',
        )
        reply = VlmCombinedReply(
            img_id='crop-2',
            class_id=0,  # 'sedan'
            class_confidence='medium',
            plate_visible=True,
            plate_bbox_correct=True,
            plate_text='AAA1111',
            plate_confidence='medium',
        )
        gemma = _gemma_with_combined(reply=reply)
        task = _make_task(
            crop_id='crop-2',
            status='pending_detection',
            lpr_in_source=None,
            lpr_score=0.0,
        )
        await worker._process_crop(
            task,
            lpr=MagicMock(detect_batch=AsyncMock(return_value=[lpr_cand])),
            sam3=MagicMock(segment_plate=AsyncMock(return_value=None), aclose=AsyncMock()),
            ocr_recognizer=MagicMock(
                detect_regions=AsyncMock(return_value=[]),
                pick_best_plate_region=MagicMock(return_value=None),
            ),
            gemma=gemma,
        )
        gemma.label_combined.assert_awaited_once()
        gemma.verify_plate.assert_not_awaited()
        assert task.update_doc[F.status] == 'detected'
        assert task.update_doc['class_id'] == 0
        assert task.update_doc['class_source'] == 'vlm'
        assert 'vlm_verify_completed_at' in task.update_doc
