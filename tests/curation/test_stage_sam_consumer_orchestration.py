"""``stage_a_sam_consumer`` routing, pinned with fakes for the segmenter, gate, OCR and
the candidate helpers: where each outcome of the secondary-segmenter pass sends the
task (writer queue, combined VLM queue, or released for a retry), what it records in
the detector trace, and that ``sam_q`` bookkeeping stays balanced on every path."""

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from scripts.curation.worker import stage_sam
from scripts.curation.worker.client import SegmenterUnavailable
from scripts.curation.worker.no_verdict import NoVerdictCounter
from scripts.curation.worker.pipeline import PipelineContext
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.detection.segmenter_gate import RUN, GateDecision


SEG = 'sam-x'


def _task(crop_id: str = 'c1', *, jpeg: bytes | None = b'jpeg') -> _ItemTask:
    return _ItemTask(
        crop_id=crop_id,
        image_path=f'/{crop_id}.jpg',
        item_bbox_norm=(0.1, 0.1, 0.9, 0.9),
        region_status='pending_detection',
        class_name='',
        request_id=f'req-{crop_id}',
        crop_jpeg=jpeg,
        project=SimpleNamespace(slug='alpha'),  # type: ignore[arg-type]
    )


def _cand(score: float = 0.99) -> Any:
    return SimpleNamespace(
        score=score,
        bbox_in_crop=(0.2, 0.2, 0.6, 0.5),
        bbox_in_source=(0.3, 0.3, 0.5, 0.4),
    )


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.gate = MagicMock()
        self.gate.decide = AsyncMock(return_value=RUN)
        self.segmenter = MagicMock()
        self.segmenter.enabled = True
        self.segmenter.segment_multi = AsyncMock(return_value=[])
        self.ocr = MagicMock()
        self.ocr.regions_from_lines = MagicMock(return_value=[])
        self.ocr.detect_regions = AsyncMock(return_value=[])
        self.ocr.pick_best_text_region = MagicMock(return_value=None)
        self.rt = SimpleNamespace(
            profile=SimpleNamespace(
                segmenter_name=SEG, segmenter_version='v1', ocr_rec_model='rec-x'
            ),
            segmenter=self.segmenter,
            ocr_recognizer=self.ocr,
            text_rules='rules-x',
            vlm_available=True,
            vlm_model='model-x',
            text_hint_on=False,
        )
        self.paused = False
        self.selected: list[Any] = []
        self.plausible = True
        self.resegment: Any = (None, None)
        self.calls: list[tuple[str, Any]] = []
        self.sleeps: list[float] = []
        self.sam_metric = MagicMock()
        self.request_metrics: list[tuple[str, float]] = []
        self.ctx = PipelineContext(
            args=MagicMock(),
            stop_event=asyncio.Event(),
            opensearch=MagicMock(),
            sentinel=MagicMock(),
            started_at=0.0,
            region_embed_pe=None,
            project_registry=MagicMock(),
            fairness_scheduler=MagicMock(),
            runtime_holder=MagicMock(),
            in_q=asyncio.Queue(),
            vlm_visible_q=asyncio.Queue(),
            sam_q=asyncio.Queue(),
            combined_q=asyncio.Queue(),
            out_q=asyncio.Queue(),
            in_flight=set(),
            in_flight_owner={},
            in_flight_lock=asyncio.Lock(),
            released_at={},
            metrics=Counter(),  # type: ignore[arg-type]
            no_verdict_cap=3,
            visible_no_verdict=NoVerdictCounter(3),
            combined_no_verdict=NoVerdictCounter(3),
            crop_gate=self.gate,
        )
        self.ctx.rt_for = lambda _t: self.rt  # type: ignore[method-assign]
        m = monkeypatch
        m.setattr(stage_sam, 'bind_task_project', lambda t: self.calls.append(('bind', t.crop_id)))
        m.setattr(stage_sam, 'is_region_stage_paused', lambda _p: self.paused)
        m.setattr(stage_sam, '_select_candidates', self._select)
        m.setattr(stage_sam, '_bbox_shape_is_plausible', lambda _b: self.plausible)
        m.setattr(stage_sam, '_resegment_from_text_hint', self._resegment_from_text_hint)
        m.setattr(stage_sam, 'apply_region_text', self._apply_region_text)
        m.setattr(stage_sam, '_box_with_resolved_text', lambda box, _doc: box)
        m.setattr(stage_sam, 'accept_without_vlm', self._accept_without_vlm)
        m.setattr(stage_sam, '_sync_singular_candidate', lambda t: self.calls.append(('sync', t)))
        m.setattr(
            stage_sam,
            '_box_list_doc',
            lambda _t, boxes, status, *, extra=None: {
                'boxes': boxes,
                'status': status,
                'extra': extra,
            },
        )
        m.setattr(stage_sam, 'OP_STAGE_A_SEGMENTER_DURATION_SECONDS', self.sam_metric)
        m.setattr(
            stage_sam, 'record_segmenter_request', lambda o, s: self.request_metrics.append((o, s))
        )
        m.setattr(stage_sam.asyncio, 'sleep', self._sleep)

    def _select(self, raw: Any, **kw: Any) -> list[Any]:
        self.calls.append(('select', (raw, kw)))
        return self.selected

    async def _resegment_from_text_hint(self, jpeg: bytes, bbox: Any, _segmenter: Any) -> Any:
        self.calls.append(('resegment', (jpeg, bbox)))
        if isinstance(self.resegment, Exception):
            raise self.resegment
        return self.resegment

    async def _apply_region_text(self, doc: dict[str, Any], **kw: Any) -> None:
        self.calls.append(('apply_text', kw))
        doc['applied'] = True

    async def _accept_without_vlm(self, t: _ItemTask, **_kw: Any) -> None:
        self.calls.append(('accept_without_vlm', t.crop_id))
        t.update_doc = {'accepted_without_vlm': True}

    async def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    async def run(self, *tasks: _ItemTask) -> None:
        for t in tasks:
            self.ctx.in_flight.add(t.crop_id)
            await self.ctx.sam_q.put(t)
        await self.ctx.sam_q.put(None)
        await asyncio.wait_for(stage_sam.stage_a_sam_consumer(self.ctx, 2), 5)
        # Every dequeued entry was marked done exactly once.
        await asyncio.wait_for(self.ctx.sam_q.join(), 1)

    def out(self) -> list[_ItemTask]:
        return _drain(self.ctx.out_q)

    def combined(self) -> list[_ItemTask]:
        return _drain(self.ctx.combined_q)


def _drain(q: asyncio.Queue[Any]) -> list[Any]:
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(monkeypatch)


F = get_region_fields()


@pytest.mark.asyncio
async def test_unreadable_crop_is_written_as_failed(h: Harness) -> None:
    t = _task(jpeg=None)
    await h.run(t)
    assert h.out() == [t]
    assert RegionStatus.DETECTION_FAILED in t.update_doc.values()
    assert 'image_unavailable' in t.update_doc.values()
    h.segmenter.segment_multi.assert_not_awaited()


@pytest.mark.asyncio
async def test_paused_project_releases_the_task_untouched(h: Harness) -> None:
    h.paused = True
    t = _task()
    await h.run(t)
    assert h.out() == []
    assert h.ctx.in_flight == set()
    assert t.update_doc == {}
    h.segmenter.segment_multi.assert_not_awaited()


@pytest.mark.asyncio
async def test_gate_skip_writes_no_region_box_with_the_gate_label(h: Harness) -> None:
    h.gate.decide = AsyncMock(return_value=GateDecision(run=False, tier=3, reason='hit_rate'))
    t = _task()
    await h.run(t)
    assert h.out() == [t]
    assert t.detection_trace == ['gate:tier3_hit_rate']
    assert t.update_doc == {
        'boxes': [],
        'status': RegionStatus.NO_REGION_BOX,
        'extra': {F.gate_skip: 'tier3_hit_rate'},
    }
    h.segmenter.segment_multi.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_segmenter_skips_the_gate_and_the_observation(h: Harness) -> None:
    h.segmenter.enabled = False
    t = _task()
    await h.run(t)
    h.gate.decide.assert_not_awaited()
    h.gate.observe.assert_not_called()
    # A disabled segmenter leaves no ':miss' trace entry either.
    assert t.detection_trace == []
    assert h.out() == [t]
    assert t.update_doc['status'] == RegionStatus.NO_REGION_BOX


@pytest.mark.asyncio
async def test_segmenter_outage_leaves_the_task_pending_and_backs_off(h: Harness) -> None:
    h.segmenter.segment_multi = AsyncMock(side_effect=SegmenterUnavailable('all hosts down'))
    t = _task()
    await h.run(t)
    assert h.out() == []
    assert h.combined() == []
    assert h.ctx.in_flight == set()
    assert t.update_doc == {}
    assert h.sleeps == [1.0]
    h.sam_metric.labels.assert_called_once_with(outcome='error')
    assert [o for o, _ in h.request_metrics] == ['error']


@pytest.mark.asyncio
async def test_unexpected_segmenter_error_releases_the_task_and_keeps_going(h: Harness) -> None:
    h.segmenter.segment_multi = AsyncMock(side_effect=[RuntimeError('boom'), []])
    bad, ok = _task('bad'), _task('ok')
    await h.run(bad, ok)
    assert h.ctx.in_flight == {'ok'}
    assert h.out() == [ok]
    assert [o for o, _ in h.request_metrics] == ['error', 'miss']
    assert h.sleeps == []


@pytest.mark.asyncio
async def test_confident_single_hit_skips_the_vlm_and_writes_an_accepted_box(h: Harness) -> None:
    h.selected = [_cand(0.99)]
    h.segmenter.segment_multi = AsyncMock(return_value=['raw'])
    t = _task()
    await h.run(t)

    assert h.out() == [t]
    assert h.combined() == []
    assert t.detection_trace == [f'{SEG}:hit', f'{SEG}:skip_vlm_verify']
    doc = t.update_doc
    assert doc['status'] == RegionStatus.DETECTED
    (box,) = doc['boxes']
    assert box.bbox_norm == (0.3, 0.3, 0.5, 0.4)
    assert box.state == 'accepted'
    assert box.score == 0.99
    assert box.detector == SEG
    assert doc['extra'][F.skip_verify] is True
    assert doc['extra'][F.verified] is False
    assert doc['extra'][F.verifier] is None
    text_call = dict(h.calls)['apply_text']
    assert text_call['region_in_crop'] == (0.2, 0.2, 0.6, 0.5)
    assert text_call['vlm_available'] is True
    assert text_call['crop_jpeg'] == b'jpeg'
    h.gate.observe.assert_called_once()
    assert h.gate.observe.call_args.kwargs['hit'] is True
    h.sam_metric.labels.assert_called_once_with(outcome='hit')
    assert [o for o, _ in h.request_metrics] == ['hit']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('cands', 'plausible'),
    [
        ([_cand(0.5)], True),  # below the skip score
        ([_cand(0.99)], False),  # not region-shaped
        ([_cand(0.99), _cand(0.99)], True),  # more than one: the VLM adjudicates
    ],
)
async def test_other_hits_go_to_the_combined_vlm_queue(
    h: Harness, cands: list[Any], plausible: bool
) -> None:
    h.selected = cands
    h.plausible = plausible
    h.segmenter.segment_multi = AsyncMock(return_value=['raw'])
    t = _task()
    await h.run(t)
    assert h.combined() == [t]
    assert h.out() == []
    assert t.candidates == cands
    assert t.detection_trace == [f'{SEG}:hit']
    assert ('sync', t) in h.calls
    # The task stays in flight: Stage B owns it now.
    assert h.ctx.in_flight == {'c1'}


@pytest.mark.asyncio
async def test_hit_without_a_vlm_is_accepted_directly(h: Harness) -> None:
    h.rt.vlm_available = False
    h.selected = [_cand(0.5)]
    h.segmenter.segment_multi = AsyncMock(return_value=['raw'])
    t = _task()
    await h.run(t)
    assert h.out() == [t]
    assert h.combined() == []
    assert t.update_doc == {'accepted_without_vlm': True}


@pytest.mark.asyncio
async def test_selection_uses_the_profile_and_item_box(h: Harness) -> None:
    h.segmenter.segment_multi = AsyncMock(return_value=['raw'])
    await h.run(_task())
    raw, kw = dict(h.calls)['select']
    assert raw == ['raw']
    assert kw['item_bbox_norm'] == (0.1, 0.1, 0.9, 0.9)
    assert kw['detector'] == SEG
    assert kw['detector_version'] == 'v1'
    assert kw['source'] == 'segmenter'


@pytest.mark.asyncio
async def test_miss_without_text_hint_is_final(h: Harness) -> None:
    t = _task()
    await h.run(t)
    assert h.out() == [t]
    assert t.detection_trace == [f'{SEG}:miss']
    assert t.update_doc == {'boxes': [], 'status': RegionStatus.NO_REGION_BOX, 'extra': None}
    h.gate.observe.assert_called_once()
    assert h.gate.observe.call_args.kwargs['hit'] is False
    h.sam_metric.labels.assert_called_once_with(outcome='miss')


def _hint_pick() -> Any:
    return SimpleNamespace(bbox_norm=(0.4, 0.4, 0.6, 0.5), text='AB12', rec_score=0.8)


@pytest.mark.asyncio
async def test_text_hint_without_ocr_regions_is_a_miss(h: Harness) -> None:
    h.rt.text_hint_on = True
    t = _task()
    await h.run(t)
    h.ocr.detect_regions.assert_awaited_once_with(b'jpeg')
    assert t.detection_trace == ['rec-x:text_hint:miss']
    assert t.update_doc['status'] == RegionStatus.NO_REGION_BOX


@pytest.mark.asyncio
async def test_text_hint_prefers_stored_ocr_lines(h: Harness) -> None:
    h.rt.text_hint_on = True
    t = _task()
    t.item_ocr_lines = [{'line': 1}]
    await h.run(t)
    h.ocr.regions_from_lines.assert_called_once_with([{'line': 1}])
    h.ocr.detect_regions.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_hint_ocr_failure_counts_as_no_regions(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.ocr.detect_regions = AsyncMock(side_effect=RuntimeError('ocr down'))
    t = _task()
    await h.run(t)
    assert t.detection_trace == ['rec-x:text_hint:miss']
    assert h.out() == [t]


@pytest.mark.asyncio
async def test_text_hint_regions_without_a_usable_shape(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.ocr.detect_regions = AsyncMock(return_value=['region'])
    t = _task()
    await h.run(t)
    assert t.detection_trace == ['rec-x:text_hint:no_region_shape']
    assert t.update_doc['status'] == RegionStatus.NO_REGION_BOX


@pytest.mark.asyncio
async def test_text_hint_resegment_miss_falls_through_to_no_region_box(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.ocr.detect_regions = AsyncMock(return_value=['region'])
    h.ocr.pick_best_text_region = MagicMock(return_value=_hint_pick())
    h.resegment = (None, None)
    t = _task()
    await h.run(t)
    assert t.detection_trace == ['rec-x:text_hint:hit', f'{SEG}:text_hint:miss']
    assert h.out() == [t]
    assert t.update_doc['status'] == RegionStatus.NO_REGION_BOX


@pytest.mark.asyncio
async def test_text_hint_resegment_hit_builds_a_hinted_candidate(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.ocr.detect_regions = AsyncMock(return_value=['region'])
    h.ocr.pick_best_text_region = MagicMock(return_value=_hint_pick())
    h.resegment = (SimpleNamespace(bbox_norm=(0.2, 0.2, 0.8, 0.6), score=0.7), None)
    t = _task()
    await h.run(t)

    assert h.combined() == [t]
    assert h.out() == []
    assert t.detection_trace == ['rec-x:text_hint:hit']
    (cand,) = t.candidates
    assert cand.bbox_in_crop == (0.2, 0.2, 0.8, 0.6)
    assert cand.source == 'segmenter_text_hint'
    assert cand.detector == SEG
    assert cand.hint_text == 'AB12'
    assert cand.hint_text_confidence == 0.8
    # The crop box mapped into source coordinates through the item box.
    assert cand.bbox_in_source == pytest.approx((0.26, 0.26, 0.74, 0.58))
    assert t.candidate_text == 'AB12'
    assert dict(h.calls)['resegment'] == (b'jpeg', (0.4, 0.4, 0.6, 0.5))


@pytest.mark.asyncio
async def test_text_hint_resegment_hit_without_a_vlm_is_accepted_directly(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.rt.vlm_available = False
    h.ocr.detect_regions = AsyncMock(return_value=['region'])
    h.ocr.pick_best_text_region = MagicMock(return_value=_hint_pick())
    h.resegment = (SimpleNamespace(bbox_norm=(0.2, 0.2, 0.8, 0.6), score=0.7), None)
    t = _task()
    await h.run(t)
    assert h.out() == [t]
    assert h.combined() == []
    assert t.update_doc == {'accepted_without_vlm': True}


@pytest.mark.asyncio
async def test_text_hint_resegment_outage_leaves_the_task_pending(h: Harness) -> None:
    h.rt.text_hint_on = True
    h.ocr.detect_regions = AsyncMock(return_value=['region'])
    h.ocr.pick_best_text_region = MagicMock(return_value=_hint_pick())
    h.resegment = SegmenterUnavailable('down')
    t = _task()
    await h.run(t)
    assert h.out() == []
    assert h.combined() == []
    assert h.ctx.in_flight == set()
    assert h.sleeps == [1.0]
    assert t.detection_trace == ['rec-x:text_hint:hit']


@pytest.mark.asyncio
async def test_request_id_is_unbound_after_every_task(h: Harness) -> None:
    await h.run(_task('a'), _task('b', jpeg=None))
    assert 'request_id' not in structlog.contextvars.get_contextvars()
    assert len(h.out()) == 2
