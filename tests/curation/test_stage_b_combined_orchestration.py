"""``stage_b_combined`` wiring, pinned with fakes for the VLM, the reply resolver and the
project binding: how a chunk is built, which tasks reach the writer queue, what is
released from ``in_flight``, which counters move, and when the no-verdict cap forces a
resolution. The reply arithmetic itself is covered by the combined-resolve tests."""

from __future__ import annotations

import asyncio
from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from scripts.curation.worker import stage_b
from scripts.curation.worker.combined_resolve import CombinedResolution
from scripts.curation.worker.no_verdict import NoVerdictCounter
from scripts.curation.worker.pipeline import PipelineContext
from scripts.curation.worker.state import RegionProfileNotConfiguredError, _ItemTask
from src.config.region_state import RegionStatus
from src.services.labeling.vlm_client import VlmIdentity


def _candidate(detector: str = 'det') -> Any:
    return SimpleNamespace(stored=None, detector=detector, bbox_in_crop=(0.1, 0.2, 0.3, 0.4))


def _task(crop_id: str, *, project: str = 'alpha', jpeg: bytes | None = b'jpeg') -> _ItemTask:
    return _ItemTask(
        crop_id=crop_id,
        image_path=f'/{crop_id}.jpg',
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status='pending_detection',
        class_name='',
        request_id=f'req-{crop_id}',
        crop_jpeg=jpeg,
        candidates=[_candidate()],
        project=SimpleNamespace(slug=project),  # type: ignore[arg-type]
    )


def _resolution(outcome: str = 'resolved', **kw: Any) -> CombinedResolution:
    defaults: dict[str, Any] = {'boxes': ['box'], 'status': RegionStatus.DETECTED}
    return CombinedResolution(outcome=outcome, **(defaults | kw))


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, cap: int = 3) -> None:
        self.vlm = MagicMock()
        self.vlm.label_combined_batch = AsyncMock(side_effect=self._replies)
        self.rt = SimpleNamespace(
            vlm=self.vlm,
            vlm_identity=VlmIdentity('env@None', 'model-x'),
            pack_ref=('pack-x', 2),
            vlm_model='model-x',
            profile='profile-x',
            text_rules='rules-x',
            ocr_recognizer='ocr-x',
        )
        self.rt_error: Exception | None = None
        self.label_calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.bound: list[str] = []
        self.resolve_calls: list[dict[str, Any]] = []
        self.resolutions: dict[str, list[CombinedResolution]] = {}
        self.replies_for: dict[str, Any] = {}
        self.metric = MagicMock()
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
            no_verdict_cap=cap,
            visible_no_verdict=NoVerdictCounter(cap),
            combined_no_verdict=NoVerdictCounter(cap),
            crop_gate=MagicMock(),
        )
        self.ctx.rt_for = self._rt_for  # type: ignore[method-assign]
        monkeypatch.setattr(stage_b, 'bind_task_project', lambda t: self.bound.append(t.crop_id))
        monkeypatch.setattr(stage_b, 'bound_class_catalog', lambda: (['car', 'bus'], {'car': 7}))
        monkeypatch.setattr(stage_b, 'resolve_combined_reply', self._resolve)
        monkeypatch.setattr(
            stage_b,
            '_box_list_doc',
            lambda t, boxes, status, *, extra=None: {
                'doc_for': t.crop_id,
                'boxes': boxes,
                'status': status,
                'extra': extra,
            },
        )
        monkeypatch.setattr(stage_b, 'OP_STAGE_B_VLM_VERIFY_DURATION_SECONDS', self.metric)

    def _rt_for(self, _t: _ItemTask) -> Any:
        if self.rt_error is not None:
            raise self.rt_error
        return self.rt

    async def _replies(self, crops: Any, **kw: Any) -> dict[str, Any]:
        self.label_calls.append((list(crops), kw))
        return {
            c.crop_id: self.replies_for.get(c.crop_id)
            for c in crops
            if c.crop_id in self.replies_for
        }

    async def _resolve(self, candidates: Any, reply: Any, **kw: Any) -> CombinedResolution:
        self.resolve_calls.append(
            {'candidates': candidates, 'reply': reply, 'crop_id': kw['crop_id'], **kw}
        )
        return self.resolutions[kw['crop_id']].pop(0)

    async def run(self, tasks: list[_ItemTask]) -> list[_ItemTask]:
        for t in tasks:
            self.ctx.in_flight.add(t.crop_id)
            await self.ctx.combined_q.put(t)
        await self.ctx.combined_q.put(None)
        await asyncio.wait_for(stage_b.stage_b_combined(self.ctx, 4), 5)
        written = []
        while not self.ctx.out_q.empty():
            written.append(self.ctx.out_q.get_nowait())
        return written


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(monkeypatch)


@pytest.mark.asyncio
async def test_resolved_chunk_is_written_with_trace_metrics_and_vlm_stamp(h: Harness) -> None:
    a, b = _task('a'), _task('b')
    h.replies_for = {'a': 'reply-a', 'b': 'reply-b'}
    h.resolutions = {
        'a': [_resolution(trace=['det:verified'], no_region_visible=True)],
        'b': [_resolution(bbox_wrong=True, extra={'k': 1}, trace=['det:rejected'])],
    }
    written = await h.run([a, b])

    assert written == [a, b]
    assert a.update_doc == {
        'doc_for': 'a',
        'boxes': ['box'],
        'status': RegionStatus.DETECTED,
        'extra': {},
    }
    assert b.update_doc is not None
    assert b.update_doc['extra'] == {'k': 1}
    # The ``:hit`` entry first, then what the resolution added.
    assert a.detection_trace == ['det:hit', 'det:verified']
    assert b.detection_trace == ['det:hit', 'det:rejected']
    assert dict(h.ctx.metrics) == {'combined_no_region_visible': 1, 'combined_bbox_wrong': 1}
    assert h.bound == ['a']
    # One VLM call carries the whole chunk, with the project's class catalog.
    ((crops, kw),) = h.label_calls
    assert [c.crop_id for c in crops] == ['a', 'b']
    assert [c.region_bboxes_norm for c in crops] == [[(0.1, 0.2, 0.3, 0.4)]] * 2
    assert kw == {'class_names': ['car', 'bus']}
    assert a.vlm_called is True
    assert a.vlm_identity == VlmIdentity('env@None', 'model-x')
    # in_flight is released by the writer, not here.
    assert h.ctx.in_flight == {'a', 'b'}
    h.metric.labels.assert_called_once_with(outcome='ok')
    # Resolver inputs come from the task and the runtime.
    first = h.resolve_calls[0]
    assert first['reply'] == 'reply-a'
    assert first['vlm_model'] == 'model-x'
    assert first['profile'] == 'profile-x'
    assert first['rules'] == 'rules-x'
    assert first['ocr'] == 'ocr-x'
    assert first['name_to_id'] == {'car': 7}
    assert first['crop_jpeg'] == b'jpeg'
    assert 'force_resolve' not in first
    assert 'request_id' not in structlog.contextvars.get_contextvars()


@pytest.mark.asyncio
async def test_unreadable_crop_is_written_as_failed_and_left_out_of_the_vlm_call(
    h: Harness,
) -> None:
    good, bad = _task('good'), _task('bad', jpeg=None)
    h.replies_for = {'good': 'r'}
    h.resolutions = {'good': [_resolution()]}
    written = await h.run([bad, good])

    assert written == [bad, good]
    assert bad.update_doc is not None
    assert RegionStatus.DETECTION_FAILED in bad.update_doc.values()
    assert 'image_unavailable' in bad.update_doc.values()
    ((crops, _kw),) = h.label_calls
    assert [c.crop_id for c in crops] == ['good']
    assert [c['crop_id'] for c in h.resolve_calls] == ['good']
    assert bad.vlm_called is False


@pytest.mark.asyncio
async def test_all_unreadable_chunk_makes_no_vlm_call(h: Harness) -> None:
    bad = _task('bad', jpeg=None)
    written = await h.run([bad])
    assert written == [bad]
    assert h.label_calls == []
    h.metric.labels.assert_not_called()


@pytest.mark.asyncio
async def test_transport_failure_releases_the_whole_chunk_and_writes_nothing(h: Harness) -> None:
    a, b = _task('a'), _task('b')
    h.vlm.label_combined_batch = AsyncMock(side_effect=RuntimeError('upstream down'))
    written = await h.run([a, b])

    assert written == []
    assert h.ctx.in_flight == set()
    assert a.update_doc == {}
    assert h.resolve_calls == []
    assert a.vlm_called is False
    h.metric.labels.assert_called_once_with(outcome='error')
    assert dict(h.ctx.metrics) == {}


@pytest.mark.asyncio
async def test_vlm_missing_from_the_runtime_is_a_transport_style_failure(h: Harness) -> None:
    a = _task('a')
    h.rt.vlm = None
    written = await h.run([a])
    assert written == []
    assert h.ctx.in_flight == set()
    h.metric.labels.assert_called_once_with(outcome='error')


@pytest.mark.asyncio
async def test_project_without_a_runtime_is_released_and_skipped(h: Harness) -> None:
    a = _task('a')
    h.rt_error = RegionProfileNotConfiguredError('no runtime')
    written = await h.run([a])
    assert written == []
    assert h.ctx.in_flight == set()
    assert h.label_calls == []


@pytest.mark.asyncio
async def test_no_verdict_without_a_reply_is_a_parse_failure_and_is_retried(h: Harness) -> None:
    a = _task('a')
    h.resolutions = {'a': [_resolution('no_verdict', boxes=[], status=None)]}
    written = await h.run([a])

    assert written == []
    assert a.update_doc == {}
    assert h.ctx.in_flight == set()
    assert dict(h.ctx.metrics) == {'combined_parse_failure': 1}
    assert h.ctx.combined_no_verdict.count('a') == 1


@pytest.mark.asyncio
async def test_no_verdict_with_a_reply_counts_the_missing_bbox_verdict(h: Harness) -> None:
    a = _task('a')
    h.replies_for = {'a': 'reply-a'}
    h.resolutions = {'a': [_resolution('no_verdict', boxes=[], status=None)]}
    written = await h.run([a])
    assert written == []
    assert dict(h.ctx.metrics) == {'combined_no_bbox_verdict': 1}
    assert h.ctx.in_flight == set()


@pytest.mark.asyncio
async def test_no_verdict_at_the_cap_is_resolved_with_force_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(monkeypatch, cap=2)
    a = _task('a')
    h.replies_for = {'a': 'reply-a'}
    h.ctx.combined_no_verdict.record('a')  # the first miss already happened
    h.resolutions = {
        'a': [
            _resolution('no_verdict', boxes=[], status=None),
            _resolution(status=RegionStatus.VERIFY_REJECTED, trace=['det:no_verdict']),
        ]
    }
    written = await h.run([a])

    assert written == [a]
    assert [c.get('force_resolve') for c in h.resolve_calls] == [None, True]
    assert a.update_doc is not None
    assert a.update_doc['status'] == RegionStatus.VERIFY_REJECTED
    assert dict(h.ctx.metrics) == {
        'combined_no_bbox_verdict': 1,
        'combined_no_verdict_cap_hits': 1,
    }
    assert a.detection_trace == ['det:hit', 'det:no_verdict']
    assert h.ctx.combined_no_verdict.count('a') == 0


@pytest.mark.asyncio
async def test_a_real_verdict_clears_the_no_verdict_count(h: Harness) -> None:
    a = _task('a')
    h.ctx.combined_no_verdict.record('a')
    h.replies_for = {'a': 'reply-a'}
    h.resolutions = {'a': [_resolution()]}
    await h.run([a])
    assert h.ctx.combined_no_verdict.count('a') == 0


@pytest.mark.asyncio
async def test_chunks_never_mix_projects(h: Harness) -> None:
    a1, a2, b1 = _task('a1'), _task('a2'), _task('b1', project='beta')
    h.replies_for = {'a1': 'r', 'a2': 'r', 'b1': 'r'}
    h.resolutions = {k: [_resolution()] for k in ('a1', 'a2', 'b1')}
    written = await h.run([a1, a2, b1])

    assert written == [a1, a2, b1]
    assert [[c.crop_id for c in crops] for crops, _ in h.label_calls] == [['a1', 'a2'], ['b1']]
    assert h.bound == ['a1', 'b1']


@pytest.mark.asyncio
async def test_a_resolver_error_propagates_and_unbinds_the_request_id(h: Harness) -> None:
    a = _task('a')
    h.replies_for = {'a': 'r'}
    h.resolutions = {}  # KeyError from the fake resolver
    with pytest.raises(KeyError):
        await h.run([a])
    assert 'request_id' not in structlog.contextvars.get_contextvars()
    assert a.update_doc == {}


@pytest.mark.asyncio
async def test_classify_flag_follows_the_task_class_state(h: Harness) -> None:
    validated = _task('v')
    validated.class_validated = True
    free = _task('f')
    h.replies_for = {'v': 'r', 'f': 'r'}
    h.resolutions = {'v': [_resolution()], 'f': [_resolution()]}
    await h.run([validated, free])

    ((crops, _kw),) = h.label_calls
    assert {c.crop_id: c.classify for c in crops} == {'v': False, 'f': True}
    by_id = {c['crop_id']: c for c in h.resolve_calls}
    assert by_id['v']['effective_class_names'] is None
    assert by_id['f']['effective_class_names'] == ['car', 'bus']
