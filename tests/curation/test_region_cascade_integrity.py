"""Region-cascade integrity regressions found on a live deployment.

1. ``region_text`` must be the text transcribed off the region — never a
   description of the region or the item's class — and every VLM reply
   field that drives an accept is read strictly (fail-closed), with batch
   replies aligned to the right crop.
2. ``region_detector_chain`` entries are exactly ``<actor>:<event>`` and
   the training-candidate readers' ``term`` filters match what the worker
   writes.
3. One pending transition = one cascade pass: a refresh-lagged pending
   search must not hand the same item to the pipeline twice.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import scripts.curation.region_worker_main as worker
from scripts.curation.worker import runner as runner_mod
from src.config import get_region_fields
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.profile_registry import get_active_region_profile
from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    CombinedParseFailure,
    RegionCrop,
    VlmCombinedReply,
    VlmLabeler,
    _align_batch_entries,
)
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


if TYPE_CHECKING:
    from pathlib import Path

    from src.config import DetectionProfile


pytestmark = pytest.mark.usefixtures('reference_region_profile')

CLASS_NAMES = ['sedan', 'chevycar', 'toyotacar', 'pickup']


def _profile() -> DetectionProfile:
    profile = get_active_region_profile()
    assert profile is not None
    return profile


def _chat(content: str) -> dict[str, Any]:
    return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}


def _labeler(reply: dict[str, Any] | str) -> VlmLabeler:
    lab = VlmLabeler(base_url='http://vlm.invalid/v1')
    content = reply if isinstance(reply, str) else json.dumps(reply)
    lab._post_chat = AsyncMock(return_value=_chat(content))  # type: ignore[method-assign]
    return lab


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (64, 48), (90, 90, 90)).save(buf, format='JPEG')
    return buf.getvalue()


def _combined(**over: Any) -> dict[str, Any]:
    """A realistic combined reply, as the deployment pack asks for it."""
    reply = {
        'class_id': 1,
        'class_confidence': 'high',
        'region_visible': True,
        'region_bbox_correct': True,
        'region_text': 'DNV20',
        'region_confidence': 'high',
        'make': 'Chevrolet',
        'model': 'Silverado',
    }
    reply.update(over)
    return reply


# =============================================================================
# 1. region_text + accept decision
# =============================================================================


class TestCombinedSingle:
    @pytest.mark.asyncio
    async def test_region_text_is_the_transcribed_text(self) -> None:
        reply = await _labeler(_combined()).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0.2, 0.6, 0.5, 0.8)
        )
        assert reply.region_text_reply == 'DNV20'
        assert reply.region_visible is True
        assert reply.region_bbox_correct is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'echo',
        ['chevycar', 'CHEVYCAR', 'Chevy Car', 'Chevrolet', 'Silverado', 'Chevrolet Silverado'],
    )
    async def test_class_or_attribute_echo_is_not_region_text(self, echo: str) -> None:
        reply = await _labeler(_combined(region_text=echo)).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0.2, 0.6, 0.5, 0.8)
        )
        assert reply.region_text_reply is None
        # The class side still resolves — only the text slot is dropped.
        assert reply.class_id == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('sentinel', [None, '', 'unknown', 'N/A', 'null'])
    async def test_no_text_is_null(self, sentinel: str | None) -> None:
        reply = await _labeler(_combined(region_text=sentinel)).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0.2, 0.6, 0.5, 0.8)
        )
        assert reply.region_text_reply is None

    @pytest.mark.asyncio
    async def test_quoted_false_booleans_do_not_accept(self) -> None:
        reply = await _labeler(
            _combined(region_visible='false', region_bbox_correct='false')
        ).label_combined('c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0, 0, 1, 1))
        assert reply.region_visible is False
        assert reply.region_bbox_correct is False

    @pytest.mark.asyncio
    async def test_unrecognized_bbox_answer_is_not_an_accept(self) -> None:
        reply = await _labeler(_combined(region_bbox_correct='maybe')).label_combined(
            'c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0, 0, 1, 1)
        )
        assert reply.region_bbox_correct is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'content',
        [
            '',
            json.dumps({k: v for k, v in _combined().items() if k != 'region_visible'}),
            json.dumps(_combined(region_visible='perhaps')),
        ],
    )
    async def test_missing_or_garbled_visible_answer_is_a_parse_failure(self, content: str) -> None:
        with pytest.raises(CombinedParseFailure):
            await _labeler(content).label_combined(
                'c1', _jpeg(), class_names=CLASS_NAMES, region_bbox_norm=(0, 0, 1, 1)
            )


class TestCombinedBatch:
    def _crops(self, n: int) -> list[CombinedCrop]:
        return [
            CombinedCrop(
                crop_id=f'c{i}', jpeg_bytes=b'', region_bbox_norm=(0, 0, 1, 1), classify=True
            )
            for i in range(1, n + 1)
        ]

    def _parse(self, entries: list[dict[str, Any]], n: int) -> dict[str, Any]:
        raw = json.dumps({'results': entries})
        return VlmLabeler._parse_combined_batch_response(
            raw, self._crops(n), get_region_fields(), class_names=CLASS_NAMES
        )

    def test_each_crop_gets_its_own_text(self) -> None:
        out = self._parse(
            [
                {'img': 2, **_combined(region_text='XYZ789')},
                {'img': 1, **_combined(region_text='DNV20')},
                {'img': 3, **_combined(region_text='chevycar')},
            ],
            3,
        )
        assert out['c1'].region_text_reply == 'DNV20'
        assert out['c2'].region_text_reply == 'XYZ789'
        assert out['c3'].region_text_reply is None

    def test_zero_based_indices_are_shifted_not_misassigned(self) -> None:
        out = self._parse(
            [
                {'img': 0, **_combined(region_text='AAA111', region_bbox_correct=True)},
                {'img': 1, **_combined(region_text='BBB222', region_bbox_correct=False)},
            ],
            2,
        )
        assert out['c1'].region_text_reply == 'AAA111'
        assert out['c1'].region_bbox_correct is True
        assert out['c2'].region_bbox_correct is False

    @pytest.mark.parametrize(
        'entries',
        [
            # duplicate index
            [{'img': 1, **_combined()}, {'img': 1, **_combined()}],
            # out of range
            [{'img': 1, **_combined()}, {'img': 7, **_combined()}],
            # indexed + unindexed mix
            [{'img': 1, **_combined()}, _combined()],
            # unindexed, wrong count
            [_combined()],
        ],
    )
    def test_untrustworthy_alignment_fails_the_whole_chunk_closed(
        self, entries: list[dict[str, Any]]
    ) -> None:
        out = self._parse(entries, 2)
        assert out == {'c1': None, 'c2': None}

    def test_missing_entry_is_none_not_a_neighbours_verdict(self) -> None:
        out = self._parse([{'img': 2, **_combined()}], 2)
        assert out['c1'] is None
        assert out['c2'] is not None

    def test_entry_without_visible_answer_is_none(self) -> None:
        bad = {k: v for k, v in _combined().items() if k != 'region_visible'}
        out = self._parse([{'img': 1, **bad}, {'img': 2, **_combined()}], 2)
        assert out['c1'] is None
        assert out['c2'].region_visible is True


class TestStandaloneVerify:
    def test_single_text_comes_from_text_not_reason(self) -> None:
        raw = json.dumps(
            {
                'is_region': True,
                'confidence': 'high',
                'reason': 'a red taillight next to it',
                'text': 'DNV20',
                'text_confidence': 'high',
            }
        )
        v = VlmLabeler._parse_region_response(raw, RegionCrop(crop_id='c1', jpeg_bytes=b''))
        assert v is not None
        assert v.is_region is True
        assert v.text == 'DNV20'

    def test_single_quoted_false_is_a_reject(self) -> None:
        raw = json.dumps({'is_region': 'false', 'confidence': 'high', 'text': 'X'})
        v = VlmLabeler._parse_region_response(raw, RegionCrop(crop_id='c1', jpeg_bytes=b''))
        assert v is not None
        assert v.is_region is False
        assert v.text is None

    def test_batch_misaligned_yields_no_verdicts_not_rejects(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps(
            [
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'A1'},
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'B2'},
            ]
        )
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        assert out == []

    def test_batch_missing_entry_is_omitted_not_a_reject(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps([{'img': 2, 'is_region': True, 'confidence': 'high', 'text': 'B2'}])
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        by_id = {v.crop_id: v for v in out}
        assert 'c1' not in by_id
        assert by_id['c2'].is_region is True

    def test_batch_empty_reply_yields_no_verdicts(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        assert VlmLabeler._parse_region_batch_response('', crops) == []

    def test_batch_aligned_by_index(self) -> None:
        crops = [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'') for i in (1, 2)]
        raw = json.dumps(
            [
                {'img': 2, 'is_region': False, 'confidence': 'high', 'text': None},
                {'img': 1, 'is_region': True, 'confidence': 'high', 'text': 'A1'},
            ]
        )
        out = VlmLabeler._parse_region_batch_response(raw, crops)
        assert (out[0].crop_id, out[0].is_region, out[0].text) == ('c1', True, 'A1')
        assert (out[1].crop_id, out[1].is_region) == ('c2', False)


class TestAlignBatchEntries:
    def test_positional_only_when_counts_match(self) -> None:
        a, b = {'x': 1}, {'x': 2}
        assert _align_batch_entries([a, b], 2) == [a, b]
        assert _align_batch_entries([a], 2) is None

    def test_bool_index_rejected(self) -> None:
        assert _align_batch_entries([{'img': True}], 1) is None


# =============================================================================
# Runner wiring: the worker must use the deployment's prompt pack
# =============================================================================


def _capture_signal_handler(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    handlers: list[Any] = []

    def _record(_self: Any, _sig: Any, cb: Any, *_a: Any) -> None:
        handlers.append(cb)

    monkeypatch.setattr(
        asyncio.get_event_loop().__class__, 'add_signal_handler', _record, raising=False
    )
    return handlers


class _FakeOpenSearch:
    """Items index with realtime ``mget`` and refresh-gated ``search``.

    ``search`` sees the last refreshed snapshot, and responds
    ``search_delay`` seconds after it took it (a search in flight while a
    write lands returns pre-write state). A bulk with
    ``refresh='wait_for'`` / ``True`` is searchable once it returns; any
    other bulk only after ``lag_searches`` further searches.
    """

    def __init__(
        self, docs: dict[str, dict[str, Any]], *, search_delay: float, lag_searches: int
    ) -> None:
        self.live = copy.deepcopy(docs)
        self.searchable = copy.deepcopy(docs)
        self.search_delay = search_delay
        self.lag_searches = lag_searches
        self._pending_refresh: int | None = None
        self.seq = dict.fromkeys(docs, 1)
        self.writes: list[tuple[str, dict[str, Any]]] = []
        self.pending_statuses = {
            'pending',
            'pending_detection',
            'pending_verify',
            'pending_verification',
        }

    async def search(self, **_kw: Any) -> dict[str, Any]:
        if self._pending_refresh is not None:
            self._pending_refresh -= 1
            if self._pending_refresh <= 0:
                self.searchable = copy.deepcopy(self.live)
                self._pending_refresh = None
        status_key = get_region_fields().status
        hits = [
            {'_id': cid, '_source': copy.deepcopy(src)}
            for cid, src in self.searchable.items()
            if src.get(status_key) in self.pending_statuses
        ]
        await asyncio.sleep(self.search_delay)
        return {'hits': {'hits': hits}}

    async def mget(self, *, body: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        docs = []
        for d in body['docs']:
            cid = d['_id']
            docs.append(
                {
                    '_id': cid,
                    'found': cid in self.live,
                    '_seq_no': self.seq.get(cid, 1),
                    '_primary_term': 1,
                    '_source': copy.deepcopy(self.live.get(cid, {})),
                }
            )
        return {'docs': docs}

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            cid = action['update']['_id']
            if action['update'].get('if_seq_no') != self.seq[cid]:
                items.append({'update': {'_id': cid, 'status': 409}})
                continue
            self.live[cid].update(doc['doc'])
            self.seq[cid] += 1
            self.writes.append((cid, copy.deepcopy(doc['doc'])))
            items.append({'update': {'_id': cid, 'status': 200, 'result': 'updated'}})
        if refresh in ('wait_for', True):
            self.searchable = copy.deepcopy(self.live)
        else:
            self._pending_refresh = self.lag_searches
        return {'errors': False, 'items': items}

    async def close(self) -> None:
        return None


def _item(status: str = 'pending_detection') -> dict[str, Any]:
    F = get_region_fields()
    return {
        'crop_id': 'c1',
        'image_path': '/nonexistent/source.jpg',
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        F.status: status,
        'class_name': 'sedan',
        'group': 'cars',
        'class_source': 'classifier_model',
        'confidence': 0.95,
        'created_at': '2026-09-24T00:00:00+00:00',
    }


async def _drive_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_os: _FakeOpenSearch,
    primary: RegionCandidate | None,
    segmenter: RegionCandidate | None,
    reply: VlmCombinedReply,
    visible: bool | None = True,
    combined_side_effect: Any = None,
    visible_side_effect: Any = None,
    until_writes: int = 1,
    on_write: Any = None,
) -> dict[str, Any]:
    """Run the streaming worker in continuous mode until the item is
    written plus several more polls, then stop it. Returns the mocks.
    ``visible=None``: the visibility pre-filter gives no verdict at all.
    ``combined_side_effect`` / ``visible_side_effect`` replace the VLM
    mocks' fixed answers (called with the crop list). The run stops once
    ``until_writes`` writes landed (or a timeout); ``on_write(n)`` is
    called as the n-th write is seen."""
    handlers = _capture_signal_handler(monkeypatch)
    monkeypatch.setenv('OP_REGION_WORKER_METRICS_PORT', '0')

    pool = MagicMock(initialize=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(worker, 'AsyncTritonPool', MagicMock(return_value=pool))
    monkeypatch.setattr(worker, 'AsyncOpenSearch', MagicMock(return_value=fake_os))

    primary_det = MagicMock()
    primary_det.detect_batch = AsyncMock(return_value=[primary])
    monkeypatch.setattr(runner_mod, 'RegionDetector', MagicMock(return_value=primary_det))
    ocr = MagicMock()
    ocr.detect_regions = AsyncMock(return_value=[])
    ocr.pick_best_text_region = MagicMock(return_value=None)
    monkeypatch.setattr(runner_mod, 'PaddleOcrTextRecognizer', MagicMock(return_value=ocr))
    monkeypatch.setattr(runner_mod, '_crop_jpeg_for_task', lambda *_a: _jpeg())

    seg = MagicMock(aclose=AsyncMock())
    seg.segment = AsyncMock(return_value=segmenter)
    monkeypatch.setattr(worker, 'SegmenterClient', MagicMock(return_value=seg))

    vlm = MagicMock(aclose=AsyncMock())
    vlm.class_names = []
    vlm.label_combined_batch = AsyncMock(
        side_effect=combined_side_effect or (lambda crops, **_kw: {c.crop_id: reply for c in crops})
    )
    vlm.region_visible_batch = AsyncMock(
        side_effect=visible_side_effect
        or (lambda crops, **_kw: {} if visible is None else {c.crop_id: visible for c in crops})
    )
    vlm_cls = MagicMock(return_value=vlm)
    monkeypatch.setattr(worker, 'VlmLabeler', vlm_cls)
    # Registry load is best-effort; keep it out of the test.
    monkeypatch.setattr(
        'src.clients.curation_opensearch.ClassRegistry',
        MagicMock(side_effect=RuntimeError('no registry in test')),
    )
    # Stage A resolves the class group through the process-wide registry
    # singleton; without this the test depends on another test having
    # loaded it first.
    monkeypatch.setattr('scripts.curation.worker.state._class_group', lambda _name: None)

    args = worker.parse_args(
        [
            '--opensearch=http://os.invalid:9200',
            '--triton=triton.invalid:8001',
            '--segmenter-url=http://seg.invalid:8000',
            '--vlm-url=http://vlm.invalid:8000',
            f'--pause-sentinel={tmp_path / "absent.sentinel"}',
            '--continuous',
            '--poll-interval=0.01',
            '--batch-size=4',
            '--concurrency=2',
        ]
    )

    async def _stopper() -> None:
        seen = 0
        for _ in range(500):
            await asyncio.sleep(0.01)
            while seen < len(fake_os.writes):
                seen += 1
                if on_write is not None:
                    on_write(seen)
            if seen >= until_writes:
                break
        # Keep polling well past the write so a lagged re-fetch would show.
        await asyncio.sleep(0.6)
        handlers[0]()

    stopper = asyncio.create_task(_stopper())
    rc = await asyncio.wait_for(runner_mod.run(args), timeout=20)
    await stopper
    assert rc == 0
    return {'primary': primary_det, 'seg': seg, 'vlm': vlm, 'vlm_cls': vlm_cls}


def _accept(text: str = 'DNV20') -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_bbox_correct=True,
        region_text_reply=text,
        region_confidence='high',
    )


class TestRunnerUsesDeploymentPromptPack:
    @pytest.mark.asyncio
    async def test_resolved_pack_is_passed_to_the_labeler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack = PromptPack(**{**GENERIC_ITEM_PACK.__dict__, 'name': 'deployment_pack'})
        monkeypatch.setattr(runner_mod, 'resolve_prompt_pack', lambda: pack)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
        )
        assert mocks['vlm_cls'].call_args.kwargs['pack'] is pack


# =============================================================================
# 2 + 3. Chain format, reader agreement, one pass per item
# =============================================================================

_CANONICAL = re.compile(r'^[^:@\s]+:[^:@\s][^@\s]*$')


class TestOnePassPerItem:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('search_delay', 'lag_searches'),
        [
            # Refresh lag: the write isn't searchable for a few polls.
            (0.0, 5),
            # A search in flight while the write lands returns pre-write state.
            (0.15, 0),
        ],
    )
    async def test_item_detected_and_verified_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        search_delay: float,
        lag_searches: int,
    ) -> None:
        fake_os = _FakeOpenSearch(
            {'c1': _item()}, search_delay=search_delay, lag_searches=lag_searches
        )
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
        )
        assert mocks['primary'].detect_batch.await_count == 1
        verified = [
            c.crop_id
            for call in mocks['vlm'].label_combined_batch.await_args_list
            for c in call.args[0]
        ]
        assert verified == ['c1']
        assert len(fake_os.writes) == 1

        F = get_region_fields()
        doc = fake_os.live['c1']
        det = _profile().detector_model
        assert doc[F.status] == 'detected'
        assert doc[F.text] == 'DNV20'
        assert doc[F.detector_chain] == [f'{det}:hit', f'{det}:combined_verify_ok']

    @pytest.mark.asyncio
    async def test_stale_write_is_dropped_by_the_writer(self) -> None:
        """Backstop: a result computed from a pending state the doc has
        already left (a duplicate consumer, or a human edit) never lands."""
        F = get_region_fields()
        fake_os = _FakeOpenSearch(
            {'c1': {**_item(), F.status: 'detected'}}, search_delay=0.0, lag_searches=0
        )
        task = worker._ItemTask(
            crop_id='c1',
            image_path='',
            vehicle_bbox_norm=(0.1, 0.1, 0.9, 0.9),
            region_status='pending_detection',
            class_name='sedan',
            group='cars',
        )
        task.update_doc = {F.status: 'detected', F.text: 'AGAIN'}
        task.detection_trace = ['det:hit']
        n_written, _ = await worker._bulk_update(fake_os, [task])  # type: ignore[arg-type]
        assert n_written == 0
        assert fake_os.writes == []


class TestChainFormatMatchesReaders:
    @pytest.mark.asyncio
    async def test_blind_spot_cohort_matches_what_the_worker_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.routers.curation.regions import _training_candidate_query

        profile = _profile()
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.8, source='seg'),
            reply=_accept(),
        )
        F = get_region_fields()
        chain = fake_os.live['c1'][F.detector_chain]
        assert chain == [
            f'{profile.detector_model}:miss',
            'vlm_visible:yes',
            f'{profile.segmenter_name}:hit',
            f'{profile.segmenter_name}:combined_verify_ok',
        ]
        assert all(_CANONICAL.match(e) for e in chain), chain

        query, _ = _training_candidate_query('detector_blind_spots', profile)
        chain_terms = [
            clause['term'][F.detector_chain]
            for clause in query['bool']['filter']
            if 'term' in clause and F.detector_chain in clause['term']
        ]
        assert chain_terms, 'blind-spot cohort no longer filters on the chain'
        for term in chain_terms:
            assert term in chain

    def test_disagreement_cohort_terms_are_canonical_entries(self) -> None:
        from src.routers.curation.regions import _training_candidate_query

        F = get_region_fields()
        profile = _profile()
        query, _ = _training_candidate_query('disagreement', profile)
        terms = [
            c['term'][F.detector_chain]
            for c in query['bool']['filter']
            if 'term' in c and F.detector_chain in c['term']
        ]
        assert terms == [f'{profile.detector_model}:hit', f'{profile.segmenter_name}:hit']
        assert all(_CANONICAL.match(t) for t in terms)
