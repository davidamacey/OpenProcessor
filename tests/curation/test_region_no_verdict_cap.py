"""A deterministic no-verdict reply must not retry forever.

The VLM runs at temperature 0: a crop it answers without a verdict (a null
box verdict, an unparseable entry, an empty visibility reply) gets the same
answer on every retry. Before the cap the region worker looped such items
through the segmenter + VLM indefinitely (5 items ~330 times each on a live
deployment). Now each item gets ``OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS``
no-verdict replies, then:

* combined stage -- a reviewable ``verify_rejected`` with reason
  ``verifier_no_verdict``, the candidate kept, ``region_bbox_correct`` null;
* visibility stage -- fail open, on to detection.

A transport failure (no reply at all) is not a no-verdict reply: it keeps
retrying and never becomes a terminal write.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from scripts.curation.worker.no_verdict import (
    DEFAULT_MAX_NO_VERDICT_ATTEMPTS,
    NoVerdictCounter,
    max_no_verdict_attempts,
)
from src.config import get_region_fields
from src.config.region_rejection import (
    REJECT_REASON_NO_VERDICT,
    REJECT_REASON_SANITY_PREFIX,
    REJECT_REASON_VERIFIER,
    rejection_reason_catalog,
)
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    CombinedParseFailure,
    CombinedTransportError,
    VlmCombinedReply,
    VlmLabeler,
)

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item, _profile


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

ENV = 'OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS'
BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det')
SEG_BOX = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='seg')


def _reply(bbox_correct: bool | None) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        plate_visible=True,
        plate_bbox_correct=bbox_correct,
        plate_text='DNV20',
        plate_confidence='high',
        make='Chevrolet',
    )


def _scripted(*answers: Any) -> Any:
    """Combined-call side effect answering ``answers`` in turn, then the
    last one forever. ``'parse'`` = an unparseable entry (``None``),
    ``'transport'`` = the call raises."""

    def answer(crops: list[CombinedCrop], **_kw: Any) -> dict[str, Any]:
        n = answer.calls  # type: ignore[attr-defined]
        answer.calls += 1  # type: ignore[attr-defined]
        a = answers[min(n, len(answers) - 1)]
        if a == 'transport':
            raise CombinedTransportError('http error: connection refused')
        return {c.crop_id: None if a == 'parse' else a for c in crops}

    answer.calls = 0  # type: ignore[attr-defined]
    return answer


async def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_os: _FakeOpenSearch,
    *,
    combined: Any = None,
    visible: Any = None,
    primary: RegionCandidate | None = BOX,
    **kw: Any,
) -> dict[str, Any]:
    return await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=primary,
        segmenter=SEG_BOX,
        reply=_reply(True),
        combined_side_effect=combined,
        visible_side_effect=visible,
        **kw,
    )


def _assert_no_verdict_reject(doc: dict[str, Any]) -> None:
    F = get_region_fields()
    assert doc[F.status] == 'verify_rejected'
    assert doc[F.rejection_reason] == REJECT_REASON_NO_VERDICT
    # No verdict was given: null, never false.
    assert F.bbox_correct in doc
    assert doc[F.bbox_correct] is None
    # The box is kept as a reviewable candidate, never as an accepted region.
    assert doc[F.bbox_norm] is None
    assert doc[F.candidate_bbox_norm] is not None
    assert doc[F.candidate_detector] == _profile().detector_model
    det = _profile().detector_model
    assert f'{det}:combined_verify_reject:{REJECT_REASON_NO_VERDICT}' in doc[F.detector_chain]


class TestCombinedNoVerdictIsCapped:
    @pytest.mark.asyncio
    async def test_null_box_verdict_is_capped_into_a_reviewable_reject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(tmp_path, monkeypatch, fake_os, combined=_scripted(_reply(None)))
        F = get_region_fields()
        assert mocks['vlm'].label_combined_batch.await_count == DEFAULT_MAX_NO_VERDICT_ATTEMPTS
        assert len(fake_os.writes) == 1
        doc = fake_os.writes[0][1]
        _assert_no_verdict_reject(doc)
        # The reply's class side still lands (the VLM did answer it).
        assert doc[F.visible] is True
        assert doc['vlm_item_make'] == 'Chevrolet'
        assert fake_os.live['c1'][F.status] == 'verify_rejected'

    @pytest.mark.asyncio
    async def test_cap_is_configurable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV, '5')
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(tmp_path, monkeypatch, fake_os, combined=_scripted(_reply(None)))
        assert mocks['vlm'].label_combined_batch.await_count == 5
        _assert_no_verdict_reject(fake_os.writes[0][1])

    @pytest.mark.asyncio
    async def test_parse_failure_is_capped_the_same_way(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(tmp_path, monkeypatch, fake_os, combined=_scripted('parse'))
        assert mocks['vlm'].label_combined_batch.await_count == DEFAULT_MAX_NO_VERDICT_ATTEMPTS
        assert len(fake_os.writes) == 1
        doc = fake_os.writes[0][1]
        _assert_no_verdict_reject(doc)
        # No reply, so no class side.
        assert 'vlm_verify_completed_at' not in doc

    @pytest.mark.asyncio
    async def test_null_verdicts_and_parse_failures_share_one_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(
            tmp_path, monkeypatch, fake_os, combined=_scripted(_reply(None), 'parse', _reply(None))
        )
        assert mocks['vlm'].label_combined_batch.await_count == 3
        _assert_no_verdict_reject(fake_os.writes[0][1])

    @pytest.mark.asyncio
    async def test_real_verdict_before_the_cap_writes_normally_and_clears_the_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """null, null, accept -> detected. Requeued afterwards, the item
        starts from zero: three more null verdicts are needed to cap it
        (a stale count of 2 would cap it after one)."""
        monkeypatch.delenv(ENV, raising=False)
        F = get_region_fields()
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)

        def requeue(n: int) -> None:
            if n == 1:
                fake_os.live['c1'][F.status] = 'pending_detection'
                fake_os.searchable = copy.deepcopy(fake_os.live)

        mocks = await _run(
            tmp_path,
            monkeypatch,
            fake_os,
            combined=_scripted(_reply(None), _reply(None), _reply(True), _reply(None)),
            until_writes=2,
            on_write=requeue,
        )
        assert fake_os.writes[0][1][F.status] == 'detected'
        assert fake_os.writes[0][1][F.rejection_reason] is None
        assert len(fake_os.writes) == 2
        _assert_no_verdict_reject(fake_os.writes[1][1])
        assert mocks['vlm'].label_combined_batch.await_count == 3 + 3

    @pytest.mark.asyncio
    async def test_explicit_false_is_still_a_verifier_reject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _run(tmp_path, monkeypatch, fake_os, combined=_scripted(_reply(None), _reply(False)))
        F = get_region_fields()
        doc = fake_os.writes[0][1]
        assert doc[F.rejection_reason] == REJECT_REASON_VERIFIER
        assert doc[F.bbox_correct] is False

    @pytest.mark.asyncio
    async def test_transport_failure_keeps_retrying_and_never_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(tmp_path, monkeypatch, fake_os, combined=_scripted('transport'))
        F = get_region_fields()
        assert mocks['vlm'].label_combined_batch.await_count > DEFAULT_MAX_NO_VERDICT_ATTEMPTS
        assert fake_os.writes == []
        assert fake_os.live['c1'][F.status] == 'pending_detection'

    @pytest.mark.asyncio
    async def test_transport_failures_do_not_count_toward_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """null, outage x3, null -> still pending after the outage; the cap
        is reached only by the third no-verdict reply."""
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(
            tmp_path,
            monkeypatch,
            fake_os,
            combined=_scripted(
                _reply(None), 'transport', 'transport', 'transport', _reply(None), _reply(None)
            ),
        )
        assert mocks['vlm'].label_combined_batch.await_count == 6
        _assert_no_verdict_reject(fake_os.writes[0][1])


class TestVisibilityNoVerdictIsCapped:
    @pytest.mark.asyncio
    async def test_empty_visibility_reply_fails_open_to_detection_at_the_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(
            tmp_path,
            monkeypatch,
            fake_os,
            primary=None,
            combined=_scripted(_reply(True)),
            visible=lambda _crops, **_kw: {},
        )
        F = get_region_fields()
        assert mocks['vlm'].plate_visible_batch.await_count == DEFAULT_MAX_NO_VERDICT_ATTEMPTS
        mocks['seg'].segment_plate.assert_awaited_once()
        doc = fake_os.writes[0][1]
        assert doc[F.status] == 'detected'
        assert 'vlm_visible:no_verdict' in doc[F.detector_chain]
        assert 'vlm_visible:no' not in doc[F.detector_chain]

    @pytest.mark.asyncio
    async def test_real_visibility_verdict_before_the_cap_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV, raising=False)
        calls = {'n': 0}

        def visible(crops: list[Any], **_kw: Any) -> dict[str, bool]:
            calls['n'] += 1
            return {} if calls['n'] == 1 else {c.crop_id: False for c in crops}

        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _run(tmp_path, monkeypatch, fake_os, primary=None, visible=visible)
        F = get_region_fields()
        assert mocks['vlm'].plate_visible_batch.await_count == 2
        mocks['seg'].segment_plate.assert_not_awaited()
        assert fake_os.writes[0][1][F.status] == 'no_region_visible'


class TestLabelerSeparatesTransportFromNoVerdict:
    def _labeler(self) -> VlmLabeler:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        lab._post_chat = AsyncMock(side_effect=httpx.ConnectError('boom'))  # type: ignore[method-assign]
        return lab

    @pytest.mark.asyncio
    @pytest.mark.parametrize('n_crops', [1, 3])
    async def test_batch_http_failure_raises_transport_failure(self, n_crops: int) -> None:
        crops = [
            CombinedCrop(crop_id=f'c{i}', jpeg_bytes=b'x', plate_bbox_norm=None)
            for i in range(n_crops)
        ]
        with pytest.raises(CombinedTransportError):
            await self._labeler().label_combined_batch(crops, draw_overlay=False)

    @pytest.mark.asyncio
    async def test_single_call_transport_failure_is_still_a_parse_failure(self) -> None:
        with pytest.raises(CombinedParseFailure) as info:
            await self._labeler().label_combined('c1', b'x', draw_overlay=False)
        assert isinstance(info.value, CombinedTransportError)

    @pytest.mark.asyncio
    async def test_unparseable_reply_is_still_none(self) -> None:
        lab = VlmLabeler(base_url='http://vlm.invalid/v1')
        lab._post_chat = AsyncMock(  # type: ignore[method-assign]
            return_value={'choices': [{'message': {'content': 'no json'}}]}
        )
        crops = [
            CombinedCrop(crop_id=f'c{i}', jpeg_bytes=b'x', plate_bbox_norm=None) for i in range(2)
        ]
        assert await lab.label_combined_batch(crops, draw_overlay=False) == {
            'c0': None,
            'c1': None,
        }


class TestCounterAndConfig:
    def test_counter_reaches_the_cap_then_forgets(self) -> None:
        c = NoVerdictCounter(3)
        assert [c.record('a'), c.record('a'), c.record('a')] == [False, False, True]
        assert c.count('a') == 0
        assert c.record('a') is False

    def test_clear_resets_the_count(self) -> None:
        c = NoVerdictCounter(2)
        c.record('a')
        c.clear('a')
        assert c.record('a') is False
        assert len(c) == 1

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [(None, 3), ('', 3), ('7', 7), ('0', 1), ('-2', 1), ('lots', 3)],
    )
    def test_env_parsing(
        self, monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
    ) -> None:
        if raw is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, raw)
        assert max_no_verdict_attempts() == expected


class TestRejectionReasonVocabulary:
    def test_catalog_covers_every_pipeline_reason(self) -> None:
        by_id = {r['id']: r for r in rejection_reason_catalog()}
        assert by_id[REJECT_REASON_VERIFIER]['kind'] == 'model_verdict'
        assert by_id[REJECT_REASON_NO_VERDICT]['kind'] == 'needs_human'
        assert 'needs human review' in by_id[REJECT_REASON_NO_VERDICT]['label']
        sanity = by_id[REJECT_REASON_SANITY_PREFIX]
        assert (sanity['kind'], sanity['match']) == ('automatic', 'prefix')
        assert '{detail}' in sanity['label_template']
        for r in by_id.values():
            assert set(r) == {'id', 'label', 'kind', 'match', 'label_template'}
            assert r['label']

    def test_served_by_the_regions_vocabulary_route(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.routers.curation import router as curation_router

        app = FastAPI()
        app.include_router(curation_router)
        with TestClient(app) as client:
            body = client.get('/curation/regions/vocabulary').json()
        assert body['rejection_reasons'] == rejection_reason_catalog()

    def test_no_vlm_geometry_reject_uses_the_served_prefix(self) -> None:
        """The no-VLM path's gate reject is covered by the ``sanity_reject:``
        vocabulary entry like every other geometry reject."""
        import asyncio

        from scripts.curation.worker.region_text_stage import accept_without_vlm
        from scripts.curation.worker.state import _ItemTask

        t = _ItemTask(
            crop_id='c1',
            image_path='',
            vehicle_bbox_norm=(0.1, 0.1, 0.9, 0.9),
            plate_status='pending_detection',
            class_name='sedan',
        )
        t.crop_jpeg = b'x'
        t.candidate_in_crop = (0.5, 0.5, 0.5, 0.6)  # zero width
        t.candidate_in_source = (0.5, 0.5, 0.5, 0.6)
        t.candidate_source = 'detector'
        asyncio.run(accept_without_vlm(t, ocr=MagicMock(), profile=_profile()))
        F = get_region_fields()
        assert (
            t.update_doc[F.rejection_reason] == f'{REJECT_REASON_SANITY_PREFIX}degenerate_zero_size'
        )
