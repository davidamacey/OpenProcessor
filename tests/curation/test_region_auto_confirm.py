"""Machine auto-confirm is not human validation (DQ-M1).

``region_validated`` means a human confirmed the region; only human
writers set it. When the worker's auto-confirm policy accepts a box
(detector + verifier agree strongly enough) it records that as
``region_auto_confirmed`` and leaves ``region_validated`` false, so the
box stays in the human region-review queue while still being an accepted
(``detected``) region for export. The repair re-labels rows an older
worker stamped ``region_validated=true`` without a human.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from scripts.curation.worker.verify import _region_write_doc
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.services.curation.region_validation_repair import (
    apply_region_validation_repair,
    is_human_region_verdict,
    plan_region_validation_repair,
)
from src.services.curation.review_queries import build_tab_query
from src.services.curation.wire import serialize_item
from src.services.detection.cascade_detect import RegionCandidate
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch, _item
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


F = get_region_fields()
INDEX = base_curation_config().items_index


class TestWorkerWrites:
    def test_auto_confirm_is_recorded_apart_from_validation(self) -> None:
        doc = _region_write_doc(
            region_in_source=(0.1, 0.1, 0.2, 0.2),
            score=0.9,
            detector='det_model',
            detector_version='1',
            chain=[],
            auto_confirmed=True,
            verifier='vlm-model',
        )
        assert doc[F.validated] is False
        assert doc[F.auto_confirmed] is True

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_combined_accept_writes_an_accepted_box(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W8: the combined-call accept path writes the box-list shape
        (``region_boxes[i].state == 'accepted'``) AND restores the
        item-level ``region_verified``/``region_auto_confirmed``/
        ``region_validated`` fields (W8 M2 fix, pipeline-wiring review
        2026-09-27) -- these stopped being written when the box-list
        rewrite landed, going blind every reader that still filters on
        them (regions.py's ``verified`` filter, the training-candidate
        cohorts, region_requeue.py). ``region_validated`` stays human-only
        (unchanged, DQ-M1); ``region_auto_confirmed`` is the box-aware
        rule: >=1 accepted box, VLM confidence 'high' here, so it fires.
        """
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=VlmCombinedReply(
                img_id='c1',
                region_visible=True,
                region_boxes=[
                    VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply='DNV20')
                ],
            ),
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.boxes][0]['state'] == 'accepted'
        # M2: item-level verification fields restored.
        assert doc[F.verified] is True
        assert doc[F.validated] is False
        assert doc[F.auto_confirmed] is True
        assert doc[F.verifier] is not None
        assert doc[F.verifier_version] is not None
        assert doc[F.verified_at] is not None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_low_confidence_accept_is_not_auto_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2 box-aware rule: an accepted box the VLM only rated 'medium'
        confidence, off a detector score below the auto-confirm floor,
        is verified (the VLM did answer) but NOT auto-confirmed."""
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.5, source='det'),
            segmenter=None,
            reply=VlmCombinedReply(
                img_id='c1',
                region_visible=True,
                region_boxes=[VlmBoxVerdict(box=1, bbox_correct=True, confidence='medium')],
            ),
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.verified] is True
        assert doc[F.auto_confirmed] is False
        assert doc[F.validated] is False

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_streaming_worker_no_vlm_path_is_unverified_and_unconfirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2: the no-VLM-configured accept path (``accept_without_vlm``)
        never calls the VLM -- verified/auto_confirmed must both stay
        False, the same as the pre-W8 skip-verify write."""
        fake = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive(
            tmp_path,
            monkeypatch,
            fake_os=fake,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            vlm_url='',
        )
        doc = fake.live['c1']
        assert doc[F.status] == 'detected'
        assert doc[F.boxes][0]['state'] == 'accepted'
        assert doc[F.verified] is False
        assert doc[F.auto_confirmed] is False
        assert doc[F.validated] is False


def test_auto_confirmed_region_is_on_the_wire_and_not_label_validated() -> None:
    item = serialize_item(
        {'crop_id': 'x', F.status: 'detected', F.auto_confirmed: True, F.validated: False},
        'x',
        api_prefix='',
    )
    assert item['region_auto_confirmed'] is True
    assert item['region_validated'] is False
    assert item['label_validated'] is False


def test_region_review_tab_keeps_auto_confirmed_regions() -> None:
    """The region tab excludes only human-validated regions, so an
    auto-confirmed (unvalidated) region is in the human queue."""
    _must, must_not, _reason = build_tab_query(
        'regions', include_test=True, text=None, max_rank=None
    )
    assert {'term': {F.validated: True}} in must_not
    assert not any(F.auto_confirmed in str(clause) for clause in must_not)


def _machine(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        F.status: 'detected',
        F.bbox_norm: [0.1, 0.1, 0.2, 0.2],
        F.validated: True,
        F.verified: True,
        F.verifier: 'vlm_model',
        F.detector: 'det_model',
        **extra,
    }


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            INDEX: {
                'm1': _machine('m1'),
                'm2': _machine('m2', **{F.source: 'segmenter'}),
                'h_label': _machine('h_label', **{F.label_source: 'human'}),
                'h_verifier': _machine('h_verifier', **{F.verifier: 'human'}),
                'h_detector': _machine('h_detector', **{F.detector: 'human'}),
                'h_history': _machine(
                    'h_history',
                    edit_history=[{'kind': 'region', 'writer': 'human:patch_region_meta'}],
                ),
                'fp': _machine('fp', **{F.status: 'false_positive'}),
                'plain': {'crop_id': 'plain', F.status: 'detected', F.validated: False},
            }
        }
    )


class TestRepair:
    @pytest.mark.parametrize(
        ('doc', 'human'),
        [
            (_machine('a'), False),
            (_machine('a', **{F.label_source: 'human:batch'}), True),
            (_machine('a', **{F.verifier: 'human'}), True),
            (_machine('a', **{F.detector: 'human'}), True),
        ],
    )
    def test_human_verdict_detection(self, doc: dict[str, Any], human: bool) -> None:
        assert is_human_region_verdict(doc) is human

    @pytest.mark.asyncio
    async def test_dry_run_plans_only_machine_validated_rows(
        self, fake_os: QueryFakeOpenSearch
    ) -> None:
        plan = await plan_region_validation_repair(fake_os, index=INDEX)
        assert sorted(plan.machine_ids) == ['fp', 'm1', 'm2']
        assert plan.human_kept == 4
        assert plan.by_status == {'detected': 2, 'false_positive': 1}
        # Planning writes nothing.
        assert fake_os.docs(INDEX)['m1'][F.validated] is True

    @pytest.mark.asyncio
    async def test_apply_moves_machine_validation_to_auto_confirmed(
        self, fake_os: QueryFakeOpenSearch
    ) -> None:
        plan = await plan_region_validation_repair(fake_os, index=INDEX)
        result = await apply_region_validation_repair(fake_os, plan, index=INDEX)
        assert result['updated'] == 3
        docs = fake_os.docs(INDEX)
        for cid in ('m1', 'm2', 'fp'):
            assert docs[cid][F.validated] is False
            assert docs[cid][F.auto_confirmed] is True
        for cid in ('h_label', 'h_verifier', 'h_detector', 'h_history'):
            assert docs[cid][F.validated] is True
            assert F.auto_confirmed not in docs[cid]

    @pytest.mark.asyncio
    async def test_apply_skips_a_row_a_human_touched_since_planning(
        self, fake_os: QueryFakeOpenSearch
    ) -> None:
        plan = await plan_region_validation_repair(fake_os, index=INDEX)
        fake_os.docs(INDEX)['m1'][F.label_source] = 'human'
        result = await apply_region_validation_repair(fake_os, plan, index=INDEX)
        assert result['updated'] == 2
        assert fake_os.docs(INDEX)['m1'][F.validated] is True


@pytest.mark.asyncio
async def test_cli_dry_run_reports_and_writes_nothing(
    fake_os: QueryFakeOpenSearch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.curation import repair_region_validation as cli

    args = cli.build_parser().parse_args(['--index', INDEX])
    assert await cli.run(args, fake_os) == 0
    out = capsys.readouterr().out
    assert '7 validated region(s): 3 machine-set, 4 human' in out
    assert 'Dry-run only' in out
    assert fake_os.docs(INDEX)['m1'][F.validated] is True
