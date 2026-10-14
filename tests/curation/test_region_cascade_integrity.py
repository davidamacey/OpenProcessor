"""Region-cascade integrity regressions found on a live deployment.

1. ``region_detector_chain`` entries are exactly ``<actor>:<event>`` and
   the training-candidate readers' ``term`` filters match what the worker
   writes.
2. One pending transition = one cascade pass: a refresh-lagged pending
   search must not hand the same item to the pipeline twice.
3. The worker resolves the deployment's prompt pack and hands it to the labeler.

The combined-reply parsing regressions live in ``test_vlm_combined_reply_parsing.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from scripts.curation.worker.bulk_writer import _bulk_update
from scripts.curation.worker.state import _ItemTask
from src.config import get_region_fields
from src.config.project_context import current_project
from src.services.detection.cascade_detect.candidate import RegionCandidate
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


if TYPE_CHECKING:
    from pathlib import Path

from curation.region_cascade_helpers import _accept, _drive_worker, _FakeOpenSearch, _item, _profile


pytestmark = pytest.mark.usefixtures('reference_region_profile')


# =============================================================================
# Runner wiring: the worker must use the deployment's prompt pack
# =============================================================================


class TestRunnerUsesDeploymentPromptPack:
    @pytest.mark.asyncio
    async def test_resolved_pack_is_passed_to_the_labeler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack = PromptPack(**{**GENERIC_ITEM_PACK.__dict__, 'name': 'deployment_pack'})
        from src.services.labeling import vlm_prompts as vlm_prompts_mod

        monkeypatch.setattr(vlm_prompts_mod, 'active_prompt_pack', lambda *_a, **_k: pack)
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        mocks = await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det'),
            segmenter=None,
            reply=_accept(),
        )
        assert mocks['vlm_cls'].call_args.args[1] is pack


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
        assert mocks['primary'].detect_batch_multi.await_count == 1
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
        assert doc[F.boxes][0]['text'] == 'DNV20'
        assert doc[F.boxes][0]['state'] == 'accepted'
        assert doc[F.detector_chain] == [f'{det}:hit', f'{det}:combined_verify_ok']

    @pytest.mark.asyncio
    async def test_stale_write_is_dropped_by_the_writer(self) -> None:
        """Backstop: a result computed from a pending state the doc has
        already left (a duplicate consumer, or a human edit) never lands."""
        F = get_region_fields()
        fake_os = _FakeOpenSearch(
            {'c1': {**_item(), F.status: 'detected'}}, search_delay=0.0, lag_searches=0
        )
        task = _ItemTask(
            project=current_project().record,
            crop_id='c1',
            image_path='',
            item_bbox_norm=(0.1, 0.1, 0.9, 0.9),
            region_status='pending_detection',
            class_name='sedan',
            group='cars',
        )
        task.update_doc = {F.status: 'detected', F.reason: 'again'}
        task.detection_trace = ['det:hit']
        n_written, _ = await _bulk_update(fake_os, [task])  # type: ignore[arg-type]
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

        query, _box, _ = _training_candidate_query('detector_blind_spots', profile)
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
        query, _box, _ = _training_candidate_query('disagreement', profile)
        terms = [
            c['term'][F.detector_chain]
            for c in query['bool']['filter']
            if 'term' in c and F.detector_chain in c['term']
        ]
        assert terms == [f'{profile.detector_model}:hit', f'{profile.segmenter_name}:hit']
        assert all(_CANONICAL.match(t) for t in terms)
