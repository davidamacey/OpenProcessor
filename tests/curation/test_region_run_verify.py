"""Selectable segmenter verification: per item (`region_run_verify`, stamped by an
assist run) and per project (`region_verify` in the VLM policy). `none` takes the
no-VLM acceptance path even when a VLM is active; an explicit `vlm` with no VLM fails
closed instead of being accepted unverified; the high-confidence skip-verify shortcut
applies only under `vlm`."""

from __future__ import annotations

from typing import Any

import pytest

from curation.region_cascade_helpers import _accept, _drive_worker, _FakeOpenSearch, _item, _profile
from scripts.curation.worker import verify_choice, verify_policy
from src.config import get_region_fields
from src.services.detection.cascade_detect.candidate import RegionCandidate


pytestmark = [pytest.mark.usefixtures('reference_region_profile'), pytest.mark.asyncio]

F = get_region_fields()
PRIMARY = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.9, source='det')
SEGMENT_HIT = RegionCandidate(bbox_norm=(0.3, 0.6, 0.6, 0.75), score=0.99, source='seg')


@pytest.fixture(autouse=True)
def _fresh_policy_cache() -> Any:
    verify_policy.reset_cache()
    yield
    verify_policy.reset_cache()


async def _drive(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, doc: dict[str, Any], **kw: Any
) -> tuple[dict[str, Any], dict[str, Any], _FakeOpenSearch]:
    fake = _FakeOpenSearch({'c1': doc}, search_delay=0.0, lag_searches=0)
    policy = kw.pop('policy', None)
    if policy is not None:
        fake.settings_doc = {'vlm_policy': {'scope': 'all', 'revision': 1, 'region_verify': policy}}
    mocks = await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake,
        primary=kw.pop('primary', PRIMARY),
        segmenter=kw.pop('segmenter', None),
        reply=_accept(),
        **kw,
    )
    return mocks, fake.live['c1'], fake


async def test_run_verify_none_skips_the_vlm_even_when_one_is_active(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = {**_item(), 'region_run_verify': 'none', 'region_run_id': 'asr_1'}

    mocks, live, _ = await _drive(tmp_path, monkeypatch, doc)

    assert mocks['vlm'].label_combined_batch.await_count == 0
    assert live[F.status] == 'detected'
    assert [b['state'] for b in live[F.boxes]] == ['accepted']
    assert live['region_run_verify'] is None, (
        'the per-item choice is spent once the item is written'
    )
    assert live['region_run_id'] == 'asr_1', 'the run id stays as provenance'


async def test_run_verify_vlm_without_a_vlm_fails_closed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = {**_item(), 'region_run_verify': 'vlm', 'region_run_id': 'asr_1'}

    mocks, live, _ = await _drive(tmp_path, monkeypatch, doc, vlm_active=False)

    assert mocks['vlm'].label_combined_batch.await_count == 0
    assert live[F.status] == 'detection_failed'
    assert live[F.reason] == 'vlm_unavailable'
    assert not [b for b in live.get(F.boxes) or [] if b['state'] == 'accepted']


async def test_without_a_choice_and_without_a_vlm_the_item_is_still_accepted_as_before(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mocks, live, _ = await _drive(tmp_path, monkeypatch, _item(), vlm_active=False)

    assert live[F.status] == 'detected'


async def test_the_project_policy_none_applies_to_items_without_a_choice(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks, live, _ = await _drive(tmp_path, monkeypatch, _item(), policy='none')

    assert mocks['vlm'].label_combined_batch.await_count == 0
    assert live[F.status] == 'detected'


async def test_a_per_item_vlm_wins_over_the_project_policy(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = {**_item(), 'region_run_verify': 'vlm'}

    mocks, live, _ = await _drive(tmp_path, monkeypatch, doc, policy='none')

    assert mocks['vlm'].label_combined_batch.await_count == 1
    assert live['region_run_verify'] is None


async def test_the_policy_default_still_verifies_with_the_vlm(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks, _, _ = await _drive(tmp_path, monkeypatch, _item())

    assert mocks['vlm'].label_combined_batch.await_count == 1


async def test_the_skip_verify_shortcut_applies_only_under_vlm(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('scripts.curation.worker.stage_sam._SKIP_VLM_VERIFY_SECONDARY_SCORE', 0.5)
    kw: dict[str, Any] = {'primary': None, 'segmenter': SEGMENT_HIT}
    doc = {**_item(), 'region_run_verify': 'none'}

    _, none_live, _ = await _drive(tmp_path, monkeypatch, doc, **kw)
    assert none_live.get(F.skip_verify) is not True

    verify_policy.reset_cache()
    _, vlm_live, _ = await _drive(tmp_path, monkeypatch, _item(), **kw)
    assert vlm_live.get(F.skip_verify) is True


async def test_the_effective_choice_is_item_then_policy_then_vlm() -> None:
    from scripts.curation.worker.state import _ItemTask

    def task(item: str | None, policy: str) -> Any:
        return _ItemTask(
            crop_id='c',
            image_path='/x',
            item_bbox_norm=(0, 0, 1, 1),
            region_status='pending_detection',
            class_name='',
            region_run_verify=item,
            policy_verify=policy,
        )

    assert verify_choice.effective_verify(task(None, 'vlm')) == 'vlm'
    assert verify_choice.effective_verify(task(None, 'none')) == 'none'
    assert verify_choice.effective_verify(task('vlm', 'none')) == 'vlm'
    assert verify_choice.effective_verify(task('none', 'vlm')) == 'none'
    assert _profile() is not None
