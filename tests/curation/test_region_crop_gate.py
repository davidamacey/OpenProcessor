"""The crop region stage's tier-3 gate and per-project stage pause, driven through the
streaming worker (``_drive_worker``): skipped items are stamped and never lost, a
human-owned class is never skipped, a paused stage touches nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation._project_worker_utils import REGION_STAGE_PAUSED_FLAG_NAME
from src.config import get_region_fields

from .test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures('reference_region_profile')

GATED = {
    'gate_hit_rate': True,
    'gate_hit_window': 1,
    'gate_hit_miss_threshold': 1,
    'gate_hit_sample_floor': 0.0,
}


def _items(n: int, **overrides: Any) -> dict[str, dict[str, Any]]:
    out = {}
    for i in range(n):
        doc = {
            **_item(),
            **overrides,
            'crop_id': f'c{i}',
            'created_at': f'2026-09-24T00:00:0{i}+00:00',
        }
        out[f'c{i}'] = doc
    return out


async def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docs: dict, **kw: Any) -> tuple:
    fake_os = _FakeOpenSearch(docs, search_delay=0.0, lag_searches=0)
    mocks = await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=None,
        reply=_accept(),
        until_writes=kw.pop('until_writes', len(docs)),
        **kw,
    )
    return fake_os, mocks


@pytest.mark.asyncio
async def test_off_by_default_every_item_gets_its_segmenter_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_os, mocks = await _run(tmp_path, monkeypatch, _items(4))
    F = get_region_fields()
    assert mocks['seg'].segment_multi.await_count == 4
    assert all(F.gate_skip not in d for d in fake_os.live.values())


@pytest.mark.asyncio
async def test_a_missing_class_is_skipped_stamped_and_stays_rerunnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_os, mocks = await _run(tmp_path, monkeypatch, _items(4), profile_overrides=GATED)
    F = get_region_fields()
    skipped = [d for d in fake_os.live.values() if d.get(F.gate_skip)]
    asked = mocks['seg'].segment_multi.await_count
    assert skipped, 'once the class missed, the next items are skipped'
    assert asked + len(skipped) == 4  # every item was either looked at or stamped
    for doc in skipped:
        assert doc[F.status] == 'no_region_box'
        assert doc[F.gate_skip] == 'tier3_hit_rate'
        assert 'gate:tier3_hit_rate' in doc[F.detector_chain]


@pytest.mark.asyncio
async def test_a_human_owned_class_is_never_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = _items(4, class_source='human', class_validated=True)
    fake_os, mocks = await _run(tmp_path, monkeypatch, docs, profile_overrides=GATED)
    F = get_region_fields()
    assert mocks['seg'].segment_multi.await_count == 4
    assert all(F.gate_skip not in d for d in fake_os.live.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('fetch_sees_pause', [True, False])
async def test_a_paused_region_stage_runs_nothing_and_loses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fetch_sees_pause: bool
) -> None:
    # False: the producer still fetches the project (as for an item already
    # queued when the pause lands), so only the consumer-side guard holds it.
    if not fetch_sees_pause:
        monkeypatch.setattr(
            'scripts.curation.worker.fairness.is_region_stage_paused', lambda _record: False
        )
    import src.config.curation as curation_config

    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(curation_config, '_default_curation_config', None)
    state_dir = tmp_path / 'state' / 'projects' / 'default'
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / REGION_STAGE_PAUSED_FLAG_NAME).touch()
    fake_os, mocks = await _run(tmp_path, monkeypatch, _items(2), until_writes=0)
    F = get_region_fields()
    assert fake_os.writes == []
    assert mocks['seg'].segment_multi.await_count == 0
    assert {d[F.status] for d in fake_os.live.values()} == {'pending_detection'}
