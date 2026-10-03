"""The sweeper resumes ingest-time passes a restart left ``pending``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import FakeSegmenter, StatefulRegistry, cand, ingested_world
from curation.reprocess_fixtures import docs, images_index
from src.services.config_store.store import StoredConfig, get_config_store, reset_config_stores
from src.services.curation.open_vocab_sweeper import (
    STALE_AFTER_S,
    start_open_vocab_sweeper,
    sweep_pending_open_vocab,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_config_stores()
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)
    yield
    reset_config_stores()


def _activate(*, run_on_ingest: bool = True) -> None:
    body = {'targets': [{'prompt': 'cone', 'class_name': 'cone'}], 'run_on_ingest': run_on_ingest}
    stored = StoredConfig(kind='open_vocab_set', name='street', revision=2, body=body)
    get_config_store().apply_local(
        config_revision=1,
        open_vocab_set=stored,
        active_open_vocab=('street', 2),
        active_open_vocab_body=stored,
    )


def _stamp(fake: Any, image_id: str, age_s: float | None) -> None:
    doc = fake.docs(images_index())[image_id]
    doc['open_vocab_status'] = 'pending'
    if age_s is not None:
        doc['open_vocab_status_at'] = (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()


async def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, list[str], Any]:
    fake, service, ids = await ingested_world(tmp_path, monkeypatch, n=3)
    seg = FakeSegmenter()
    seg.default = [cand()]
    monkeypatch.setattr('src.services.curation.open_vocab_sweeper.segment_image_http', seg)

    async def factory(_opensearch: Any) -> Any:
        return service

    return fake, factory, ids, seg


@pytest.mark.asyncio
async def test_stale_and_unstamped_pending_images_are_finished_and_fresh_ones_are_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, factory, (old, unstamped, fresh), seg = await _setup(tmp_path, monkeypatch)
    _activate()
    _stamp(fake, old, STALE_AFTER_S + 60)
    _stamp(fake, unstamped, None)
    _stamp(fake, fresh, 5)

    assert await sweep_pending_open_vocab(fake, factory) == 2

    status = {i: fake.docs(images_index())[i]['open_vocab_status'] for i in (old, unstamped, fresh)}
    assert status == {old: 'done', unstamped: 'done', fresh: 'pending'}
    assert len(seg.calls) == 2
    assert {d['image_id'] for d in docs(fake).values()} == {old, unstamped}


@pytest.mark.asyncio
async def test_nothing_runs_without_an_opted_in_active_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, factory, ids, seg = await _setup(tmp_path, monkeypatch)
    _stamp(fake, ids[0], None)

    assert await sweep_pending_open_vocab(fake, factory) == 0  # no active set
    _activate(run_on_ingest=False)
    assert await sweep_pending_open_vocab(fake, factory) == 0  # set does not opt in
    assert seg.calls == []
    assert fake.docs(images_index())[ids[0]]['open_vocab_status'] == 'pending'


@pytest.mark.asyncio
async def test_a_segmenter_outage_leaves_the_images_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, factory, ids, seg = await _setup(tmp_path, monkeypatch)
    _activate()
    seg.down = True
    _stamp(fake, ids[0], None)

    await sweep_pending_open_vocab(fake, factory)

    assert fake.docs(images_index())[ids[0]]['open_vocab_status'] == 'pending'
    assert docs(fake) == {}


@pytest.mark.asyncio
async def test_the_interval_setting_can_turn_the_sweeper_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_OPEN_VOCAB_SWEEP_S', '0')
    assert start_open_vocab_sweeper() is None
    monkeypatch.setenv('OP_OPEN_VOCAB_SWEEP_S', '3600')
    task = start_open_vocab_sweeper()
    assert task is not None
    task.cancel()
