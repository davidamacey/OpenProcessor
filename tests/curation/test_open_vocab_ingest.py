"""The ingest-time opt-in -- off by default, never inline, durable ``pending`` marker, the drain
job takes every pending image, an outage leaves the image pending.

The start of the job is faked here (no SQL is under test); ``test_open_vocab_coalesce.py`` starts
the real job on a database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import FakeSegmenter, StatefulRegistry, cand, ingested_world
from curation.reprocess_fixtures import docs, images_index
from src.services.config_store.snapshot import StoredConfig
from src.services.config_store.store import get_config_store, reset_config_stores
from src.services.curation import open_vocab_ingest
from src.services.curation.open_vocab_ingest import drain_pending, schedule_open_vocab_after_ingest


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_config_stores()
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)
    yield
    reset_config_stores()


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Every start of the drain job, recorded instead of queued."""
    started: list[bool] = []

    async def start_drain() -> bool:
        started.append(True)
        return True

    monkeypatch.setattr(open_vocab_ingest, 'start_drain', start_drain)
    return started


def _activate(*, run_on_ingest: bool) -> None:
    body = {'targets': [{'prompt': 'cone', 'class_name': 'cone'}], 'run_on_ingest': run_on_ingest}
    stored = StoredConfig(kind='open_vocab_set', name='street', revision=2, body=body)
    get_config_store().apply_local(
        config_revision=0,
        open_vocab_set=stored,
        active_open_vocab=('street', 2),
        active_open_vocab_body=stored,
    )


async def _world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 1
) -> tuple[Any, Any, list[str]]:
    return await ingested_world(tmp_path, monkeypatch, n=n)


def _seg() -> FakeSegmenter:
    seg = FakeSegmenter()
    seg.default = [cand()]
    return seg


def _status(fake: Any, image_id: str) -> str | None:
    return fake.docs(images_index())[image_id].get('open_vocab_status')


@pytest.mark.asyncio
async def test_off_by_default_nothing_is_stamped_or_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, starts: list[bool]
) -> None:
    fake, service, (image_id,) = await _world(tmp_path, monkeypatch)
    seg = _seg()
    _activate(run_on_ingest=False)

    assert await schedule_open_vocab_after_ingest(fake, [image_id]) is False
    assert await drain_pending(fake, service, seg) == {'drained': 0, 'segmenter_down': 0}
    assert (seg.calls, _status(fake, image_id), docs(fake), starts) == ([], None, {}, [])


@pytest.mark.asyncio
async def test_no_active_set_means_no_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, starts: list[bool]
) -> None:
    fake, _service, (image_id,) = await _world(tmp_path, monkeypatch)
    assert await schedule_open_vocab_after_ingest(fake, [image_id]) is False
    assert starts == []


@pytest.mark.asyncio
async def test_opted_in_images_are_stamped_pending_and_the_job_is_started_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, starts: list[bool]
) -> None:
    fake, _service, (image_id,) = await _world(tmp_path, monkeypatch)
    _activate(run_on_ingest=True)

    assert await schedule_open_vocab_after_ingest(fake, [image_id]) is True
    # Returned before the work ran: the stamp is the durable record, the job does the rest.
    assert _status(fake, image_id) == 'pending'
    assert starts == [True]
    assert docs(fake) == {}


@pytest.mark.asyncio
async def test_the_drain_processes_every_pending_image_including_ones_it_was_not_told_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await _world(tmp_path, monkeypatch, n=3)
    seg = _seg()
    _activate(run_on_ingest=True)
    for image_id in ids:
        fake.docs(images_index())[image_id]['open_vocab_status'] = 'pending'

    assert await drain_pending(fake, service, seg) == {'drained': 3, 'segmenter_down': 0}
    assert {_status(fake, i) for i in ids} == {'done'}
    assert len(seg.calls) == 3
    assert {d['open_vocab_set'] for d in docs(fake).values()} == {'street'}


@pytest.mark.asyncio
async def test_an_outage_leaves_the_image_pending_for_a_later_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await _world(tmp_path, monkeypatch, n=4)
    seg = _seg()
    seg.down = True
    _activate(run_on_ingest=True)
    for image_id in ids:
        fake.docs(images_index())[image_id]['open_vocab_status'] = 'pending'

    result = await drain_pending(fake, service, seg)

    assert result['segmenter_down'] == 1  # three images in a row trip it
    assert {_status(fake, i) for i in ids} == {'pending'}
    assert docs(fake) == {}


@pytest.mark.asyncio
async def test_a_stop_request_ends_the_drain_before_the_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, (image_id,) = await _world(tmp_path, monkeypatch)
    seg = _seg()
    _activate(run_on_ingest=True)
    fake.docs(images_index())[image_id]['open_vocab_status'] = 'pending'

    result = await drain_pending(fake, service, seg, should_stop=lambda: True)

    assert result == {'drained': 0, 'segmenter_down': 0}
    assert (seg.calls, _status(fake, image_id)) == ([], 'pending')


@pytest.mark.asyncio
async def test_the_ingest_routes_never_fail_because_scheduling_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, (image_id,) = await _world(tmp_path, monkeypatch)
    _activate(run_on_ingest=True)

    async def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError('opensearch down')

    monkeypatch.setattr('src.services.curation.open_vocab_ingest.current_active_set', boom)
    assert await schedule_open_vocab_after_ingest(fake, [image_id]) is False
