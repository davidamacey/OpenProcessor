"""The per-photo flow of an ingest request: each photo moves on its own from the engine's answer to
its write, the database work runs in at most three commit groups, the answers follow the table of
``src/services/curation/ingest_finish.py``, and the finish survives a cancellation."""

from __future__ import annotations

import asyncio
import dataclasses
import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from curation.engine_fakes import engine_service, register
from curation.ingest_fakes import _jpeg_bytes, _two_class_registry
from curation.ingest_session_fakes import Sessions, count_sessions
from src.domain.embedding_state import EMBEDDED
from src.services.curation import ingest_gpu_decode, ingest_index
from src.services.curation.ingest_policy import EmbeddingPolicy, IngestPolicy, select_for_embedding
from src.services.near_duplicates import ingest_claim


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _fresh_ready_cache() -> Any:
    ingest_gpu_decode._READY.clear()
    yield
    ingest_gpu_decode._READY.clear()


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> Sessions:
    return count_sessions(monkeypatch)


def _photos(sizes: dict[bytes, tuple[int, int]], n: int, *, seed0: int = 1) -> list[bytes]:
    return [register(sizes, _jpeg_bytes((64, 48), seed=seed0 + i), (640, 480)) for i in range(n)]


def _paths(n: int, prefix: str = 'p') -> list[str]:
    return [f'/flow/{prefix}{i}.jpg' for i in range(n)]


def _one_at_a_time(engine: Any, pause_s: float = 0.002) -> None:
    """The engine answers one photo at a time (as a busy card does), so photos arrive spread out
    and the commit groups form while later photos are still in their stage."""
    lock = asyncio.Lock()
    original = engine.infer

    async def infer(*args: Any, **kwargs: Any) -> Any:
        async with lock:
            await asyncio.sleep(pause_s)
            return await original(*args, **kwargs)

    engine.infer = infer


async def _until(condition: Any, seconds: float = 10.0) -> None:
    async with asyncio.timeout(seconds):
        while not condition():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_the_first_photo_is_written_before_the_last_photos_engine_answer(
    tmp_path: Path,
) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    photos = _photos(sizes, 6)
    last = photos[-1]
    engine.gates[last] = asyncio.Event()
    written_before_last_answer: list[int] = []

    async def release_the_last_photo_once_a_document_exists() -> None:
        await _until(lambda: bool(os_fake.images))
        written_before_last_answer.append(len(os_fake.images))
        engine.gates[last].set()

    releaser = asyncio.create_task(release_the_last_photo_once_a_document_exists())
    async with asyncio.timeout(20):  # a barrier would wait for the last answer: it never comes
        result = await svc.ingest_batch(photos, _paths(6))
    await releaser
    assert written_before_last_answer
    assert written_before_last_answer[0] >= 1
    assert result.summary.successful == 6


@pytest.mark.parametrize('n', [1, 8, 32, 64])
@pytest.mark.asyncio
async def test_sessions_per_request_follow_the_formula_whatever_the_photo_count(
    sessions: Sessions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    """Fresh photos in a warm project: the lookup, one transaction per commit group (at most 3),
    the events' publish: at most 5, one at a time."""
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    _one_at_a_time(engine)
    groups: list[int] = []
    real_commit_group = ingest_index.commit_group

    async def counted(service: Any, prepared: Any) -> Any:
        groups.append(len(prepared))
        return await real_commit_group(service, prepared)

    monkeypatch.setattr('src.services.curation.ingest_finish.commit_group', counted)
    result = await svc.ingest_batch(_photos(sizes, n), _paths(n))
    assert result.summary.successful == n
    assert len(os_fake.images) == n
    assert 1 <= len(groups) <= 3
    assert sum(groups) == n
    assert len(groups) == (1 if n == 1 else len(groups))
    assert sessions.opened == 1 + len(groups) + 1  # lookup, the groups, the one publish
    assert sessions.opened <= 5
    assert sessions.peak == 1  # one connection at a time, not one per photo


@pytest.mark.asyncio
async def test_a_duplicate_rich_batch_opens_the_sessions_of_its_distinct_photos(
    sessions: Sessions, tmp_path: Path
) -> None:
    svc, _os, _triton, engine, sizes = engine_service(tmp_path)
    _one_at_a_time(engine)
    distinct = _photos(sizes, 8)
    batch = [*distinct, *distinct, *distinct, *distinct]
    result = await svc.ingest_batch(batch, _paths(len(batch)))
    assert result.summary.successful == 8
    assert result.summary.duplicates == 24
    assert sessions.opened <= 5
    assert sessions.peak == 1


@pytest.mark.asyncio
async def test_a_pool_timeout_in_the_first_group_is_a_503_with_nothing_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, os_fake, _triton, _engine, sizes = engine_service(tmp_path)

    async def busy(*_a: Any) -> Any:
        raise PoolTimeoutError('QueuePool limit reached')

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', busy)
    with pytest.raises(PoolTimeoutError):  # through run_all's unwrap: the 503 handler answers
        await svc.ingest_batch(_photos(sizes, 3), _paths(3))
    assert not os_fake.images


@pytest.mark.asyncio
async def test_a_pool_timeout_in_the_second_group_keeps_the_first_groups_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    photos = _photos(sizes, 6)
    for data in photos[2:]:
        engine.gates[data] = asyncio.Event()  # the later photos reach the engine's answer late
    calls = [0]
    real = ingest_claim.claim_and_mark_in_session

    async def second_group_busy(*args: Any) -> Any:
        calls[0] += 1
        if calls[0] == 2:
            raise PoolTimeoutError('QueuePool limit reached')
        return await real(*args)

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', second_group_busy)
    announced: list[str] = []

    async def publish(crops: Any) -> None:
        announced.extend(crop_id for crop_id, _path in crops)

    svc._publish_created = publish

    async def release_the_rest_after_the_first_group() -> None:
        await _until(lambda: len(os_fake.images) >= 2)
        for data in photos[2:]:
            engine.gates[data].set()

    releaser = asyncio.create_task(release_the_rest_after_the_first_group())
    result = await svc.ingest_batch(photos, _paths(6))
    await releaser
    by_status = [(r.status, r.error_kind) for r in result.results]
    assert by_status[:2] == [('success', None)] * 2
    assert by_status[2:] == [('failed', 'database_busy')] * 4
    assert result.status == 'partial'
    assert len(os_fake.images) == 2
    assert sorted(announced) == sorted(os_fake.items)  # the stored photos' events went out


@pytest.mark.asyncio
async def test_open_vocabulary_is_scheduled_for_the_stored_photos_of_a_partial_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.curation import upload_ingest

    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    svc.config = dataclasses.replace(
        svc.config, upload_root=tmp_path / 'uploads', upload_accepted_extensions=('.jpg',)
    )
    photos = _photos(sizes, 6)
    for data in photos[2:]:
        engine.gates[data] = asyncio.Event()
    calls = [0]
    real = ingest_claim.claim_and_mark_in_session

    async def second_group_busy(*args: Any) -> Any:
        calls[0] += 1
        if calls[0] == 2:
            raise PoolTimeoutError('busy')
        return await real(*args)

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', second_group_busy)
    scheduled: list[list[str]] = []

    async def schedule(_opensearch: Any, image_ids: list[str]) -> bool:
        scheduled.append(list(image_ids))
        return True

    monkeypatch.setattr(upload_ingest, 'schedule_open_vocab_after_ingest', schedule)

    async def release() -> None:
        await _until(lambda: len(os_fake.images) >= 2)
        for data in photos[2:]:
            engine.gates[data].set()

    releaser = asyncio.create_task(release())
    entries = [upload_ingest.StagedEntry(f'u{i}', data, '.jpg') for i, data in enumerate(photos)]
    result, early = await upload_ingest.stage_and_ingest(
        os_fake, svc, entries, config=svc.config, source='upload', run_id=None
    )
    await releaser
    assert not early
    assert result is not None
    stored = sorted(r.image_id for r in result.results if r.status == 'success')
    assert len(stored) == 2
    assert [sorted(ids) for ids in scheduled] == [stored]


@pytest.mark.asyncio
async def test_a_failure_of_one_photo_never_cancels_the_photos_around_it(tmp_path: Path) -> None:
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path)
    photos = _photos(sizes, 5)
    original = engine.infer

    async def infer(model: str, inputs: list, outputs: list, **k: Any) -> Any:
        if len(engine.calls) == 2:
            engine.raises = OSError('one call failed')
        else:
            engine.raises = None
        return await original(model, inputs, outputs, **k)

    engine.infer = infer
    result = await svc.ingest_batch(photos, _paths(5))
    assert sorted(r.status for r in result.results) == [
        'failed',
        'success',
        'success',
        'success',
        'success',
    ]
    assert len(os_fake.images) == 4


@pytest.mark.asyncio
async def test_the_finish_survives_two_cancellations_of_the_request(tmp_path: Path) -> None:
    """A stored photo never loses its events: the finish runs to its end although the task
    awaiting the request is cancelled, and again."""
    svc, os_fake, _triton, _engine, sizes = engine_service(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    announced: list[str] = []

    async def publish(crops: Any) -> None:
        started.set()
        await release.wait()
        announced.extend(crop_id for crop_id, _path in crops)

    svc._publish_created = publish
    (photo,) = _photos(sizes, 1)
    task = asyncio.create_task(svc.ingest_batch([photo], _paths(1)))
    async with asyncio.timeout(10):
        await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()  # the second cancellation, while the events are being announced
    await asyncio.sleep(0)
    assert announced == []
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(announced) == sorted(os_fake.items)
    assert len(announced) == 1


def _name_registry_service(tmp_path: Path, *, assigns_class: bool, mode: str, classes: list[str]):
    labels = tmp_path / 'labels.txt'
    labels.write_text('gadget\nWidget Thing\n')
    svc, os_fake, _triton, engine, sizes = engine_service(tmp_path, registry=_two_class_registry())
    svc.registry.classes[1].class_name = 'widget_thing'
    profile = dataclasses.replace(svc.profile, assigns_class=assigns_class, labels_path=str(labels))
    svc.profile = profile
    svc.detector.profile = profile
    svc.policy = IngestPolicy(embedding=EmbeddingPolicy(mode=mode, classes=classes))
    return svc, os_fake, engine, sizes


@pytest.mark.parametrize('assigns_class', [True, False])
@pytest.mark.parametrize(
    ('mode', 'classes'),
    [
        ('all', []),
        ('selected', ['widget_thing']),
        ('selected', ['gadget']),
        ('lazy', []),
        ('none', []),
    ],
)
@pytest.mark.asyncio
async def test_the_embedding_mask_and_the_region_seed_equal_resolving_classes_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assigns_class: bool,
    mode: str,
    classes: list[str],
) -> None:
    from src.config.region_fields import get_region_fields
    from src.services.curation import item_doc

    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    monkeypatch.setattr(
        item_doc,
        'get_active_region_profile',
        lambda: types.SimpleNamespace(parent_classes=['widget_thing']),
    )
    svc, os_fake, _engine, sizes = _name_registry_service(
        tmp_path, assigns_class=assigns_class, mode=mode, classes=classes
    )
    (photo,) = _photos(sizes, 1)
    # the reference: the items of the one detection, classes resolved BEFORE the policy reads them
    reference = svc.detector.decode_primary_row(
        np.array([[16.0, 16.0, 192.0, 192.0]], 'float32'),
        np.array([0.9], 'float32'),
        np.array([1], 'int32'),
        1.0,
        (0.0, 0.0),
    )
    await svc.resolve_classes([reference])
    expected_state = select_for_embedding(reference, svc.policy.embedding, 640, 480)[0]
    expected_region = item_doc.region_seed_status(reference[0])

    result = await svc.ingest_batch([photo], _paths(1))
    assert result.summary.successful == 1
    [doc] = list(os_fake.items.values())
    assert doc['embedding_state'] == expected_state
    assert ('pe_embedding' in doc) == (expected_state == EMBEDDED)
    assert doc['class_name'] == reference[0].class_name == 'widget_thing'
    stored_region = doc.get(get_region_fields().status)
    assert (None if expected_region is None else expected_region.value) == stored_region


@pytest.mark.parametrize(('classes', 'embedded'), [(['widget_thing'], True), (['gadget'], False)])
@pytest.mark.asyncio
async def test_a_plan_that_skips_a_class_leaves_that_crop_without_a_vector_and_unshifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classes: list[str], embedded: bool
) -> None:
    """With ``embed_classes`` the engine answers only the picked rows; the API must read each
    detection's own row (here the one detection is the only row or no row), never a shifted one."""
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(tmp_path))
    svc, os_fake, engine, sizes = _name_registry_service(
        tmp_path, assigns_class=False, mode='selected', classes=classes
    )
    (photo,) = _photos(sizes, 1)
    result = await svc.ingest_batch([photo], _paths(1))
    assert result.summary.successful == 1
    [call] = engine.calls
    assert call['plan'].embed_classes == (frozenset({1}) if embedded else frozenset({0}))
    [doc] = list(os_fake.items.values())
    assert ('pe_embedding' in doc) == embedded
    assert doc['embedding_state'] == (EMBEDDED if embedded else 'not_selected')
    if embedded:
        assert doc['pe_embedding'] == pytest.approx([0.0, 1.0, 0.0])
