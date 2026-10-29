"""The embed pass reads chunk N+1 while chunk N is processed, and runs its preprocess off the
event-loop thread (#259). Neither changes what is embedded or in which order."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from curation.reprocess_fixtures import item, jpeg_bytes, make_fake, servable_root
from src.services.curation import reprocess_embed, reprocess_images
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items
from src.services.curation.reprocess_embed_spec import EmbedSpec


IDS = [f'img-{i}' for i in range(5)]


class _Pass:
    """Stands in for the datastore reads and the embed step; records their order."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail_chunk: int | None = None) -> None:
        self.events: list[tuple[str, tuple[str, ...]]] = []
        self.release = asyncio.Event()
        self.fail_chunk = fail_chunk
        self.processed: list[list[str]] = []
        monkeypatch.setattr(reprocess_images, 'CHUNK', 2)
        monkeypatch.setattr(reprocess_images, 'existing_images', self.existing_images)
        monkeypatch.setattr(reprocess_images, 'embed_targets', self.embed_targets)
        monkeypatch.setattr(reprocess_images, 'reembed_items', self.reembed_items)
        monkeypatch.setattr(reprocess_images, 'get_ingest_policy', self.policy)
        monkeypatch.setattr(
            reprocess_images, 'get_curation_config', lambda: SimpleNamespace(images_index='img')
        )

    async def existing_images(self, _os: Any, ids: list[str], *, index: str) -> dict[str, Any]:  # noqa: ARG002
        self.events.append(('read', tuple(ids)))
        return {i: {'image_id': i} for i in ids}

    async def embed_targets(self, _os: Any, docs: dict[str, Any], _spec: Any) -> list[str]:
        if self.fail_chunk is not None and IDS[2 * self.fail_chunk] in docs:
            raise RuntimeError('targets read failed')
        return list(docs)

    async def policy(self, _os: Any) -> Any:
        return SimpleNamespace(clustering=SimpleNamespace(automatic=False))

    async def reembed_items(self, _os: Any, _pe: Any, targets: list[str], **_kw: Any) -> dict:
        self.events.append(('embed', tuple(targets)))
        self.processed.append(list(targets))
        if len(self.processed) == 1:
            await self.release.wait()  # chunk 1 is "slow": chunk 2's read must not wait for it
        return {
            'images': len(targets),
            'items': len(targets),
            'crop_written': len(targets),
            'frame_written': 0,
            'region_written': 0,
            'missing_image': 0,
            'decode_failed': 0,
        }


async def _run(**kw: Any) -> Any:
    # GPU decode off: every photo takes the CPU route, which these fakes stand in for.
    service = SimpleNamespace(pe_encoder=object(), config=SimpleNamespace(ingest_gpu_decode=False))
    return await reprocess_images.process_images(
        object(),  # type: ignore[arg-type]
        service,  # type: ignore[arg-type]
        scopes=['embed'],
        image_ids=IDS,
        embed=EmbedSpec(parts=['crop']),
        **kw,
    )


@pytest.mark.asyncio
async def test_the_next_chunk_is_read_and_embedded_while_this_one_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)
    task = asyncio.create_task(_run())
    for _ in range(50):
        await asyncio.sleep(0)
    assert ('read', ('img-2', 'img-3')) in p.events  # read ahead
    # chunk 2 starts while chunk 1 is still being embedded (no barrier between chunks), chunk 3
    # waits: two chunks at most are embedded at once
    assert p.processed == [['img-0', 'img-1'], ['img-2', 'img-3']]
    p.release.set()
    (res,), cancelled = await task
    assert not cancelled
    assert (res.queued, res.failed) == (5, 0)
    assert p.processed[-1] == ['img-4']


@pytest.mark.asyncio
async def test_progress_counts_chunks_in_order_even_when_a_later_one_finishes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)  # chunk 1 is held; chunk 2 finishes first
    seen: list[int] = []
    task = asyncio.create_task(_run(on_progress=lambda done, _failed: seen.append(done)))
    for _ in range(50):
        await asyncio.sleep(0)
    assert seen == []  # nothing is reported done before the first chunk is
    p.release.set()
    await task
    assert seen == [2, 4, 5]


@pytest.mark.asyncio
async def test_a_chunk_embed_is_not_overlapped_when_another_scope_runs_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)
    detected: list[str] = []

    async def redetect(_os: Any, _svc: Any, image_id: str, _doc: Any) -> dict[str, int]:
        detected.append(image_id)
        return dict.fromkeys(
            ('locked_untouched', 'merged', 'refreshed', 'replaced', 'created', 'removed'), 0
        )

    monkeypatch.setattr(reprocess_images, 'redetect_image', redetect)
    p.release.set()
    service = SimpleNamespace(pe_encoder=object(), config=SimpleNamespace(ingest_gpu_decode=False))
    await reprocess_images.process_images(
        object(),  # type: ignore[arg-type]
        service,  # type: ignore[arg-type]
        scopes=['detect', 'embed'],
        image_ids=IDS,
        embed=EmbedSpec(parts=['crop']),
    )
    # detect of chunk N+1 never starts before the embed of chunk N has: one chunk at a time
    assert p.processed == [['img-0', 'img-1'], ['img-2', 'img-3'], ['img-4']]


@pytest.mark.asyncio
async def test_order_and_content_match_the_serial_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pass(monkeypatch)
    p.release.set()
    await _run()
    assert p.processed == [['img-0', 'img-1'], ['img-2', 'img-3'], ['img-4']]


@pytest.mark.asyncio
async def test_a_failed_chunk_does_not_drop_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pass(monkeypatch, fail_chunk=1)
    p.release.set()
    (res,), _ = await _run()
    assert p.processed == [['img-0', 'img-1'], ['img-4']]  # chunk 2's targets read failed
    assert (res.queued, res.failed) == (3, 2)


@pytest.mark.asyncio
async def test_cancel_stops_the_read_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pass(monkeypatch)
    p.release.set()
    calls = iter([False, True, True, True])
    _, cancelled = await _run(should_cancel=lambda: next(calls))
    assert cancelled
    assert p.processed == [['img-0', 'img-1']]


@pytest.mark.asyncio
async def test_the_embed_pass_preprocess_runs_on_the_pool_not_the_loop(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    doc = item('c0', image_id='img-0', image_path=str(path), bbox_norm=(0.1, 0.1, 0.6, 0.6))
    target = EmbedTarget(image_id='img-0', image_path=str(path), items=[('c0', doc)])
    fake = make_fake([doc], [{'image_id': 'img-0', 'image_path': str(path)}])

    seen: list[str] = []
    real = reprocess_embed.crop_pil

    def spy(img: Any, box: Any) -> Any:
        seen.append(threading.current_thread().name)
        return real(img, box)

    monkeypatch.setattr(reprocess_embed, 'crop_pil', spy)

    class PE:
        async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
            return np.ones((len(crops), 3), np.float32)

        async def embed_whole_frame_image(self, img: Any) -> np.ndarray:  # noqa: ARG002
            return np.ones(3, np.float32)

    await reembed_items(fake, PE(), [target], parts=frozenset({'crop'}))
    assert seen
    assert all(name.startswith('op-cpu-') for name in seen), seen


@pytest.mark.asyncio
async def test_a_cancel_waits_for_the_chunks_already_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)
    calls = iter([False, False, True, True])
    task = asyncio.create_task(_run(should_cancel=lambda: next(calls)))
    for _ in range(50):
        await asyncio.sleep(0)
    assert not task.done()  # chunk 1 is held and the pass waits for it, as for any started chunk
    p.release.set()
    (res,), cancelled = await task
    assert cancelled
    assert p.processed == [['img-0', 'img-1'], ['img-2', 'img-3']]
    assert res.queued == 4


@pytest.mark.asyncio
async def test_cancelling_the_job_cancels_the_chunks_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)
    task = asyncio.create_task(_run())
    for _ in range(50):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert pending == []  # no chunk is left running behind a cancelled job
    assert p.processed == [['img-0', 'img-1'], ['img-2', 'img-3']]


@pytest.mark.asyncio
async def test_chunks_in_flight_share_one_image_bound(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = servable_root(tmp_path, monkeypatch)
    targets = []
    for i in range(6):
        path = root / f'im{i}.jpg'
        path.write_bytes(jpeg_bytes(seed=i))
        doc = item(
            f'c{i}', image_id=f'img-{i}', image_path=str(path), bbox_norm=(0.1, 0.1, 0.6, 0.6)
        )
        targets.append(
            EmbedTarget(image_id=f'img-{i}', image_path=str(path), items=[(f'c{i}', doc)])
        )
    fake = make_fake(
        [t.items[0][1] for t in targets],
        [{'image_id': t.image_id, 'image_path': t.image_path} for t in targets],
    )

    class PE:
        inflight = peak = 0

        async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
            PE.inflight += 1
            PE.peak = max(PE.peak, PE.inflight)
            await asyncio.sleep(0.02)
            PE.inflight -= 1
            return np.ones((len(crops), 3), np.float32)

    gate = asyncio.Semaphore(2)
    await asyncio.gather(
        reembed_items(fake, PE(), targets[:3], parts=frozenset({'crop'}), gate=gate),
        reembed_items(fake, PE(), targets[3:], parts=frozenset({'crop'}), gate=gate),
    )
    assert PE.peak == 2  # six images, two calls, one bound of two
