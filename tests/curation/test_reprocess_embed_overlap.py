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
    service = SimpleNamespace(pe_encoder=object())
    return await reprocess_images.process_images(
        object(),  # type: ignore[arg-type]
        service,  # type: ignore[arg-type]
        scopes=['embed'],
        image_ids=IDS,
        embed=EmbedSpec(parts=['crop']),
        **kw,
    )


@pytest.mark.asyncio
async def test_the_next_chunk_is_read_while_this_one_is_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pass(monkeypatch)
    task = asyncio.create_task(_run())
    for _ in range(50):
        await asyncio.sleep(0)
    assert ('read', ('img-2', 'img-3')) in p.events  # read ahead
    assert p.processed == [['img-0', 'img-1']]  # while chunk 1 is still being embedded
    p.release.set()
    (res,), cancelled = await task
    assert not cancelled
    assert (res.queued, res.failed) == (5, 0)


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
