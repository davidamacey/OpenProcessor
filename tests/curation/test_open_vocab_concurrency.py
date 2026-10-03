"""The full-image pass keeps the segmenter busy: images run concurrently up to
``OP_OPEN_VOCAB_CONCURRENCY``, and the reprocess job reports progress per image."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import FakeSegmenter, StatefulRegistry, cand, ingested_world
from curation.reprocess_fixtures import docs
from src.services.config_store.store import StoredConfig, get_config_store, reset_config_stores
from src.services.curation.reprocess_images import process_images


if TYPE_CHECKING:
    from pathlib import Path


class OverlapSegmenter(FakeSegmenter):
    """Records the most calls in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, jpeg: bytes, prompt: str, **kw: Any) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)
            return await super().__call__(jpeg, prompt, **kw)
        finally:
            self.in_flight -= 1


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_config_stores()
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)
    yield
    reset_config_stores()


def _activate(targets: list[dict[str, Any]]) -> None:
    stored = StoredConfig(kind='open_vocab_set', name='s', revision=1, body={'targets': targets})
    get_config_store().apply_local(
        config_revision=0,
        open_vocab_set=stored,
        active_open_vocab=('s', 1),
        active_open_vocab_body=stored,
    )


@pytest.mark.parametrize(('limit', 'peak'), [('1', 1), ('3', 3)])
@pytest.mark.asyncio
async def test_images_overlap_up_to_the_concurrency_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, peak: int
) -> None:
    fake, service, ids = await ingested_world(tmp_path, monkeypatch, n=6)
    monkeypatch.setenv('OP_OPEN_VOCAB_CONCURRENCY', limit)
    _activate([{'prompt': 'cone', 'class_name': 'cone'}])
    seg = OverlapSegmenter()
    seg.default = [cand()]

    results, cancelled = await process_images(
        fake, service, scopes=['open_vocab'], image_ids=ids, segment=seg
    )

    assert not cancelled
    assert results[0].queued == 6
    assert seg.peak == peak
    assert len({d['image_id'] for d in docs(fake).values()}) >= 1


@pytest.mark.asyncio
async def test_progress_advances_per_image_not_per_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await ingested_world(tmp_path, monkeypatch, n=5)
    monkeypatch.setenv('OP_OPEN_VOCAB_CONCURRENCY', '1')
    _activate([{'prompt': 'cone', 'class_name': 'cone'}])
    seg = FakeSegmenter()
    seg.default = [cand()]
    seen: list[tuple[int, int]] = []

    await process_images(
        fake,
        service,
        scopes=['open_vocab'],
        image_ids=ids,
        segment=seg,
        on_progress=lambda done, failed: seen.append((done, failed)),
    )

    assert [d for d, _ in seen][:5] == [1, 2, 3, 4, 5]  # CHUNK is 20: one chunk, five ticks
