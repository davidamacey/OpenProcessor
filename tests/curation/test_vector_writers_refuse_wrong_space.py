"""A project whose item or box space is of another model gets no crop vector of the crop encoder.

The writers: ingest (engine and host), a drawn object and a geometry edit (inline embed), the
embed reprocess (and with it the backfill, the auto-label embed stage and the region refresh) and
the detection worker's box vectors. Each asks the one check (``require_encoder_for``) before it
encodes or writes; the refusal carries the remedy (no migration: re-ingest into a new project).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from _embedding_registry_fake import new_space

from curation.reprocess_fixtures import docs, item, jpeg_bytes, make_fake, servable_root
from scripts.curation.worker import bulk_writer
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.domain.embedding_state import DEFERRED
from src.services.curation import ingest_index
from src.services.curation.ingest_finish import PhotoJob, finish_photos
from src.services.curation.item_embed_inline import embed_item_inline
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.1, 0.6, 0.6)


class CountingPE:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:
        import numpy as np

        self.calls += 1
        return np.ones((len(crops), 3), dtype=np.float32)

    async def embed_whole_frame_image(self, img: Any) -> Any:
        import numpy as np

        self.calls += 1
        return np.ones(3, dtype=np.float32)


@pytest.fixture
def old_item_space(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.ITEM, 'older_encoder'), primary=True)


@pytest.fixture
def old_box_space(embedding_registry) -> None:
    embedding_registry.add(new_space(Target.BOX, 'older_encoder'), primary=True)


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[EmbedTarget]]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'im.jpg'
    path.write_bytes(jpeg_bytes(seed=1))
    doc = item(
        'c0', image_id='img-0', image_path=str(path), bbox_norm=BOX, class_id=1, class_name='x'
    )
    target = EmbedTarget(image_id='img-0', image_path=str(path), items=[('c0', doc)])
    return make_fake([doc], [{'image_id': 'img-0', 'image_path': str(path)}]), [target]


@pytest.mark.asyncio
async def test_the_embed_pass_writes_no_crop_vector_into_an_old_item_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_item_space
) -> None:
    fake, targets = _world(tmp_path, monkeypatch)
    pe = CountingPE()
    with pytest.raises(EmbeddingSpaceError) as err:
        await reembed_items(fake, pe, targets, parts=frozenset({'crop'}))
    assert 'older_encoder_image' in err.value.detail
    assert pe.calls == 0  # refused before any GPU work
    assert not [k for k in docs(fake)['c0'] if 'embedding' in k and k != 'embedding_state']


@pytest.mark.asyncio
async def test_the_embed_pass_writes_no_region_vector_into_an_old_box_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_box_space
) -> None:
    fake, targets = _world(tmp_path, monkeypatch)
    pe = CountingPE()
    with pytest.raises(EmbeddingSpaceError):
        await reembed_items(fake, pe, targets, parts=frozenset({'region'}))
    assert pe.calls == 0


@pytest.mark.asyncio
async def test_an_old_box_space_does_not_stop_the_item_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_box_space
) -> None:
    fake, targets = _world(tmp_path, monkeypatch)
    pe = CountingPE()
    counts = await reembed_items(fake, pe, targets, parts=frozenset({'crop'}))
    assert counts['crop_written'] == 1


@pytest.mark.asyncio
async def test_a_drawn_box_is_kept_and_served_deferred_with_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_item_space
) -> None:
    fake, _ = _world(tmp_path, monkeypatch)
    pe = CountingPE()
    outcome = await embed_item_inline(fake, pe, 'c0')
    assert outcome.state == DEFERRED
    assert outcome.message is not None
    assert 'older_encoder_image' in outcome.message
    assert 'new project' in outcome.message
    assert pe.calls == 0


@pytest.mark.asyncio
async def test_ingest_refuses_before_the_engine_or_the_host_is_asked(old_item_space) -> None:
    service = SimpleNamespace()  # touched by nothing: the refusal comes first
    with pytest.raises(EmbeddingSpaceError):
        await finish_photos(service, [PhotoJob(b'x', '/p/a.jpg')])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_the_single_image_index_path_refuses_before_it_embeds(old_item_space) -> None:
    service = SimpleNamespace()
    with pytest.raises(EmbeddingSpaceError):
        await ingest_index.index_items(service, SimpleNamespace(), [])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_the_worker_writes_no_box_vector_into_an_old_box_space(
    monkeypatch: pytest.MonkeyPatch, old_box_space
) -> None:
    written: list[Any] = []
    logged: list[tuple[str, dict[str, Any]]] = []

    async def record(_os: Any, *, index: str, by_crop: dict[str, Any]) -> None:
        written.append(by_crop)

    class Log:
        def error(self, event: str, **kw: Any) -> None:
            logged.append((event, kw))

        def warning(self, event: str, **kw: Any) -> None:
            logged.append((event, kw))

    monkeypatch.setattr(bulk_writer, 'write_box_embeddings', record)
    monkeypatch.setattr(bulk_writer, 'logger', Log())
    task = SimpleNamespace(crop_id='c0', box_embedding_entries=[{'box_id': 'b1'}])
    await bulk_writer._write_box_embeddings(object(), [cast('Any', task)])  # type: ignore[arg-type]
    assert written == []
    assert [e for e, _ in logged] == ['region_box_embeddings_refused']
    assert 'older_encoder_image' in logged[0][1]['error']


@pytest.mark.asyncio
async def test_the_worker_writes_box_vectors_of_a_matching_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[Any] = []

    async def record(_os: Any, *, index: str, by_crop: dict[str, Any]) -> None:
        written.append(by_crop)

    monkeypatch.setattr(bulk_writer, 'write_box_embeddings', record)
    task = SimpleNamespace(crop_id='c0', box_embedding_entries=[{'box_id': 'b1'}])
    await bulk_writer._write_box_embeddings(object(), [cast('Any', task)])  # type: ignore[arg-type]
    assert written == [{'c0': [{'box_id': 'b1'}]}]
