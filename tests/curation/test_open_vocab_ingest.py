"""Wave 5: the ingest-time opt-in -- off by default, never inline, durable
``pending`` marker, outage leaves the image pending."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.reprocess_fixtures import (
    FakeTriton,
    docs,
    images_index,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.config_store.store import StoredConfig, get_config_store, reset_config_stores
from src.services.curation.open_vocab_ingest import (
    schedule_open_vocab_after_ingest,
    wait_for_scheduled,
)
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.segmenter_http import SegmenterCallError


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.2, 0.3, 0.5)


class _Entry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id, self.class_name, self.deprecated = class_id, class_name, False


class _Registry:
    def __init__(self) -> None:
        self.entries: list[_Entry] = []

    def load(self) -> Any:
        return type('F', (), {'classes': self.entries})()

    def add_class(self, name: str, group: str = '', notes: str = '') -> int:  # noqa: ARG002
        self.entries.append(_Entry(len(self.entries), name))
        return len(self.entries) - 1


class _Seg:
    def __init__(self) -> None:
        self.calls = 0
        self.down = False

    async def __call__(self, jpeg: bytes, prompt: str, **kw: Any) -> list[RegionCandidate]:  # noqa: ARG002
        self.calls += 1
        if self.down:
            raise SegmenterCallError('segmenter call failed: down')
        return [RegionCandidate(bbox_norm=BOX, score=0.9, source='sam3')]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_config_stores()
    monkeypatch.setattr(
        'src.services.curation.open_vocab_run.get_class_registry', lambda: _Registry()
    )
    yield
    reset_config_stores()


def _activate(*, run_on_ingest: bool) -> None:
    body = {'targets': [{'prompt': 'cone', 'class_name': 'cone'}], 'run_on_ingest': run_on_ingest}
    stored = StoredConfig(kind='open_vocab_set', name='street', revision=2, body=body)
    get_config_store().apply_local(
        config_revision=1,
        open_vocab_set=stored,
        active_open_vocab=('street', 2),
        active_open_vocab_body=stored,
    )


async def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, str]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = make_fake([])
    service = make_service(fake, FakeTriton([]))
    res = await service.ingest_one(path.read_bytes(), str(path))
    return fake, service, res.image_id


def _status(fake: Any, image_id: str) -> str | None:
    return fake.docs(images_index())[image_id].get('open_vocab_status')


@pytest.mark.asyncio
async def test_off_by_default_nothing_is_stamped_or_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, image_id = await _world(tmp_path, monkeypatch)
    seg = _Seg()
    _activate(run_on_ingest=False)

    assert await schedule_open_vocab_after_ingest(fake, service, [image_id], segment=seg) is False
    await wait_for_scheduled()
    assert (seg.calls, _status(fake, image_id), docs(fake)) == (0, None, {})


@pytest.mark.asyncio
async def test_no_active_set_means_no_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, service, image_id = await _world(tmp_path, monkeypatch)
    seg = _Seg()
    assert await schedule_open_vocab_after_ingest(fake, service, [image_id], segment=seg) is False
    assert seg.calls == 0


@pytest.mark.asyncio
async def test_opted_in_images_are_stamped_pending_then_processed_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, image_id = await _world(tmp_path, monkeypatch)
    seg = _Seg()
    _activate(run_on_ingest=True)

    assert await schedule_open_vocab_after_ingest(fake, service, [image_id], segment=seg) is True
    # Returned before the work ran: the stamp is the durable record.
    assert _status(fake, image_id) == 'pending'
    assert seg.calls == 0

    await wait_for_scheduled()
    assert _status(fake, image_id) == 'done'
    assert seg.calls == 1
    (item,) = docs(fake).values()
    assert (item['open_vocab_set'], item['open_vocab_revision']) == ('street', 2)


@pytest.mark.asyncio
async def test_an_outage_leaves_the_image_pending_for_a_later_reprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, image_id = await _world(tmp_path, monkeypatch)
    seg = _Seg()
    seg.down = True
    _activate(run_on_ingest=True)

    await schedule_open_vocab_after_ingest(fake, service, [image_id], segment=seg)
    await wait_for_scheduled()

    assert _status(fake, image_id) == 'pending'
    assert docs(fake) == {}


@pytest.mark.asyncio
async def test_the_ingest_routes_never_fail_because_scheduling_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, image_id = await _world(tmp_path, monkeypatch)
    _activate(run_on_ingest=True)

    async def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError('opensearch down')

    monkeypatch.setattr('src.services.curation.open_vocab_ingest.current_active_set', boom)
    assert (
        await schedule_open_vocab_after_ingest(fake, service, [image_id], segment=_Seg()) is False
    )
