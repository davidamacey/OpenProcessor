"""Wave 4: the full-image SAM 3 pass writes normal items, idempotently, under
the lock rule, and an outage writes nothing."""

from __future__ import annotations

import copy
import io
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from curation.reprocess_fixtures import (
    FakeTriton,
    docs,
    images_index,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.curation.open_vocab_run import run_open_vocab_image
from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.open_vocab_set import decode_open_vocab_set
from src.services.detection.segmenter_http import SegmenterCallError


if TYPE_CHECKING:
    from pathlib import Path

    from curation.query_fakes import QueryFakeOpenSearch
    from src.services.curation.ingest import CurationIngestService

BOX = (0.1, 0.2, 0.3, 0.5)
POLY = ((0.1, 0.2), (0.3, 0.2), (0.3, 0.5))


class _Entry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id, self.class_name, self.deprecated = class_id, class_name, False


class _File:
    def __init__(self, classes: list[_Entry]) -> None:
        self.classes = classes


class StatefulRegistry:
    def __init__(self) -> None:
        self.entries = [_Entry(0, 'gadget')]
        self.added: list[str] = []

    def load(self) -> _File:
        return _File(self.entries)

    def add_class(self, name: str, group: str = 'unknown', notes: str = '') -> int:  # noqa: ARG002
        self.added.append(name)
        self.entries.append(_Entry(len(self.entries), name))
        return len(self.entries) - 1


class FakeSegmenter:
    def __init__(self) -> None:
        self.by_prompt: dict[str, list[RegionCandidate]] = {}
        self.calls: list[dict[str, Any]] = []
        self.down = False

    async def __call__(
        self,
        jpeg: bytes,
        prompt: str,
        *,
        min_score: float | None,
        max_candidates: int,
        return_masks: bool,
    ) -> list[RegionCandidate]:
        self.calls.append(
            {
                'prompt': prompt,
                'min_score': min_score,
                'max_candidates': max_candidates,
                'return_masks': return_masks,
                'size': Image.open(io.BytesIO(jpeg)).size,
            }
        )
        if self.down:
            raise SegmenterCallError('segmenter call failed: down')
        return list(self.by_prompt.get(prompt, []))


def _cand(box: tuple[float, float, float, float] = BOX, score: float = 0.9) -> RegionCandidate:
    return RegionCandidate(bbox_norm=box, score=score, source='sam3', mask_polygon=POLY)


def _set(**kw: Any) -> Any:
    body: dict[str, Any] = {
        'targets': [{'prompt': 'traffic cone', 'class_name': 'cone'}],
        'max_enabled_targets': 8,
    }
    body.update(kw)
    return decode_open_vocab_set('street', body)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> StatefulRegistry:
    reg = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: reg)
    return reg


async def _world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int] = (400, 300)
) -> tuple[QueryFakeOpenSearch, CurationIngestService, str, dict[str, Any]]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes(size))
    fake = make_fake([])
    service = make_service(fake, FakeTriton([]))
    res = await service.ingest_one(path.read_bytes(), str(path))
    assert res.status == 'success'
    assert docs(fake) == {}
    return fake, service, res.image_id, fake.docs(images_index())[res.image_id]


async def _run(fake: Any, service: Any, image_id: str, image_doc: Any, ov: Any, seg: Any) -> Any:
    return await run_open_vocab_image(
        fake, service, image_id, image_doc, ov, revision=3, segment=seg
    )


@pytest.mark.asyncio
async def test_a_hit_becomes_a_normal_item_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]

    result = await _run(fake, service, image_id, image_doc, _set(), seg)

    assert (result.calls, result.hits, result.written) == (1, 1, 1)
    (doc,) = docs(fake).values()
    assert doc['image_id'] == image_id
    assert doc['bbox_norm'] == pytest.approx(list(BOX), abs=1e-6)
    assert doc['class_name'] == 'cone'
    assert doc['class_id'] == registry.entries[-1].class_id
    assert doc['cluster_id'] == doc['class_id']
    assert doc['class_source'] == 'open_vocab_proposal'
    assert doc['class_detector'] == 'sam3'
    assert (doc['source_prompt'], doc['open_vocab_set'], doc['open_vocab_revision']) == (
        'traffic cone',
        'street',
        3,
    )
    assert doc['mask_polygon'] == [list(p) for p in POLY]
    assert doc['confidence'] == pytest.approx(0.9)
    assert doc['class_validated'] is False
    assert registry.added == ['cone']
    assert seg.calls[0]['return_masks'] is True
    assert seg.calls[0]['min_score'] == 0.5


@pytest.mark.asyncio
async def test_a_second_run_does_not_duplicate_and_creates_the_class_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]
    await _run(fake, service, image_id, image_doc, _set(), seg)
    ids = sorted(docs(fake))

    again = await _run(fake, service, image_id, image_doc, _set(), seg)

    assert sorted(docs(fake)) == ids
    assert (again.written, again.removed) == (1, 0)
    assert registry.added == ['cone']


@pytest.mark.asyncio
async def test_own_output_a_rerun_no_longer_produces_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand(BOX)]
    await _run(fake, service, image_id, image_doc, _set(), seg)
    (old_id,) = docs(fake)
    seg.by_prompt['traffic cone'] = [_cand((0.6, 0.6, 0.8, 0.9))]

    result = await _run(fake, service, image_id, image_doc, _set(), seg)

    assert old_id not in docs(fake)
    assert len(docs(fake)) == 1
    assert (result.written, result.removed) == (1, 1)


@pytest.mark.asyncio
async def test_a_locked_item_is_never_overwritten_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    locked = {
        'crop_id': 'human-1',
        'image_id': image_id,
        'bbox_norm': list(BOX),
        'class_id': 0,
        'class_name': 'gadget',
        'class_source': 'human',
        'label_source': 'human',
        'class_validated': True,
    }
    docs(fake)['human-1'] = locked
    before = copy.deepcopy(locked)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]

    result = await _run(fake, service, image_id, image_doc, _set(), seg)

    assert list(docs(fake)) == ['human-1']
    assert docs(fake)['human-1'] == before
    assert (result.written, result.dropped['skipped_locked'], result.locked_untouched) == (0, 1, 1)


@pytest.mark.asyncio
async def test_a_same_class_machine_item_wins_and_a_different_class_one_coexists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    machine = {
        'crop_id': 'm-1',
        'image_id': image_id,
        'bbox_norm': [0.1, 0.2, 0.3, 0.49],
        'class_source': 'primary_model',
        'class_name': 'cone',
    }
    docs(fake)['m-1'] = machine
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]
    same = await _run(fake, service, image_id, image_doc, _set(), seg)
    assert (same.written, same.dropped['agree_existing']) == (0, 1)
    assert list(docs(fake)) == ['m-1']

    other = _set(targets=[{'prompt': 'traffic cone', 'class_name': 'barrel'}])
    await _run(fake, service, image_id, image_doc, other, seg)
    assert len(docs(fake)) == 2


@pytest.mark.asyncio
async def test_a_hit_on_exactly_the_box_of_another_class_never_overwrites_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    from src.services.detection.geometry import crop_id, stored_bbox_norm

    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    cid = crop_id(image_id, stored_bbox_norm(BOX, 400, 300))
    machine = {
        'crop_id': cid,
        'image_id': image_id,
        'bbox_norm': list(BOX),
        'class_source': 'primary_model',
        'class_name': 'barrel',
    }
    docs(fake)[cid] = machine
    before = copy.deepcopy(machine)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]

    result = await _run(fake, service, image_id, image_doc, _set(), seg)

    assert docs(fake) == {cid: before}
    assert (result.written, result.dropped['id_collision']) == (0, 1)


@pytest.mark.asyncio
async def test_an_outage_writes_and_deletes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]
    await _run(fake, service, image_id, image_doc, _set(), seg)
    before = copy.deepcopy(docs(fake))
    seg.down = True

    with pytest.raises(SegmenterCallError):
        await _run(fake, service, image_id, image_doc, _set(), seg)

    assert docs(fake) == before
    assert registry.added == ['cone']


@pytest.mark.asyncio
async def test_a_failure_of_one_target_aborts_the_whole_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]
    original = seg.__call__

    async def flaky(jpeg: bytes, prompt: str, **kw: Any) -> list[RegionCandidate]:
        if prompt == 'cup':
            raise SegmenterCallError('segmenter call failed: boom')
        return await original(jpeg, prompt, **kw)

    ov = _set(
        targets=[
            {'prompt': 'traffic cone', 'class_name': 'cone'},
            {'prompt': 'cup', 'class_name': 'cup'},
        ]
    )
    with pytest.raises(SegmenterCallError):
        await run_open_vocab_image(
            fake, service, image_id, image_doc, ov, revision=1, segment=flaky
        )
    assert docs(fake) == {}
    assert registry.added == []


@pytest.mark.asyncio
async def test_discovery_target_is_stored_as_a_proposal_named_by_its_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    seg.by_prompt['barrel'] = [_cand()]
    ov = _set(targets=[{'prompt': 'barrel'}])

    await _run(fake, service, image_id, image_doc, ov, seg)

    (doc,) = docs(fake).values()
    assert doc['proposal_name'] == 'barrel'
    assert 'class_name' not in doc
    assert 'class_id' not in doc
    assert doc['class_source'] == 'open_vocab_proposal'
    assert doc['source_prompt'] == 'barrel'
    assert registry.added == []


@pytest.mark.asyncio
async def test_a_downscaled_call_still_yields_source_frame_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch, size=(400, 300))
    seg = FakeSegmenter()
    seg.by_prompt['traffic cone'] = [_cand()]

    await _run(fake, service, image_id, image_doc, _set(image_max_side=256), seg)

    assert seg.calls[0]['size'] == (256, 192)
    (doc,) = docs(fake).values()
    assert doc['bbox_norm'] == pytest.approx(list(BOX), abs=1e-6)


@pytest.mark.asyncio
async def test_a_small_image_is_never_upscaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch, size=(400, 300))
    seg = FakeSegmenter()
    await _run(fake, service, image_id, image_doc, _set(image_max_side=4096), seg)
    assert seg.calls[0]['size'] == (400, 300)


@pytest.mark.asyncio
async def test_an_unservable_image_fails_before_any_segmenter_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: StatefulRegistry
) -> None:
    fake, service, image_id, image_doc = await _world(tmp_path, monkeypatch)
    seg = FakeSegmenter()
    with pytest.raises(ValueError, match='not under a configured source root'):
        await _run(fake, service, image_id, {**image_doc, 'image_path': '/etc/passwd'}, _set(), seg)
    assert seg.calls == []
