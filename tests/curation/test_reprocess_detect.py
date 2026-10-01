"""The ``detect`` scope: re-run the ingest detectors on a stored image and
merge under the lock rule (a proposal never overwrites a human or imported
label; stale unlocked machine items go; new ones arrive)."""

from __future__ import annotations

import copy
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
from src.services.curation.reprocess import apply_reprocess, plan_reprocess
from src.services.curation.reprocess_models import ReprocessRequest, ReprocessTargets


if TYPE_CHECKING:
    from pathlib import Path

    from curation.query_fakes import QueryFakeOpenSearch
    from src.services.curation.ingest import CurationIngestService

D1 = (0.05, 0.05, 0.6, 0.6, 0.9, 1)
D2 = (0.65, 0.65, 0.95, 0.95, 0.9, 0)
HUMAN = {
    'class_source': 'human',
    'label_source': 'human',
    'class_id': 0,
    'class_name': 'gadget',
    'class_validated': True,
}


async def _world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detections: list[tuple[Any, ...]]
) -> tuple[QueryFakeOpenSearch, FakeTriton, CurationIngestService, str, list[str]]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = make_fake([])
    triton = FakeTriton(detections)
    service = make_service(fake, triton)
    res = await service.ingest_one(path.read_bytes(), str(path))
    assert res.status == 'success'
    return fake, triton, service, res.image_id, sorted(docs(fake))


def _req(image_id: str, *, dry_run: bool = False) -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(image_ids=[image_id]), scopes=['detect'], dry_run=dry_run
    )


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


@pytest.mark.asyncio
async def test_a_proposal_matching_a_locked_item_only_notes_the_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    docs(fake)[crop_id].update(HUMAN)
    before = copy.deepcopy(docs(fake)[crop_id])

    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))

    detect = resp.scopes[0]
    assert (detect.queued, detect.failed, detect.locked_skipped) == (1, 0, 1)
    assert detect.detail['merged'] == 1
    assert list(docs(fake)) == [crop_id]
    after = docs(fake)[crop_id]
    assert after.pop('proposal_chain') == ['primary:match']
    assert after == before  # class, provenance, holdout: all untouched


@pytest.mark.asyncio
async def test_an_unlocked_machine_item_is_refreshed_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    docs(fake)[crop_id].update({'class_id': 0, 'class_name': 'gadget'})

    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))

    assert resp.scopes[0].detail['refreshed'] == 1
    assert list(docs(fake)) == [crop_id]  # same crop_id: refreshed, not duplicated
    assert docs(fake)[crop_id]['class_name'] == 'widget'


@pytest.mark.asyncio
async def test_a_stale_machine_item_is_deleted_and_a_locked_one_never_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    keep = {**copy.deepcopy(docs(fake)[crop_id]), 'crop_id': 'locked', **HUMAN}
    keep['bbox_norm'] = [0.7, 0.7, 0.9, 0.9]
    docs(fake)['locked'] = keep
    before_locked = copy.deepcopy(keep)
    triton.detections = []  # the detector no longer finds anything

    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))

    detect = resp.scopes[0]
    assert detect.detail['removed'] == 1
    assert list(docs(fake)) == ['locked']
    assert docs(fake)['locked'] == before_locked


@pytest.mark.asyncio
async def test_an_unmatched_proposal_becomes_a_new_item_and_inherits_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    docs(fake)[crop_id].update(HUMAN)
    fake.docs(images_index())[image_id]['dataset_split'] = 'test'
    triton.detections = [D1, D2]

    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))

    assert resp.scopes[0].detail['created'] == 1
    assert len(docs(fake)) == 2
    new = next(d for cid, d in docs(fake).items() if cid != crop_id)
    assert new['class_validated'] is False
    assert new['dataset_split'] == 'test'
    assert docs(fake)[crop_id]['class_source'] == 'human'


@pytest.mark.asyncio
async def test_a_moved_unlocked_item_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _triton, service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    moved = copy.deepcopy(docs(fake)[crop_id])
    moved['crop_id'] = 'moved'
    x1, y1, x2, y2 = moved['bbox_norm']
    moved['bbox_norm'] = [x1 + 0.02, y1 + 0.02, x2 + 0.02, y2 + 0.02]
    docs(fake).clear()
    docs(fake)['moved'] = moved

    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))

    assert resp.scopes[0].detail['replaced'] == 1
    assert list(docs(fake)) == [crop_id]


@pytest.mark.asyncio
async def test_detect_counts_locked_items_in_the_dry_run_and_runs_no_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, triton, _service, image_id, (crop_id,) = await _world(tmp_path, monkeypatch, [D1])
    docs(fake)[crop_id].update(HUMAN)
    calls = triton.calls
    plan = await plan_reprocess(fake, _req(image_id, dry_run=True))
    assert (plan.results[0].selected, plan.results[0].locked_skipped) == (1, 1)
    assert triton.calls == calls


@pytest.mark.asyncio
async def test_an_unservable_stored_path_is_never_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, triton, service, image_id, _ids = await _world(tmp_path, monkeypatch, [D1])
    servable_root(tmp_path / 'elsewhere', monkeypatch)  # the stored path is now outside every root
    calls = triton.calls
    resp = await apply_reprocess(fake, _req(image_id), service_factory=_factory(service))
    assert (resp.scopes[0].failed, resp.scopes[0].queued) == (1, 0)
    assert triton.calls == calls
